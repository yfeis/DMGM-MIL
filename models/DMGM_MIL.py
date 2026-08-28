import numpy as np
from timm.models.layers import DropPath

try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba.mamba_ssm import BiMamba
from timm.models.vision_transformer import Mlp

## Mamba module
class MambaBlock(nn.Module):
    def __init__(self, in_dim, layer=1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.layers = nn.ModuleList()

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(512),
                    BiMamba(
                        d_model=512,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    ),
                )
            )
    def forward(self, x):
        h = x
        for norm, mamba in self.layers:
            h = h + mamba(norm(h))
        h = self.norm(h)
        return h


class LinearAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        h = self.norm(x)
        scores = self.attn(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        summary = (x * weights.unsqueeze(-1)).sum(dim=1)

        return summary, weights

class ChunkMamba(nn.Module):
    def __init__(
            self,
            d_model,
            group_size=128,
            anchor_ratio=0.1,
    ):
        super().__init__()

        self.dim = d_model
        self.group_size = group_size
        self.anchor_ratio = anchor_ratio

        self.pad_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pad_token, std=0.02)

        # chunk
        self.intra_mamba = MambaBlock(in_dim=d_model, layer=2)
        self.pool = LinearAttentionPooling(d_model=d_model, hidden_dim=128)


        self.importance_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.seg_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1)
        )
        self.seg_norm = nn.LayerNorm(d_model)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.write_proj = nn.Linear(d_model, d_model)
        self.memory_decay = nn.Parameter(torch.tensor(0.98))  #  sigmoid  0.727  0~1

        self.write_memory_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.memory_fuse_logit = nn.Parameter(torch.tensor(-1.3863))  #  sigmoid 0.2   0~1
        self.fuse_proj = nn.Linear(d_model, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.chunk_mamba = MambaBlock(in_dim=d_model, layer=1)

    def extract_summary_state(self, h_chunks, mask):
        B, Ng, G, D = h_chunks.shape
        h = rearrange(h_chunks, "b ng g d -> (b ng) g d")
        m = rearrange(mask, "b ng g -> (b ng) g")
        summary, _ = self.pool(h, m)
        return rearrange(summary, "(b ng) d -> b ng d", b=B, ng=Ng)

    def compute_importance_score(self, state, mask):
        logit = self.importance_mlp(state).squeeze(-1)
        logit = logit.masked_fill(~mask, -1e4)
        score = torch.sigmoid(logit)
        score = score.masked_fill(~mask, 0.0)
        return logit, score

    def select_important_mask_batch1(self, score, mask):
        score = score[0]
        mask = mask[0]

        valid_idx = torch.where(mask)[0]
        valid_num = valid_idx.numel()

        important_mask = torch.zeros_like(mask, dtype=torch.bool)

        if valid_num == 0:
            return important_mask

        k = max(1, int(valid_num * self.anchor_ratio))

        valid_score = score[valid_idx]
        top_local_idx = torch.topk(valid_score, k=k).indices
        top_idx = valid_idx[top_local_idx]

        important_mask[top_idx] = True

        # smooth
        important_mask = F.max_pool1d(
            important_mask.float().view(1, 1, -1),
            kernel_size=3,
            stride=1,
            padding=1
        ).view(-1).bool()

        important_mask = important_mask & mask

        return important_mask

    def build_segments(self, state, score, mask):
        valid_len = int(mask[0].sum().detach().cpu())

        if valid_len == 0:
            return []

        important_mask = self.select_important_mask_batch1(score, mask)
        important_cpu = important_mask[:valid_len].detach().cpu()

        state_1 = state[0, :valid_len]
        mask_1 = mask[0, :valid_len]

        segments = []
        start = 0
        current_type = bool(important_cpu[0])

        for i in range(1, valid_len):
            next_type = bool(important_cpu[i])
            if next_type != current_type:
                segments.append({
                    "x": state_1[start:i].unsqueeze(0),
                    "mask": mask_1[start:i].unsqueeze(0),
                    "is_important": current_type,
                })
                start = i
                current_type = next_type

        segments.append({
            "x": state_1[start:valid_len].unsqueeze(0),
            "mask": mask_1[start:valid_len].unsqueeze(0),
            "is_important": current_type,
        })

        return segments

    def segment_interaction(self, seg_x, seg_mask):
        y = self.seg_conv(seg_x.transpose(1, 2)).transpose(1, 2)
        seg_x = self.seg_norm(seg_x + y)
        seg_x = seg_x * seg_mask.unsqueeze(-1).float()
        return seg_x

    def init_memory(self, state, score):
        B, N, D = state.shape
        k = max(1, int(N * self.anchor_ratio))
        top_idx = torch.topk(score, k=k, dim=-1).indices
        memory = torch.gather(
            state,
            dim=1,
            index=top_idx.unsqueeze(-1).expand(-1, -1, D)
        )
        return memory  # [1,K,D]

    def read_memory(self, seg_x, memory):
        q = self.q(seg_x)
        k = self.k(memory)
        v = self.v(memory)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        return context

    def write_memory(self, memory, seg_vec, dyn_score):
        sim = torch.matmul(
            seg_vec.unsqueeze(1),
            memory.transpose(-1, -2)
        ).squeeze(1)/ math.sqrt(self.dim)
        attn = torch.softmax(sim, dim=-1)
        delta = self.write_proj(seg_vec)

        write_strength = dyn_score
        decay = torch.sigmoid(self.memory_decay)
        memory = decay * memory + write_strength.unsqueeze(-1).unsqueeze(-1) * attn.unsqueeze(-1) * delta.unsqueeze(1)
        return memory

    #  segment memory
    def process_segments_with_memory(self, state, score, mask):
        segments = self.build_segments(state, score, mask)
        memory = self.init_memory(state, score)
        dyn_scores = []

        for seg in segments:
            seg_x = seg["x"]
            seg_mask = seg["mask"]
            seg_x = self.segment_interaction(seg_x, seg_mask)
            context = self.read_memory(seg_x, memory)
            seg_x = seg_x + context
            seg_vec = (seg_x * seg_mask.unsqueeze(-1).float()).sum(dim=1)
            seg_vec = seg_vec / seg_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            dyn_logit = self.write_memory_gate(seg_vec)
            dyn_score = torch.sigmoid(dyn_logit).squeeze(-1)
            dyn_scores.append(dyn_score)
            memory = self.write_memory(memory, seg_vec, dyn_score)
        return memory

    def fuse_memory_to_state(self, state, memory, mask):
        q = self.q(state)
        k = self.k(memory)
        v = self.v(memory)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        gate = torch.sigmoid(self.memory_fuse_logit)
        fused = self.fuse_norm(state + gate * self.fuse_proj(context))
        fused = fused * mask.unsqueeze(-1).float()
        return fused

    def forward(self, x):
        B, L, D = x.shape
        assert B == 1

        G = self.group_size
        pad_len = (G - (L % G)) % G
        token_mask = torch.ones(B, L, dtype=torch.bool, device=x.device)

        if pad_len > 0:
            pad = self.pad_token.view(1, 1, D).expand(B, pad_len, D)
            x = torch.cat([x, pad], dim=1)
            token_mask = torch.cat([
                token_mask,
                torch.zeros(B, pad_len, dtype=torch.bool, device=x.device)
            ], dim=1)

        # chunk
        x_chunks = rearrange(x, "b (ng g) d -> b ng g d", g=G)
        mask_chunks = rearrange(token_mask, "b (ng g) -> b ng g", g=G)
        x_flat = rearrange(x_chunks, "b ng g d -> (b ng) g d")
        m_flat = rearrange(mask_chunks, "b ng g -> (b ng) g")

        # intra-Mamba h
        h = self.intra_mamba(x_flat)
        h = h * m_flat.unsqueeze(-1).float()
        h = rearrange(h, "(b ng) g d -> b ng g d", b=B)

        # chunk summary state
        state = self.extract_summary_state(h, mask_chunks)
        chunk_mask = mask_chunks.any(dim=-1)

        # importance / memory
        logit, score = self.compute_importance_score(state, chunk_mask)
        memory = self.process_segments_with_memory(state, score, chunk_mask)
        state_pre_mamba = self.fuse_memory_to_state(state, memory, chunk_mask)
        state_post_mamba = self.chunk_mamba(state_pre_mamba)
        state_post_mamba = state_post_mamba * chunk_mask.unsqueeze(-1).float()
        out_seq = state_post_mamba

        aux_dict = {
            "global_score": score,
            "chunk_mask": chunk_mask,
            "token_mask": token_mask,
            "group_size": self.group_size,
            "pad_len": pad_len,
            "num_tokens": L,
        }

        return out_seq, h, aux_dict, chunk_mask

def gated_fusion()


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mixer1 = ChunkMamba(d_model=dim,group_size=32,anchor_ratio=0.3)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        use_layer_scale = (
            layer_scale is not None
            and type(layer_scale) in [int, float]
        )

        self.gamma_2 = (
            nn.Parameter(layer_scale * torch.ones(dim))
            if use_layer_scale else 1
        )

    def forward(self, x):
        states, h, aux_dict, chunk_mask = self.mixer1(self.norm1(x))
        states = states + self.drop_path(self.gamma_2 * self.mlp(self.norm2(states)))
        states = states * chunk_mask.unsqueeze(-1).float()
        out = states
        out = out * chunk_mask.unsqueeze(-1).float()
        return out, aux_dict, chunk_mask

def hilbert_index(x, y, bits=16):

def hilbert_sort(coords):



class DMGM_MIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False):
        super(DMGM_MIL, self).__init__()
        self.embed_dim = 512

        self._fc1 = [nn.Linear(in_dim, self.embed_dim)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)

        self.n_classes = n_classes
        self.layer = Block(dim=self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.state_attention = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Linear(self.embed_dim, self.n_classes)
        self.survival = survival

    def forward(self, x, coords, return_WSI_attn=True, return_WSI_feature=True):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        h = x.float()
        order = hilbert_sort(coords)
        h = h[:, order, :]
        coords_hilbert = coords[order]
        h = self._fc1(h)
        states, aux_dict, chunk_mask = self.layer(h)
        states = self.norm(states)

        states = (states * chunk_mask.unsqueeze(-1).float())

        A_raw = self.state_attention(states).squeeze(-1)
        A_raw = A_raw.masked_fill(~chunk_mask,-1e4)
        A = F.softmax(A_raw,dim=-1).unsqueeze(1)

        h_state = torch.bmm(A,states).squeeze(1)
        logits = self.classifier(h_state)

        Y_prob = F.softmax(logits,dim=1)
        Y_hat = torch.topk(logits,1,dim=1)[1]

        group_size = aux_dict["group_size"]
        num_tokens = aux_dict["num_tokens"]

        forward_return = {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "chunk_mask": aux_dict["chunk_mask"],
            "global_score": aux_dict["global_score"],
        }

        if return_WSI_attn:
            forward_return["top_ratio"] = top_ratio
            forward_return["group_size"] = group_size
            forward_return["num_tokens"] = num_tokens

        return forward_return

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layer = self.layer.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)