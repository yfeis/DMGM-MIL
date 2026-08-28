# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from ..ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, mamba_inner_fn_no_out_proj

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from einops import rearrange


class TransposeTokenReEmbedding:

    @staticmethod
    def patches_reemmbedding(x, rate):
        x = rearrange(x, "b c l -> b l c")
        B, N, C = x.shape
        value = N // rate
        if value % 2 == 0:
            if N % rate != 0:
                padding_length = (value + 2) * rate - N
                padded_x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))
            else:
                padded_x = x
        else:
            padding_length = (value + 1) * rate - N
            padded_x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))

        x = rearrange(padded_x, "b (h w) n -> b h w n", w=rate)

        x0 = x[:, 0::2, 0::2, :]  # ... H/2 W/2
        x1 = x[:, 1::2, 0::2, :]  # ... H/2 W/2
        x2 = x[:, 0::2, 1::2, :]  # ... H/2 W/2
        x3 = x[:, 1::2, 1::2, :]  # ... H/2 W/2

        x0 = rearrange(x0, "b h w n -> b n (h w)")
        x1 = rearrange(x1, "b h w n -> b n (w h)")
        x2 = rearrange(x2, "b h w n -> b n (h w)")
        x3 = rearrange(x3, "b h w n -> b n (w h)")
        # x0 = rearrange(x0, "b h w n -> b (h w) n")
        # x1 = rearrange(x1, "b h w n -> b (w h) n")
        # x2 = rearrange(x2, "b h w n -> b (h w) n")
        # x3 = rearrange(x3, "b h w n -> b (w h) n")
        # x_merge = torch.cat([x0, x1, x2.flip([-1]), x3.flip([-1])], dim=-1)
        # x_merge = rearrange(x_merge, "b l c -> b c l")
        x_merge = [x0, x1, x2.flip([-1]), x3.flip([-1])]
        return x_merge

    @staticmethod
    def restore_x0_x2(x0, x1, rate):
        x1 = x1.flip([-1])

        x0 = rearrange(x0, "b n (h w) -> b h w n", w=rate // 2)
        x1 = rearrange(x1, "b n (h w) -> b h w n", w=rate // 2)
        b, h, w, n = x0.shape
        x_restored = torch.zeros([b, h, 2 * w, n], dtype=x0.dtype, device=x0.device)
        x_restored[:, :, 0::2, :] = x0
        x_restored[:, :, 1::2, :] = x1

        x_restored = rearrange(x_restored, "b h w n -> b n (h w)")

        return x_restored

    @staticmethod
    def restore_x1_x3(x0, x1, rate):
        x1 = x1.flip([-1])

        x0 = rearrange(x0, "b n (w h) -> b h w n", w=rate // 2)
        x1 = rearrange(x1, "b n (w h) -> b h w n", w=rate // 2)
        b, h, w, n = x0.shape
        x_restored = torch.zeros([b, h, 2 * w, n], dtype=x0.dtype, device=x0.device)
        x_restored[:, :, 0::2, :] = x0
        x_restored[:, :, 1::2, :] = x1

        x_restored = rearrange(x_restored, "b h w n -> b n (h w)")

        return x_restored

    @staticmethod
    def restore(x0, x1, rate, length):
        x0 = rearrange(x0, "b n (h w) -> b h w n", w=rate)
        x1 = rearrange(x1, "b n (h w) -> b h w n", w=rate)
        b, h, w, n = x0.shape
        x_restored = torch.zeros([b, 2 * h, w, n], dtype=x0.dtype, device=x0.device)
        x_restored[:, 0::2, :, :] = x0
        x_restored[:, 1::2, :, :] = x1

        x_restored = rearrange(x_restored, "b h w n -> b n (h w)")
        x_restored = x_restored[:, :, :length]

        return x_restored


class BiAttn(nn.Module):
    """
    This class comes from EfficientVMamba,
    the link is https://github.com/TerryPei/EfficientVMamba/blob/main/classification/lib/models/mamba/efficient_mamba.py#L668
    """

    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()
        self.out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x = rearrange(x, "b c l -> b l c")
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]

        attn = c_attn  # * s_attn  # [B, N, C]
        out = ori_x * attn
        out = self.out(out)
        out = rearrange(out, "b l c -> b c l")

        return out


class AtrousFourWayMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)  ## [1, seqlen, 2048]
        # self.in_out_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)


        self.att = BiAttn(self.d_inner, act_ratio=0.5)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_c = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.x_proj_c = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
            self.dt_proj_c.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True
        self.dt_proj_c.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        A_c = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_c_log = torch.log(A_c)  # Keep A_log in fp32
        self.A_c_log = nn.Parameter(A_c_log)
        self.A_c_log._no_weight_decay = True


        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True
        self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_c._no_weight_decay = True

        # self.out1_proj = nn.Linear(self.d_inner, 2 * self.d_inner, bias=bias, **factory_kwargs)
        # self.out_proj = nn.Linear(self.d_inner, 2 * self.d_inner, bias=bias, **factory_kwargs)
        # self.out_proj = nn.Linear(self.d_inner, 2 * self.d_inner, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, rate=10):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        assert rate % 2 == 0
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )  # [batch, 2*d_inner, seqlen] ,in here is [batch, 2048, seqlen]
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
        A_c = -torch.exp(self.A_c_log.float())  # (d_inner, d_state)


        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            N, C, L = xz.shape
            xz_1, xz_2, xz_3, xz_4 = TransposeTokenReEmbedding.patches_reemmbedding(xz, rate)
            sub_xz_1 = TransposeTokenReEmbedding.restore_x0_x2(xz_1, xz_3, rate)
            sub_xz_2 = TransposeTokenReEmbedding.restore_x1_x3(xz_2, xz_4, rate)

            xz_scan2 = torch.cat([xz_1, xz_2, xz_3, xz_4], dim=-1)
            xz_scan3 = torch.cat([sub_xz_1, sub_xz_2], dim=-1)

            out_scan2 = mamba_inner_fn_no_out_proj(
                xz_scan2,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_1, out_2, out_3, out_4 = torch.split(out_scan2, xz_1.shape[-1], dim=-1)
            out_merge1 = TransposeTokenReEmbedding.restore_x0_x2(out_1, out_3, rate)
            # print(out_merge1.shape)
            # out_merge1 = self.att(out_merge1)
            # print(out_merge1.shape)
            out_merge2 = TransposeTokenReEmbedding.restore_x1_x3(out_2, out_4, rate)
            # print(out_merge2.shape)
            # out_merge2 = self.att(out_merge2)
            # print(out_merge2.shape)
            out_sub_merge = TransposeTokenReEmbedding.restore(out_merge1, out_merge2, rate, L)
            out_sub_merge = self.att(out_sub_merge)
            out_scan3 =mamba_inner_fn_no_out_proj(
                xz_scan3,
                # out_merge1,
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            out_merge_1, out_merge_2 = torch.split(out_scan3, sub_xz_1.shape[-1], dim=-1)
            out_merge = TransposeTokenReEmbedding.restore(out_merge_1, out_merge_2.flip([-1]), rate, L)
            # # print(print(out_merge.shape))
            out_merge = self.att(out_merge)
            # print(print(out_merge.shape))
            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d_c.weight,
                self.conv1d_c.bias,
                self.x_proj_c.weight,
                self.dt_proj_c.weight,
                A_c,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D_c.float(),
                delta_bias=self.dt_proj_c.bias.float(),
                delta_softplus=True,
            )
            # print(out.shape)
            out = self.att(out)
            # out_finally = self.att(out) + self.att(rearrange(self.in_out_proj(hidden_states), 'b l c ->b c l'))
            # out = F.linear(rearrange(out_sub_merge + out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            out = F.linear(rearrange(out + out_sub_merge + out_merge, "b d l -> b l d"), self.out_proj.weight,
                           self.out_proj.bias)
            # out = self.att(out) + self.att(hidden_states)
        else:
            # x, z拆分用于两个分支的聚合
            x, z = xz.chunk(2, dim=1)
            # x_b = TransposeTokenReEmbedding.transpose_normal_padding(xz, rate=rate)
            # x_c = x.flip([-1])
            # x_d = x_b.flip([-1])
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            # 为无causal_conv1d_fn提供了手动的处理方案
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
                # x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
                # x_c = self.act(self.conv1d_c(x_c)[..., :seqlen])
                # x_d = self.act(self.conv1d_d(x_d)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                # x_b = causal_conv1d_fn(
                #     x=x_b,
                #     weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                #     bias=self.conv1d_b.bias,
                #     activation=self.activation,
                # )
                # x_c = causal_conv1d_fn(
                #     x=x_c,
                #     weight=rearrange(self.conv1d_c.weight, "d 1 w -> d w"),
                #     bias=self.conv1d_c.bias,
                #     activation=self.activation,
                # )
                # x_d = causal_conv1d_fn(
                #     x=x_d,
                #     weight=rearrange(self.conv1d_d.weight, "d 1 w -> d w"),
                #     bias=self.conv1d_d.bias,
                #     activation=self.activation,
                # )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            # SSM内核操作，分别为前向和反向生成对应的B和C
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            # x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            # dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt_b = self.dt_proj_b.weight @ dt_b.t()
            # dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            # B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            #
            # x_dbl_c = self.x_proj_c(rearrange(x_c, "b d l -> (b l) d"))  # (bl d)
            # dt_c, B_c, C_c = torch.split(x_dbl_c, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt_c = self.dt_proj_c.weight @ dt_c.t()
            # dt_c = rearrange(dt_c, "d (b l) -> b d l", l=seqlen)
            # B_c = rearrange(B_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # C_c = rearrange(C_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            #
            # x_dbl_d = self.x_proj_d(rearrange(x_d, "b d l -> (b l) d"))  # (bl d)
            # dt_d, B_d, C_d = torch.split(x_dbl_d, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt_d = self.dt_proj_d.weight @ dt_d.t()
            # dt_d = rearrange(dt_d, "d (b l) -> b d l", l=seqlen)
            # B_d = rearrange(B_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # C_d = rearrange(C_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            # y_b = selective_scan_fn(
            #     x_b,
            #     dt_b,
            #     A_b,
            #     B_b,
            #     C_b,
            #     self.D_b.float(),
            #     z=z,
            #     delta_bias=self.dt_proj_b.bias.float(),
            #     delta_softplus=True,
            #     return_last_state=ssm_state is not None,
            # )
            # y_c = selective_scan_fn(
            #     x_c,
            #     dt_c,
            #     A_c,
            #     B_c,
            #     C_c,
            #     self.D_c.float(),
            #     z=z,
            #     delta_bias=self.dt_proj_c.bias.float(),
            #     delta_softplus=True,
            #     return_last_state=ssm_state is not None,
            # )
            # y_d = selective_scan_fn(
            #     x_d,
            #     dt_d,
            #     A_d,
            #     B_d,
            #     C_d,
            #     self.D_d.float(),
            #     z=z,
            #     delta_bias=self.dt_proj_d.bias.float(),
            #     delta_softplus=True,
            #     return_last_state=ssm_state is not None,
            # )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            # y_b = rearrange(y_b, "b d l -> b l d")
            # y_c = rearrange(y_c, "b d l -> b l d")
            # y_d = rearrange(y_d, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)