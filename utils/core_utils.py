from venv import logger

from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.DMGM_MIL import DMGM_MIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_score, confusion_matrix

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

def calculate_binary_metrics(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    error = 1.0 - acc
    f1 = f1_score(labels, preds, zero_division=0)
    bacc = (sens + spec) / 2.0
    gmean = np.sqrt(sens * spec)

    return acc, error, sens, spec, f1, bacc, gmean


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=20, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, score, model, ckpt_name='checkpoint.pt'):
        # val_gmean
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)

        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, score, model, ckpt_name):
        if self.verbose:
            print(f'Validation G-Mean improved to {score:.6f}. Saving model ...')
        torch.save(model.state_dict(), ckpt_name)


def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from models.topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif args.model_type == 'dmgmmil':
        model = DMGM_MIL(in_dim=args.embed_dim, n_classes=args.n_classes, act=args.act, dropout=args.drop_out)

    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer,scheduler = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=20, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop, auc, val_score = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                 early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop, auc, val_score  = validate(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    results_dict, val_error, val_auc, val_sens, val_spec, val_f1, acc_logger = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, test_sens, test_spec, test_f1, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, test_sens ,val_sens, test_spec, val_spec, test_f1, val_f1


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-8)
    print('\n')
    loss_fn = loss_fn.cuda()

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error


        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

## loss
def global_score_aux_loss(score, mask, label, top_ratio=0.1):
    B, N = score.shape
    losses = []
    for b in range(B):
        valid_score = score[b][mask[b]]

        if valid_score.numel() == 0:
            continue

        k = max(1, int(valid_score.numel() * top_ratio))
        top_score = torch.topk(valid_score, k=k).values.mean()
        target = (label[b] > 0).float()
        loss = F.binary_cross_entropy(
            top_score.clamp(1e-6, 1 - 1e-6),
            target
        )
        losses.append(loss)

    if len(losses) == 0:
        return score.sum() * 0.0

    return torch.stack(losses).mean()

def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-8
    )
    print("\n")
    if loss_fn is not None:
        loss_fn = loss_fn.cuda()

    for batch_idx, batch_data in enumerate(loader):
        if len(batch_data) == 3:
            data, label, coords = batch_data
            data, label, coords = data.to(device), label.to(device), coords.to(device)
            forward_return = model(data,coords)
        else:
            data, label = batch_data
            data, label = data.to(device), label.to(device)
            forward_return = model(data)

        acc_logger.log(forward_return['Y_hat'], label)
        loss_cls = loss_fn(forward_return['logits'], label)
        total_loss = loss_cls

        if forward_return is not None:
            if (
                    "global_score" in forward_return
                    and "chunk_mask" in forward_return
                    and forward_return["global_score"] is not None
                    and forward_return["chunk_mask"] is not None
            ):
                loss_global_aux = global_score_aux_loss(
                    score=forward_return["global_score"],
                    mask=forward_return["chunk_mask"],
                    label=label,
                    top_ratio=0.1
                )
                total_loss = total_loss + 0.05 * loss_global_aux


        loss_value = total_loss.item()
        train_loss += loss_value

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(),
                                                                           data.size(0)))

        error = calculate_error(forward_return['Y_hat'], label)
        train_error += error

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    preds = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if len(batch_data) == 3:
                data, label, coords = batch_data
                data, label, coords = data.to(device,non_blocking=True), label.to(device,non_blocking=True), coords.to(device,non_blocking=True)
                forward_return = model(data,coords)

            else:
                data, label = batch_data
                data, label = data.to(device,non_blocking=True), label.to(device,non_blocking=True)
                forward_return = model(data)


            acc_logger.log(forward_return['Y_hat'], label)
            loss = loss_fn(forward_return['logits'], label)

            prob[batch_idx] = forward_return['Y_prob'].cpu().numpy()
            labels[batch_idx] = label.item()
            preds[batch_idx] = forward_return['Y_hat'].item()

            val_loss += loss.item()
            val_error += calculate_error(forward_return['Y_hat'], label)

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

        val_acc, val_error, val_sens, val_spec, val_f1, val_bacc, val_gmean = calculate_binary_metrics(
            labels,
            preds
        )

        val_score = val_gmean

        print(
            '\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, '
            'sens: {:.4f}, spec: {:.4f}, f1: {:.4f}, bacc: {:.4f}, gmean: {:.4f}'.format(
                val_loss,
                val_error,
                auc,
                val_sens,
                val_spec,
                val_f1,
                val_bacc,
                val_gmean
            )
        )

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
        val_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        val_score = val_f1

        print(
            '\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, macro_f1: {:.4f}'.format(
                val_loss,
                val_error,
                auc,
                val_f1
            )
        )

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/selection_score_gmean', val_score, epoch)

        if n_classes == 2:
            writer.add_scalar('val/sens', val_sens, epoch)
            writer.add_scalar('val/spec', val_spec, epoch)
            writer.add_scalar('val/f1', val_f1, epoch)
            writer.add_scalar('val/bacc', val_bacc, epoch)
            writer.add_scalar('val/gmean', val_gmean, epoch)

    if early_stopping:
        assert results_dir

        early_stopping(
            epoch,
            val_score,
            model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True, auc, val_score

    return False, auc, val_score

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None):
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    preds = np.zeros(len(loader))

    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            logits, Y_prob, Y_hat, _, instance_dict = model(
                data,
                label=label,
                instance_eval=True
            )

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            val_inst_loss += instance_loss.item()
            inst_count += 1

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            preds[batch_idx] = Y_hat.item()

            val_error += calculate_error(Y_hat, label)

    val_error /= len(loader)
    val_loss /= len(loader)

    if inst_count > 0:
        val_inst_loss /= inst_count

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

        val_acc, val_error, val_sens, val_spec, val_f1, val_bacc, val_gmean = calculate_binary_metrics(
            labels,
            preds
        )

        # G-Mean
        val_score = val_gmean

        print(
            '\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, '
            'sens: {:.4f}, spec: {:.4f}, f1: {:.4f}, bacc: {:.4f}, gmean: {:.4f}'.format(
                val_loss,
                val_error,
                auc,
                val_sens,
                val_spec,
                val_f1,
                val_bacc,
                val_gmean
            )
        )

    else:
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        aucs = []

        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
        val_f1 = f1_score(labels, preds, average='macro', zero_division=0)

        # 多分类暂时仍用 macro-F1
        val_score = val_f1

        print(
            '\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, macro_f1: {:.4f}'.format(
                val_loss,
                val_error,
                auc,
                val_f1
            )
        )

    if inst_count > 0:
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        writer.add_scalar('val/selection_score_gmean', val_score, epoch)

        if n_classes == 2:
            writer.add_scalar('val/sens', val_sens, epoch)
            writer.add_scalar('val/spec', val_spec, epoch)
            writer.add_scalar('val/f1', val_f1, epoch)
            writer.add_scalar('val/bacc', val_bacc, epoch)
            writer.add_scalar('val/gmean', val_gmean, epoch)

    if early_stopping:
        assert results_dir

        early_stopping(
            epoch,
            val_score,
            model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True, auc, val_score

    return False, auc, val_score

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx,  batch_data in enumerate(loader):
        if len(batch_data) == 3:
            data, label, coords = batch_data
            data, label, coords = data.to(device), label.to(device), coords.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.inference_mode():
                forward_return = model(data,coords)

        elif len(batch_data) == 2:
            data, label = batch_data
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.inference_mode():
                forward_return = model(data)

        acc_logger.log(forward_return['Y_hat'], label)
        probs = forward_return['Y_prob'].cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = forward_return['Y_hat'].item()

        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id),
                'prob': probs,
                'label': label.item(),
                'pred': forward_return['Y_hat'].item()
            }
        })
        error = calculate_error(forward_return['Y_hat'], label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if n_classes == 2:
        sens, spec, f1 = calculate_sens_spec(all_labels, all_preds)

    return patient_results, test_error, auc, sens, spec, f1, acc_logger
