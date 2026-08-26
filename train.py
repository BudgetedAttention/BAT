import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.collab_aug import generate_augmented_batch
def batch_to_device(batch_data, device):
    for key in batch_data:
        if isinstance(batch_data[key], tuple):
            batch_data[key] = tuple(t.to(device) for t in batch_data[key])
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def cal_loss(model_name, ys, rshft, mask, preloss):
    if model_name in ["sakt", "simplekt", "sparsekt", "akt", "robustkt", "folibikt"]:
        y = torch.masked_select(ys[0], mask)
        t = torch.masked_select(rshft.float(), mask)
        loss = F.binary_cross_entropy(y.double(), t.double())
        if preloss:
            loss = loss + preloss[0]
    else:
        raise ValueError(f" loss: {model_name}")
    return loss


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def model_forward(model, batch, device, model_name, num_q=None):

    if model_name == "cl4kt":
        batch = _move_to_device(batch, device)
        out_dict = model(batch)
        if model.training:
            loss, _, _ = model.loss(batch, out_dict)
        else:
            loss = torch.tensor(0.0, device=device)
        y_pred = out_dict["pred"]
        y_true = out_dict["true"].float()
        mask = y_true > -1
        return loss, y_pred, y_true, mask

    q, c, r, qshft, cshft, rshft, m = [x.to(device) for x in batch]
    mask = m.bool()

    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    ys, preloss = [], []

    if model_name == "sakt":
        out = model(c.long(), r.long(), cshft.long())
        y_pred = out[0] if isinstance(out, tuple) else out
        ys = [y_pred]
    elif model_name in ["akt", "robustkt", "folibikt"]:
        y_pred, reg_loss = model(cc.long(), cr.long(), pid_data=cq.long())
        ys = [y_pred[:, 1:]]
        preloss = [reg_loss]
    elif model_name == "sparsekt":
        dcur = {"qseqs": q.long(), "cseqs": c.long(), "rseqs": r.long(), "shft_qseqs": qshft.long(), "shft_cseqs": cshft.long(), "shft_rseqs": rshft.long()}
        if model.training:
            y, _, _, c_reg_loss = model(dcur, train=True)
            ys = [y[:, 1:]]
            if torch.is_tensor(c_reg_loss):
                preloss = [c_reg_loss]
        else:
            ret = model(dcur, train=False)
            ys = [(ret[0] if isinstance(ret, tuple) else ret)[:, 1:]]
    elif model_name == "simplekt":
        dcur = {"qseqs": q.long(), "cseqs": c.long(), "rseqs": r.long(), "shft_qseqs": qshft.long(), "shft_cseqs": cshft.long(), "shft_rseqs": rshft.long()}
        if model.training:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
        else:
            y = model(dcur, train=False)
            ys = [y[:, 1:]]
    else:
        raise ValueError(f"Unsupported model for this BAT release: {model_name}")

    loss = cal_loss(model_name, ys, rshft, mask, preloss)
    return loss, ys[0], rshft, mask


def evaluate(model, loader, device, model_name):
    model.eval()
    y_trues, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            _, y_pred, y_true, mask = model_forward(
                model,
                batch,
                device,
                model_name,
            )
            if y_pred is not None:
                y_trues.append(torch.masked_select(y_true.float(), mask).detach().cpu().numpy())
                y_scores.append(torch.masked_select(y_pred, mask).detach().cpu().numpy())

    if len(y_trues) == 0:
        return 0.0, 0.0

    ts = np.concatenate(y_trues)
    ps = np.concatenate(y_scores)
    if len(np.unique(ts)) < 2:
        auc = 0.5
    else:
        auc = metrics.roc_auc_score(ts, ps)
    acc = metrics.accuracy_score(ts, (ps >= 0.5))
    return float(auc), float(acc)


def train(model, loaders, args, criterion=None):

    device = torch.device(args.device)
    model_name = args.model.lower()
    data_name = args.dataset

    train_loader = loaders["train"]
    valid_loader = loaders["valid"]
    test_loader = loaders["test"]

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "l2", 0.0) if model_name == "cl4kt" else 0.0,
        )
    writer = SummaryWriter(log_dir=args.tb_dir) if SummaryWriter is not None else None

    best_auc = 0.0
    best_epoch = 0
    no_improve_cnt = 0  
    patience = getattr(args, "patience", 12)
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    print(f" >>> RUNNING: Model={model_name} | Data={data_name} | Epochs={args.epochs} | Device={device}")
    print(f"Save Dir: {args.save_dir}")
    print(f"Log Dir: {args.log_dir}")
    if writer is not None:
        print(f"TensorBoard: {args.tb_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        tr_preds, tr_trues = [], []
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}", ncols=100)

        for batch in pbar:
            optimizer.zero_grad()
            loss, y_pred, y_true, mask = model_forward(
                model,
                batch,
                device,
                model_name,
            )
            loss.backward()
            if "transformer" in model_name:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if y_pred is not None:
                y_np = torch.masked_select(y_pred, mask.bool()).detach().cpu().numpy()
                t_np = torch.masked_select(y_true.float(), mask.bool()).detach().cpu().numpy()
                if len(y_np) > 0:
                    tr_preds.append(y_np)
                    tr_trues.append(t_np)

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if tr_preds:
            all_trues = np.concatenate(tr_trues)
            all_preds = np.concatenate(tr_preds)
            if len(np.unique(all_trues)) < 2:
                train_auc = 0.5
            else:
                train_auc = metrics.roc_auc_score(all_trues, all_preds)
            train_acc = metrics.accuracy_score(all_trues, (all_preds >= 0.5))
        else:
            train_auc, train_acc = 0.0, 0.0

        loss_mean = float(np.mean(train_losses)) if train_losses else 0.0
        v_auc, v_acc = evaluate(model, valid_loader, device, model_name)
        if getattr(args, "tuning", 0) or not getattr(args, "eval_test_each_epoch", 0):
            t_auc, t_acc = float("nan"), float("nan")
        else:
            t_auc, t_acc = evaluate(model, test_loader, device, model_name)

        print(f" -- Loss: {loss_mean:.4f} | Tr.AUC: {train_auc:.4f} | Val.AUC: {v_auc:.4f} | Te.AUC: {t_auc:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", loss_mean, epoch)
            writer.add_scalar("AUC/train", train_auc, epoch)
            writer.add_scalar("AUC/valid", v_auc, epoch)
            writer.add_scalar("AUC/test", t_auc, epoch)
            writer.add_scalar("ACC/train", train_acc, epoch)
            writer.add_scalar("ACC/valid", v_acc, epoch)
            writer.add_scalar("ACC/test", t_acc, epoch)

        if v_auc > best_auc:
            best_auc = v_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if no_improve_cnt >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if writer is not None:
        writer.close()

    if os.path.exists(best_ckpt_path) and getattr(args, "tuning", 0):
        print(f"\nTuning finished. Best validation AUC: {best_auc:.6f} at Ep {best_epoch}")
        with open(os.path.join(args.log_dir, "validation_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\nBest Epoch: {best_epoch}\n")
            f.write(f"Validation AUC: {best_auc:.6f}\n")
    elif os.path.exists(best_ckpt_path):
        print(f"\nTraining finished. Loading best model from Ep {best_epoch}...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        final_auc, final_acc = evaluate(model, test_loader, device, model_name)

        print("\n" + "=" * 40)
        print(f" FINAL RESULT ({model_name} @ {data_name})")
        print("=" * 40)
        print(f"Overall AUC:   {final_auc:.4f}")
        print(f"Overall ACC:   {final_acc:.4f}")
        print("-" * 40)

        with open(os.path.join(args.log_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\nBest Epoch: {best_epoch}\n")
            f.write(f"AUC: {final_auc:.5f}\nACC: {final_acc:.5f}\n")
        print(f"\nResult saved to: {os.path.join(args.log_dir, 'result.txt')}")
    else:
        print("Error: No model saved.")



