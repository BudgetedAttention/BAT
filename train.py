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
    if model_name in ["dkt", "dkt_plus", "dktplus", "dkt+", "dkvmn", "sakt", "gkt", "simplekt", "sparsekt","cskt","mykt","akt", "robustkt", "folibikt", "lefokt_akt", "mikt", "skvmn", "atkt"]:
        y = torch.masked_select(ys[0], mask)
        t = torch.masked_select(rshft.float(), mask)
        loss = F.binary_cross_entropy(y.double(), t.double())
        if preloss:
            loss = loss + preloss[0]
    else:
        raise ValueError(f"不支持的模型 loss: {model_name}")
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

    if model_name == "dkt":
        y_all = model(c.long(), r.long())
        y_pred = (y_all * F.one_hot(cshft.long(), model.num_q)).sum(-1)
        ys = [y_pred]

    elif model_name in ["dkt_plus", "dktplus", "dkt+"]:
        y_all = model(c.long(), r.long())
        y_next = (y_all * F.one_hot(cshft.long(), model.num_q)).sum(-1)
        y_curr = (y_all * F.one_hot(c.long(), model.num_q)).sum(-1)
        loss = F.binary_cross_entropy(
            torch.masked_select(y_next, mask).double(),
            torch.masked_select(rshft.float(), mask).double(),
        )
        if model.training:
            loss_r = F.binary_cross_entropy(
                torch.masked_select(y_curr, mask).double(),
                torch.masked_select(r.float(), mask).double(),
            )
            smooth_mask = mask[:, 1:]
            state_diff = y_all[:, 1:] - y_all[:, :-1]
            loss_w1 = torch.masked_select(
                torch.norm(state_diff, p=1, dim=-1),
                smooth_mask,
            ).mean() / model.num_q
            loss_w2 = torch.masked_select(
                torch.norm(state_diff, p=2, dim=-1).pow(2),
                smooth_mask,
            ).mean() / model.num_q
            loss = (
                loss
                + model.lambda_r * loss_r
                + model.lambda_w1 * loss_w1
                + model.lambda_w2 * loss_w2
            )
        return loss, y_next, rshft, mask

    elif model_name == "sakt":
        out = model(c.long(), r.long(), cshft.long())
        y_pred = out[0] if isinstance(out, tuple) else out
        ys = [y_pred]


    elif model_name == "dkvmn":
        y_pred, _ = model(cc.long(), cr.long())
        ys = [y_pred[:, 1:]]

    elif model_name in ["akt", "robustkt", "mykt", "folibikt", "lefokt_akt"]:
        y_pred, reg_loss = model(cc.long(), cr.long(), pid_data=cq.long())
        ys = [y_pred[:, 1:]]
        preloss = [reg_loss]
    elif model_name in ["sparsekt"]:
        dcur = {
            "qseqs": q.long(),
            "cseqs": c.long(),
            "rseqs": r.long(),
            "shft_qseqs": qshft.long(),
            "shft_cseqs": cshft.long(),
            "shft_rseqs": rshft.long(),
        }
        if model.training:
            y, _, _, c_reg_loss = model(dcur, train=True)
            ys = [y[:, 1:]]
            if torch.is_tensor(c_reg_loss):
                preloss = [c_reg_loss]
        else:
            ret = model(dcur, train=False)
            y = ret[0] if isinstance(ret, tuple) else ret
            ys = [y[:, 1:]]

    elif model_name in ["simplekt"]:
        dcur = {
            "qseqs": q.long(),
            "cseqs": c.long(),
            "rseqs": r.long(),
            "shft_qseqs": qshft.long(),
            "shft_cseqs": cshft.long(),
            "shft_rseqs": rshft.long(),
        }
        if model.training:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
        else:
            y = model(dcur, train=False)
            ys = [y[:, 1:]]
    elif model_name == "cskt":
        dcur = {
            "qseqs": q.long(),
            "cseqs": c.long(),
            "rseqs": r.long(),
            "shft_qseqs": qshft.long(),
            "shft_cseqs": cshft.long(),
            "shft_rseqs": rshft.long(),
        }
        if model.training:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
        else:
            y = model(dcur, train=False)
            ys = [y[:, 1:]]
    elif model_name == "mikt":
        feed_dict = {
            "questions": cq.long() + 1,
            "skills": cc.long() + 1,
            "responses": cr.long(),
        }
        out_dict = model(feed_dict)
        ys = [out_dict["pred"]]
    elif model_name == "skvmn":
        feed_dict = {
            "skills": cc.long() + 1,
            "responses": cr.long(),
        }
        out_dict = model(feed_dict)
        ys = [out_dict["pred"]]
    elif model_name == "atkt":
        feed_dict = {
            "skills": cc.long() + 1,
            "responses": cr.long(),
        }
        out_dict = model(feed_dict)
        ys = [out_dict["pred"]]
        if model.training:
            pred = torch.masked_select(out_dict["pred"], mask)
            true = torch.masked_select(rshft.float(), mask)
            clean_loss = F.binary_cross_entropy(pred.double(), true.double())
            feature_grad = torch.autograd.grad(
                clean_loss,
                out_dict["features"],
                retain_graph=True,
            )[0]
            norm = torch.sqrt(torch.sum(feature_grad.pow(2), dim=(1, 2), keepdim=True))
            perturbation = model.epsilon * feature_grad.detach() / (norm + 1e-16)
            adv_out = model(feed_dict, perturbation)
            adv_pred = torch.masked_select(adv_out["pred"], mask)
            adv_loss = F.binary_cross_entropy(adv_pred.double(), true.double())
            preloss = [model.beta * adv_loss]
    elif model_name =="dtransformer":
        if model.training:
            loss, y_pred, _, _ = model.get_cl_loss(cc.long(), cr.long(), cq.long())
            y_pred = y_pred[:, 1:] if y_pred is not None else None
            return loss, y_pred, rshft, mask
        logits, _, _, _, _ = model.predict(cc.long(), cr.long(), cq.long())
        y_pred = torch.sigmoid(logits[:, 1:])
        return None, y_pred, rshft, mask

    else:
        raise ValueError(f"不支持的模型 forward: {model_name}")

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

    if model_name == "dtransformer":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model.default_learning_rate,
            weight_decay=model.default_weight_decay,
        )
    else:
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
