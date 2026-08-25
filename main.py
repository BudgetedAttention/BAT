import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train import train
from utils.dataloader import (
    KLTDataset,
    build_csv_five_fold_datasets,
    collate_fn,
    get_collab_dataloader,
)

from utils.collab_loss import CollabLossEngine
from utils.cl4kt_dataloader import  SimCLRDatasetWrapper
from baseline.simplekt import simpleKT
from baseline.DKT import DKT
from baseline.DKT_Plus import DKTPlus
from baseline.SAKT import SAKT
from baseline.DTransformer import DTransformer
from baseline.DKVMN import DKVMN
from baseline.AKT import AKT
from baseline.sparsekt import sparseKT
from baseline.cl4kt import CL4KT
from baseline.cskt import CSKT
from baseline.mykt import mykt
from baseline.robustkt import Robustkt
from baseline.folibikt import folibiKT
from baseline.lefokt_akt import LEFOKT_AKT
from baseline.mikt import MIKT
from baseline.skvmn import SKVMN
from baseline.atkt import ATKT


#python main.py --model alkt --dataset assist09 
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Knowledge Tracing Training Script")

    parser.add_argument("--model", "--model_name", dest="model", type=str, default="mykt",
                        help="Model name: dkt, sakt, akt, simplekt, robustkt, sparsekt/spareskt, cl4kt, etc.")
    parser.add_argument("--dataset", "--dataset_name", dest="dataset", type=str, default="assist09",
                        help="Dataset name: assist09/assistments09, assist12/assistments12, assist17/assistments17, junyi, algebra05, ednet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", "--n_epochs", dest="epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument(
        "--qk_content_mode", type=str, default="dot", choices=["dot", "cos"],
        help="Q/K content scorer for compatible attention KT backbones.",
    )
    parser.add_argument(
        "--cosine_scale_init", type=float, default=1.0,
        help="Initial positive per-head cosine content scale.",
    )
    parser.add_argument("--seq_len", "--maxstep", dest="seq_len", type=int, default=200)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--final_fc_dim", type=int, default=512)
    parser.add_argument("--separate_qa", type=int, default=0,
                        help="Use question-response pair embeddings instead of response embedding plus question embedding.")


    parser.add_argument(
        "--lefokt_emb_type",
        type=str,
        default="qid_sandwich",
        choices=["qid", "qid_alibi", "qid_log", "qid_power", "qid_t5", "qid_fire", "qid_sandwich"],
        help="Relative-position branch used by LEFOKT-AKT.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--tuning", type=int, default=0,
                        help="Validation-only hyperparameter tuning; skip all test-set evaluation.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Student-level cross-validation fold index. Omit to use the original fixed split.")
    parser.add_argument("--eval_test_each_epoch", type=int, default=0,
                        help="Evaluate the test split during training. Keep 0 for leakage-safe experiments.")
    args = parser.parse_args()
    args.model = args.model.lower()
    args.dataset_name = args.dataset

    # Dataset configuration
    if args.dataset_name in ["assist09", "assistments09"]:
        args.dataset = "assist09"
        args.dataset_name = "assistments09"
        args.n_question = 149
        args.n_pid = 17728
        args.data_dir = os.path.join("datasets", "assistments09")

    elif args.dataset_name in ["assist12", "assistments12"]:
        args.dataset = "assist12"
        args.dataset_name = "assistments12"
        args.n_question = 264
        args.n_pid = 51632
        args.data_dir = os.path.join("datasets", "assistments12")

    elif args.dataset_name == "algebra05":
        args.dataset = "algebra05"
        args.n_question = 271
        args.n_pid = 173113
        args.data_dir = os.path.join("datasets", "algebra05")

    elif args.dataset_name in ["assist17", "assistments17"]:
        args.dataset = "assist17"
        args.dataset_name = "assistments17"
        args.n_question = 411
        args.n_pid = 3162
        args.data_dir = os.path.join("datasets", "assistments17")

    elif args.dataset_name == "ednet":
        args.dataset = "ednet"
        args.n_question = 1472
        args.n_pid = 18144
        args.data_dir = os.path.join("datasets", "ednet")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    args.preprocessed_path = os.path.join(args.data_dir, "preprocessed_df.csv")
    if args.fold is not None:
        if args.fold < 0 or args.fold >= 5:
            raise ValueError("--fold must be one of 0, 1, 2, 3, or 4.")
        if not os.path.exists(args.preprocessed_path):
            raise FileNotFoundError(
                f"Five-fold mode requires the original preprocessed table: {args.preprocessed_path}"
            )

    args.num_q = args.n_question

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = get_args()
    set_seed(args.seed)


    model_name = args.model
    data_name = args.dataset
    num_skills = args.num_q
    num_questions = args.n_pid
    seq_len = args.seq_len
    device = torch.device(args.device)
    train_path = os.path.join(args.data_dir, "train.txt") if args.fold is None else None
    valid_path = os.path.join(args.data_dir, "valid.txt") if args.fold is None else None
    test_path = os.path.join(args.data_dir, "test.txt") if args.fold is None else None
    print(f"Using device: {device}")
    print(f"Dataset: {data_name} | num_skills: {num_skills} | num_questions: {num_questions}")
    print(f"Model: {model_name}")
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.params_str = time_str
    args.save_dir = os.path.join("saved_model", model_name, data_name, time_str)
    args.log_dir = os.path.join("logs", data_name, model_name, time_str)
    args.tb_dir = os.path.join("runs", data_name, model_name, time_str)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seq_len": args.seq_len,
        "patience": args.patience,
        "fold": args.fold,
        "data_dir": args.data_dir,
        "preprocessed_path": args.preprocessed_path if args.fold is not None else None,
    }

    model_config = {
        "hidden_size": args.d_model,
        "num_blocks": args.n_blocks,
        "num_attn_heads": args.num_attn_heads,
        "kq_same": 1,
        "final_fc_dim": args.final_fc_dim,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "qk_content_mode": args.qk_content_mode,
        "cosine_scale_init": args.cosine_scale_init,
    }
    if model_name == "dtransformer":
        model_config = DTransformer.default_config()
        train_config["learning_rate"] = DTransformer.default_learning_rate
        train_config["weight_decay"] = DTransformer.default_weight_decay
        train_config["optimizer"] = "AdamW"
    elif model_name == "mykt":
        model_config["kt_attn"] = "state_discrepancy"
        model_config["sd_use_gap"] = bool(args.sd_use_gap)
        model_config["sd_use_alibi"] = bool(args.sd_use_alibi)
        model_config["sd_content_mode"] = args.sd_content_mode
        model_config["sd_gap_mode"] = args.sd_gap_mode
        model_config["sd_gap_dim"] = args.sd_gap_dim
        model_config["sd_gap_reduce"] = args.sd_gap_reduce
        model_config["sd_content_scale_init"] = args.sd_content_scale_init
        model_config["sd_gap_scale_init"] = args.sd_gap_scale_init
        model_config["sd_learnable_alibi_scale"] = bool(args.sd_learnable_alibi_scale)

    elif model_name == "lefokt_akt":
        model_config["emb_type"] = args.lefokt_emb_type
    elif model_name == "mikt":
        model_config = {
            "embedding_size": args.mikt_embedding_size,
            "state_d": args.mikt_state_dim,
            "dropout": args.dropout,
            "q_matrix": "dense item-KC matrix from preprocessed_df.csv",
        }

    if model_name == "cl4kt":
        model_config.update({
            "reg_cl": 0.1,
            "mask_prob": 0.2,
            "crop_prob": 0.3,
            "permute_prob": 0.3,
            "replace_prob": 0.3,
            "negative_prob": 1.0,
            "hard_negative_weight": 1.0,
            "temp": 0.05,
            "l2": 0.0,
        })
    with open(os.path.join(args.save_dir, "configs.json"), "w", encoding="utf-8") as f:
        json.dump({"model_config": model_config, "train_config": train_config}, f, indent=2, ensure_ascii=False)

    print("Train config:", train_config)
    print("Model config:", model_config)



    if model_name == "cl4kt":
        dataset_return_type = "cl4kt"
    else:
        dataset_return_type = "standard"
    if args.fold is not None:
        print(f"Loading student-level fold {args.fold} directly from {args.preprocessed_path}...")
        fold_datasets, split_info = build_csv_five_fold_datasets(
            args.preprocessed_path,
            fold=args.fold,
            max_seq_len=seq_len,
            return_type=dataset_return_type,
            num_skills=num_skills,
            num_questions=num_questions,
        )
        train_dataset = fold_datasets["train"]
        valid_dataset = fold_datasets["valid"]
        test_dataset = fold_datasets["test"]
        train_config["cv_split"] = split_info
        print("CV split:", split_info)
    else:
        dataset_kwargs = {
            "max_seq_len": seq_len,
            "return_type": dataset_return_type,
            "num_skills": num_skills,
            "num_questions": num_questions,
        }
        train_dataset = KLTDataset(train_path, **dataset_kwargs)
        valid_dataset = KLTDataset(valid_path, **dataset_kwargs)
        test_dataset = KLTDataset(test_path, **dataset_kwargs)

    if model_name == "cl4kt":
        print("Loading CL4KT datasets with SimCLRDatasetWrapper...")

        train_loader = DataLoader(
            SimCLRDatasetWrapper(
                train_dataset,
                seq_len,
                model_config["mask_prob"],
                model_config["crop_prob"],
                model_config["permute_prob"],
                model_config["replace_prob"],
                model_config["negative_prob"],
                eval_mode=False,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        valid_loader = DataLoader(
            SimCLRDatasetWrapper(valid_dataset,seq_len,0,0,0, 0,0, eval_mode=True,),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        test_loader = DataLoader(
            SimCLRDatasetWrapper(test_dataset,seq_len,0,0,0, 0,0, eval_mode=True,),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        print("Loading standard KLTDataset loaders...")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    print(f"Initializing {model_name} model...")
    if model_name == "dkt":
        model = DKT(num_q=num_skills, emb_size=args.d_model, hidden_size=args.d_model).to(device)
    elif model_name in ["dkt_plus", "dktplus", "dkt+"]:
        model = DKTPlus(num_q=num_skills, emb_size=args.d_model, hidden_size=args.d_model).to(device)
    elif model_name == "sakt":
        model = SAKT(num_c=num_skills, seq_len=seq_len, emb_size=args.d_model, num_attn_heads=args.num_attn_heads, dropout=args.dropout).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(num_q=num_skills, dim_s=args.d_model, size_m=20).to(device)
    elif model_name == "dtransformer":
        model = DTransformer(n_questions=num_skills, n_pid=num_questions).to(device)
    elif model_name == "akt":
        model = AKT(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout, num_attn_heads=args.num_attn_heads).to(device)
    elif model_name == "folibikt":
        model = folibiKT(
            n_question=num_skills,
            n_pid=num_questions,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            final_fc_dim=args.final_fc_dim,
            num_attn_heads=args.num_attn_heads,
            separate_qa=bool(args.separate_qa),
            seq_len=seq_len,
            emb_type="qid_alibi",
        ).to(device)
    elif model_name == "lefokt_akt":
        model = LEFOKT_AKT(
            n_question=num_skills,
            n_pid=num_questions,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            final_fc_dim=args.final_fc_dim,
            num_attn_heads=args.num_attn_heads,
            separate_qa=bool(args.separate_qa),
            emb_type=args.lefokt_emb_type,
        ).to(device)
    elif model_name == "mykt":
        model = mykt(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout, num_attn_heads=args.num_attn_heads,
                     separate_qa=bool(args.separate_qa),
                     sd_use_gap=bool(args.sd_use_gap),
                     sd_use_alibi=bool(args.sd_use_alibi),
                     sd_content_mode=args.sd_content_mode,
                     sd_gap_mode=args.sd_gap_mode,
                     sd_gap_dim=args.sd_gap_dim,
                     sd_gap_reduce=args.sd_gap_reduce,
                     sd_content_scale_init=args.sd_content_scale_init,
                     sd_gap_scale_init=args.sd_gap_scale_init,
                     sd_learnable_alibi_scale=bool(args.sd_learnable_alibi_scale)).to(device)
    elif model_name == "simplekt":
        model = simpleKT(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout, num_attn_heads=args.num_attn_heads).to(device)
    elif model_name == "robustkt":
        model = Robustkt(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout, num_attn_heads=args.num_attn_heads, separate_qa=bool(args.separate_qa)).to(device)
    elif model_name in ["sparsekt"]:
        model = sparseKT(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout,
                         num_attn_heads=args.num_attn_heads, seq_len=seq_len, emb_type="qid_sparseattn").to(device)
    elif model_name == "cl4kt":
        model = CL4KT(num_skills, num_questions, seq_len, **model_config).to(device)
    elif model_name == "cskt":
        model = CSKT(n_question=num_skills, n_pid=num_questions, d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout,
                     d_ff=args.d_ff, kq_same=args.kq_same, final_fc_dim=args.final_fc_dim,
                     num_attn_heads=args.num_attn_heads, seq_len=seq_len, emb_type="qid").to(device)
    elif model_name == "mikt":
        relation_df = pd.read_csv(
            args.preprocessed_path,
            sep="\t",
            usecols=["item_id", "skill_id"],
        ).drop_duplicates()
        item_ids = torch.as_tensor(relation_df["item_id"].to_numpy() + 1, dtype=torch.long, device=device)
        skill_ids = torch.as_tensor(relation_df["skill_id"].to_numpy() + 1, dtype=torch.long, device=device)
        mikt_num_questions = int(item_ids.max().item()) + 1
        mikt_num_skills = int(skill_ids.max().item()) + 1
        pro2skill = torch.zeros((mikt_num_questions, mikt_num_skills), device=device)
        pro2skill[item_ids, skill_ids] = 1.0
        model = MIKT(
            mikt_num_skills,
            mikt_num_questions,
            seq_len,
            pro2skill,
            state_d=args.mikt_state_dim,
            embedding_size=args.mikt_embedding_size,
            dropout=args.dropout,
        ).to(device)
    elif model_name == "skvmn":
        model = SKVMN(
            num_skills + 1,
            dim_s=args.skvmn_dim_s,
            size_m=args.skvmn_size_m,
            dropout=args.dropout,
        ).to(device)
    elif model_name == "atkt":
        model = ATKT(
            num_skills + 1,
            skill_dim=args.atkt_skill_dim,
            answer_dim=args.atkt_answer_dim,
            hidden_dim=args.atkt_hidden_dim,
            attention_dim=args.atkt_attention_dim,
            epsilon=args.atkt_epsilon,
            beta=args.atkt_beta,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Model initialization is not implemented: {model_name}")

    train(model, loaders, args)


if __name__ == "__main__":
    main()

