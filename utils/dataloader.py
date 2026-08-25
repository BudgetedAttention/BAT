import itertools
import random
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KLTDataset(Dataset):
    def __init__(
        self,
        file_path=None,
        min_seq_len=3,
        max_seq_len=200,
        return_type="standard",
        num_skills=None,
        num_questions=None,
        sequences=None,
    ):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.return_type = return_type

        self.num_skills = num_skills
        self.num_questions = num_questions

        self.sequences = sequences if sequences is not None else self._load_data(file_path)

        # CL4KT 的 replace augmentation 需要 easier/harder skill
        if self.return_type == "cl4kt":
            self._build_skill_difficulty_maps()

    def _load_data(self, file_path):
        sequences = []

        with open(file_path, "r") as f:
            for length_str, ques_str, concepts_str, ans_str in itertools.zip_longest(*[f] * 4):
                if length_str is None or length_str.strip() == "":
                    break

                seq_len = int(length_str.strip())
                if seq_len < self.min_seq_len:
                    continue

                p_seq = [int(x) for x in ques_str.strip().strip(",").split(",")]
                s_seq = [int(x) for x in concepts_str.strip().strip(",").split(",")]
                r_seq = [int(x) for x in ans_str.strip().strip(",").split(",")]

                sequences.append(
                    {
                        "p": p_seq,
                        "s": s_seq,
                        "r": r_seq,
                    }
                )

        return sequences

    def _build_skill_difficulty_maps(self):
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)

        for seq in self.sequences:
            s_list = seq["s"][-self.max_seq_len:]
            r_list = seq["r"][-self.max_seq_len:]

            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s])
            for s in skill_correct
        }

        ordered_skills = [
            item[0]
            for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]

        self.easier_skills = {}
        self.harder_skills = {}

        for i, s in enumerate(ordered_skills):
            if i == 0:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        p_seq = seq["p"]
        s_seq = seq["s"]
        r_seq = seq["r"]


        if len(p_seq) > self.max_seq_len:
            p_seq = p_seq[-self.max_seq_len:]
            s_seq = s_seq[-self.max_seq_len:]
            r_seq = r_seq[-self.max_seq_len:]

            # 普通模型仍然返回 KLT 原来的格式
        if self.return_type == "standard":
            return {
                "p": p_seq,
                "s": s_seq,
                "r": r_seq,
            }

        # CL4KT 返回原版 SimCLRDatasetWrapper 需要的格式
        if self.return_type == "cl4kt":
            padded_q = torch.zeros(self.max_seq_len, dtype=torch.long)
            padded_s = torch.zeros(self.max_seq_len, dtype=torch.long)
            padded_r = torch.full((self.max_seq_len,), -1, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)

            padded_q[-len(p_seq):] = torch.tensor(p_seq, dtype=torch.long)
            padded_s[-len(s_seq):] = torch.tensor(s_seq, dtype=torch.long)
            padded_r[-len(r_seq):] = torch.tensor(r_seq, dtype=torch.long)
            attention_mask[-len(s_seq):] = torch.ones(len(s_seq), dtype=torch.long)

            return {
                "questions": padded_q,
                "skills": padded_s,
                "responses": padded_r,
                "attention_mask": attention_mask,
            }

        raise ValueError(f"Unknown return_type: {self.return_type}")


def _split_outer_user_folds(user_ids, n_splits, seed):
    shuffled = list(user_ids)
    random.Random(seed).shuffle(shuffled)
    fold_sizes = [len(shuffled) // n_splits] * n_splits
    for index in range(len(shuffled) % n_splits):
        fold_sizes[index] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(shuffled[start:start + size])
        start += size
    return folds


def build_csv_five_fold_datasets(
    csv_path,
    fold,
    max_seq_len=200,
    n_splits=5,
    valid_ratio=0.1,
    split_seed=42,
    min_seq_len=3,
    return_type="standard",
    num_skills=None,
    num_questions=None,
):
    """Build one student-level CV fold directly from a preprocessed interaction CSV."""
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], got {fold}.")

    frame = pd.read_csv(csv_path, sep="\t")
    required = {"user_id", "item_id", "timestamp", "correct", "skill_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["_row_order"] = range(len(frame))

    sequences_by_user = {}
    for user_id, group in frame.groupby("user_id", sort=False):
        if len(group) < min_seq_len:
            continue
        ordered = group.sort_values(["timestamp", "_row_order"], kind="stable")
        sequences_by_user[user_id] = {
            "p": ordered["item_id"].astype("int64").tolist(),
            "s": ordered["skill_id"].astype("int64").tolist(),
            "r": ordered["correct"].astype("int64").tolist(),
        }

    all_users = list(sequences_by_user)
    if len(all_users) < n_splits:
        raise ValueError(f"Only {len(all_users)} eligible students for {n_splits} folds.")
    test_folds = _split_outer_user_folds(all_users, n_splits, split_seed)
    test_users = test_folds[fold]
    test_set = set(test_users)
    remaining = [user for user in all_users if user not in test_set]
    random.Random(split_seed + fold + 1).shuffle(remaining)
    valid_count = max(1, round(len(all_users) * valid_ratio))
    valid_users = remaining[:valid_count]
    train_users = remaining[valid_count:]

    def make_dataset(users):
        return KLTDataset(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            return_type=return_type,
            num_skills=num_skills,
            num_questions=num_questions,
            sequences=[sequences_by_user[user] for user in users],
        )

    datasets = {
        "train": make_dataset(train_users),
        "valid": make_dataset(valid_users),
        "test": make_dataset(test_users),
    }
    split_info = {
        "eligible_students": len(all_users),
        "train_students": len(train_users),
        "valid_students": len(valid_users),
        "test_students": len(test_users),
        "fold": fold,
        "n_splits": n_splits,
        "split_seed": split_seed,
    }
    return datasets, split_info


def collate_fn(batch, pad_val=0):
    p_list = [torch.tensor(item["p"], dtype=torch.long) for item in batch]
    c_list = [torch.tensor(item["s"], dtype=torch.long) for item in batch]
    r_list = [torch.tensor(item["r"], dtype=torch.long) for item in batch]

    p_padded = pad_sequence(p_list, batch_first=True, padding_value=pad_val)
    c_padded = pad_sequence(c_list, batch_first=True, padding_value=pad_val)
    r_padded = pad_sequence(r_list, batch_first=True, padding_value=pad_val)

    p = p_padded[:, :-1]
    c = c_padded[:, :-1]
    r = r_padded[:, :-1]

    p_shft = p_padded[:, 1:]
    c_shft = c_padded[:, 1:]
    r_shft = r_padded[:, 1:]

    mask = (c != pad_val) & (c_shft != pad_val)

    return p, c, r, p_shft, c_shft, r_shft, mask


def seq_to_fixed_array(seq, maxstep=200):
    """
    把 KLTDataset 里的一个学生序列转成 [maxstep, 3]。
    注意：这里和 es_re.py 必须一致：
    - 长序列取最后 maxstep
    - 右侧 padding 0
    """
    p_seq = seq["p"]
    s_seq = seq["s"]
    r_seq = seq["r"]

    if len(p_seq) > maxstep:
        p_seq = p_seq[-maxstep:]
        s_seq = s_seq[-maxstep:]
        r_seq = r_seq[-maxstep:]

    cur_len = len(p_seq)

    arr = np.zeros((maxstep, 3), dtype=np.int64)
    arr[:cur_len, 0] = p_seq
    arr[:cur_len, 1] = s_seq
    arr[:cur_len, 2] = r_seq

    return arr


class CollabKLTDataset(KLTDataset):
    """
    基于 KLTDataset 的协同版本。

    它的读取逻辑和 KLTDataset 一致，
    但是额外读取 similar_records，并返回 peer 信息。
    """

    def __init__(
        self,
        file_path,
        init_file_path,
        similar_records_path,
        min_seq_len=3,
        max_seq_len=200,
        top_k=12,
        peer_maxstep=50,
    ):
        super().__init__(
            file_path=file_path,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            return_type="standard",
        )

        self.top_k = top_k
        self.peer_maxstep = peer_maxstep

        # 训练集知识库，用于根据 peer_student_idx 截取同伴历史
        self.init_dataset = KLTDataset(
            file_path=init_file_path,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            return_type="standard",
        )

        self.init_data = np.asarray(
            [
                seq_to_fixed_array(seq, maxstep=max_seq_len)
                for seq in self.init_dataset.sequences
            ],
            dtype=np.int64,
        )

        with open(similar_records_path, "rb") as f:
            self.similar_records = pickle.load(f)

        if len(self.sequences) != len(self.similar_records):
            raise ValueError(
                f"数据数量和 similar_records 数量不一致: "
                f"data={len(self.sequences)}, "
                f"similar_records={len(self.similar_records)}.\n"
                f"file_path={file_path}\n"
                f"similar_records_path={similar_records_path}\n"
                f"请确认 es_re.py 和 DataSet.py 的 min_seq_len / max_seq_len / 截断方式完全一致。"
            )

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        myself_np = seq_to_fixed_array(
            seq,
            maxstep=self.max_seq_len,
        )

        similar_records = self.similar_records[idx]

        topK_sub_seq, len_sub = self.top_k_sub_seq(similar_records)

        myself = torch.from_numpy(myself_np).long()
        others = torch.from_numpy(topK_sub_seq).long()
        other_len = torch.from_numpy(len_sub).long()

        return myself, others, other_len

    def top_k_sub_seq(self, similar_records):
        len_sub_seq = np.zeros(
            [self.max_seq_len, self.top_k],
            dtype=np.int64,
        )

        one_student = np.zeros(
            [self.max_seq_len, self.top_k, self.peer_maxstep, 3],
            dtype=np.int64,
        )

        for step_idx, peers in enumerate(similar_records):
            if step_idx >= self.max_seq_len:
                break

            if peers is None:
                continue

            for peer_rank, peer_info in enumerate(peers):
                if peer_rank >= self.top_k:
                    break

                if peer_info is None or len(peer_info) < 2:
                    continue

                peer_student_idx = int(peer_info[0])
                peer_step = int(peer_info[1])

                if peer_student_idx < 0 or peer_student_idx >= len(self.init_data):
                    continue

                if peer_step < 0:
                    continue

                end_idx = peer_step + 1
                start_idx = max(0, end_idx - self.peer_maxstep)

                peer_seq = self.init_data[
                    peer_student_idx,
                    start_idx:end_idx,
                    :,
                ]

                copy_len = min(peer_seq.shape[0], self.peer_maxstep)

                if copy_len <= 0:
                    continue

                len_sub_seq[step_idx, peer_rank] = copy_len
                one_student[
                    step_idx,
                    peer_rank,
                    :copy_len,
                    :,
                ] = peer_seq[:copy_len]

        return one_student, len_sub_seq


def collab_collate_fn(batch, pad_val=0):
    myself_list, others_list, other_len_list = zip(*batch)

    myself_padded = torch.stack(myself_list, dim=0)
    others_padded = torch.stack(others_list, dim=0)
    other_len_padded = torch.stack(other_len_list, dim=0)

    p = myself_padded[..., 0]
    c = myself_padded[..., 1]
    r = myself_padded[..., 2]

    p_in = p[:, :-1]
    c_in = c[:, :-1]
    r_in = r[:, :-1]

    p_shft = p[:, 1:]
    c_shft = c[:, 1:]
    r_shft = r[:, 1:]

    mask_shft = c_shft != pad_val
    mask_raw = c != pad_val

    return {
        "self_x_in": (p_in, c_in, r_in),
        "self_c_target": c_shft,
        "self_y_target": r_shft,
        "mask_target": mask_shft,

        "self_x_raw": (p, c, r),
        "self_y_raw": r,
        "mask_raw": mask_raw,

        "peer_x": others_padded,
        "peer_len": other_len_padded,
    }


def get_collab_dataloader(
    batch_size,
    init_data_path,
    train_data_path,
    valid_data_path,
    test_data_path,
    train_sim_path,
    valid_sim_path,
    test_sim_path,
    maxstep=200,
    top_k=12,
    peer_maxstep=50,
    min_seq_len=3,
):
    train_dataset = CollabKLTDataset(
        file_path=train_data_path,
        init_file_path=init_data_path,
        similar_records_path=train_sim_path,
        min_seq_len=min_seq_len,
        max_seq_len=maxstep,
        top_k=top_k,
        peer_maxstep=peer_maxstep,
    )

    valid_dataset = CollabKLTDataset(
        file_path=valid_data_path,
        init_file_path=init_data_path,
        similar_records_path=valid_sim_path,
        min_seq_len=min_seq_len,
        max_seq_len=maxstep,
        top_k=top_k,
        peer_maxstep=peer_maxstep,
    )

    test_dataset = CollabKLTDataset(
        file_path=test_data_path,
        init_file_path=init_data_path,
        similar_records_path=test_sim_path,
        min_seq_len=min_seq_len,
        max_seq_len=maxstep,
        top_k=top_k,
        peer_maxstep=peer_maxstep,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collab_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collab_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collab_collate_fn,
    )

    return train_loader, valid_loader, test_loader
