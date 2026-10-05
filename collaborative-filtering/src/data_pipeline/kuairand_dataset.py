# -*- coding: utf-8 -*-
"""
KuaiRand-Pure logged-exposure dataset with IsClick as the sole target.
Both standard logs train the models; filtered random-log users are split into
validation and held-out test following the released iDCF preprocessing.

TopTop Project (agri-behavioral-recsys)
"""

import os
import sys
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from src.utils.helpers import PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_dataset_dir


# =============================================================================
# 1. NHÃN CLICK
# =============================================================================

def compute_click_scores(df: pd.DataFrame, alpha: float = 0.0):
    """Return click labels and optional positive weights for logged exposures."""
    if "is_click" not in df.columns:
        raise ValueError("KuaiRand log is missing required is_click column")
    labels = df["is_click"].to_numpy(dtype=np.float32)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("KuaiRand is_click must contain only 0 or 1")
    weights = (1.0 + alpha * labels).astype(np.float32)
    return labels, labels, weights


# =============================================================================
# 2. LỚP PYTORCH DATASET CHỈ CHỨA EXPOSURE ĐÃ QUAN SÁT
# =============================================================================

class LoggedClickDataset(Dataset):
    """Click/no-click exposures recorded in the standard log."""
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray,
                 targets: np.ndarray, confidences: np.ndarray):
        self.users = user_indices.astype(np.int64)
        self.items = item_indices.astype(np.int64)
        self.targets = targets.astype(np.float32)
        self.confidences = confidences.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.targets[idx],
            self.confidences[idx]
        )


def collate_click_fn(batch):
    """Collate function chuyển đổi mini-batch tuple thành các PyTorch Tensor chuẩn."""
    u, i, y, c = zip(*batch)
    return (
        torch.tensor(u, dtype=torch.long),
        torch.tensor(i, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(c, dtype=torch.float32)
    )


# =============================================================================
# 3. NẠP DỮ LIỆU KUAIRAND-PURE THEO GIAO THỨC IDCF
# =============================================================================

def load_kuairand_processed(
    alpha: float = 0.0,
    data_dir: str = None,
    cache_dir: str = None,
):
    """
    Use both Pure standard logs, one-pass >=10 user/item filtering, IsClick,
    and the iDCF code's seeded 30/70 random split by user.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    data_dir = resolve_dataset_dir("kuairand_pure", data_dir)

    train_path = os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")
    second_train_path = os.path.join(data_dir, "log_standard_4_22_to_5_08_pure.csv")
    test_path = os.path.join(data_dir, "log_random_4_22_to_5_08_pure.csv")
    if cache_dir is None:
        cache_dir = os.path.abspath(os.path.join(PROCESSED_DATA_DIR, "kuairand"))
    os.makedirs(cache_dir, exist_ok=True)
    signature = {
        "version": 7,
        "protocol": "paper_idcf",
        "alpha": alpha,
        "random_val_fraction": 0.3,
        "label": "is_click",
        "files": [(os.path.abspath(p), os.path.getsize(p), os.stat(p).st_mtime_ns)
                  for p in (train_path, second_train_path, test_path)],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache_file = os.path.join(cache_dir, f"kuairand_v{signature['version']}_{digest}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if (np.any(cached["train_u"] < 0) or np.any(cached["train_u"] >= cached["num_users"])
                or np.any(cached["train_i"] < 0) or np.any(cached["train_i"] >= cached["num_items"])):
            raise ValueError(f"Invalid mapped IDs in cache: {cache_file}")
        return cached

    print(f"[Data Pipeline] Đang nạp Biased Train Set từ: {train_path} ...")
    df_train = pd.read_csv(train_path)
    df_train = pd.concat([df_train, pd.read_csv(second_train_path)], ignore_index=True)

    print(f"[Data Pipeline] Đang nạp Unbiased Test Set từ: {test_path} ...")
    df_test = pd.read_csv(test_path)
    # Matches the released iDCF notebook: filter users first, then items once.
    user_counts = df_train["user_id"].value_counts()
    df_train = df_train[df_train["user_id"].isin(user_counts[user_counts >= 10].index)]
    item_counts = df_train["video_id"].value_counts()
    df_train = df_train[df_train["video_id"].isin(item_counts[item_counts >= 10].index)]
    df_test = df_test[df_test["user_id"].isin(df_train["user_id"].unique())
                           & df_test["video_id"].isin(df_train["video_id"].unique())]
    # The released evaluator splits unique random-log users with seed 1234.
    random_users = df_test["user_id"].unique()
    rng = np.random.RandomState(1234)
    test_users = rng.choice(random_users, size=int(len(random_users) * 0.7), replace=False)
    is_test = df_test["user_id"].isin(test_users)
    df_val, df_test = df_test[~is_test].copy(), df_test[is_test].copy()

    # 1. Mã hóa ID người dùng và video liên tục [0..N-1] và [0..M-1]
    all_users = np.unique(np.concatenate([df_train['user_id'].unique(), df_val['user_id'].unique(), df_test['user_id'].unique()]))
    all_videos = np.unique(np.concatenate([df_train['video_id'].unique(), df_val['video_id'].unique(), df_test['video_id'].unique()]))

    user_mapping = {uid: idx for idx, uid in enumerate(all_users)}
    video_mapping = {vid: idx for idx, vid in enumerate(all_videos)}

    num_users = len(user_mapping)
    num_items = len(video_mapping)
    print(f"[Data Pipeline] Không gian thực nghiệm: {num_users:,} Users | {num_items:,} Videos")

    # Keep all logged exposures, including known negatives. Unknown pairs have no label.
    train_r, train_y, train_c = compute_click_scores(df_train, alpha=alpha)
    train_u = df_train['user_id'].map(user_mapping).values.astype(np.int64)
    train_i = df_train['video_id'].map(video_mapping).values.astype(np.int64)
    if (np.any(train_u < 0) or np.any(train_u >= num_users)
            or np.any(train_i < 0) or np.any(train_i >= num_items)):
        raise ValueError("Train contains unmapped user or video IDs")

    # Positive map is used only for optional negative sampling and seen-item masking.
    train_user_pos_map = {}
    for u, i in zip(train_u[train_y > 0], train_i[train_y > 0]):
        if u not in train_user_pos_map:
            train_user_pos_map[u] = set()
        train_user_pos_map[u].add(i)

    # Ma trận sparse biểu diễn train
    train_sparse = csr_matrix((train_y[train_y > 0], (train_u[train_y > 0], train_i[train_y > 0])), shape=(num_users, num_items))
    train_sparse.data[:] = 1.0

    def make_pool(frame):
        labels = compute_click_scores(frame, alpha=alpha)[0]
        pairs = pd.DataFrame({
            "u": frame["user_id"].map(user_mapping).to_numpy(dtype=np.int64),
            "i": frame["video_id"].map(video_mapping).to_numpy(dtype=np.int64),
            "y": labels,
        })
        # Repeated logged pairs count as positive if any click occurred.
        pairs = pairs.groupby(["u", "i"], as_index=False)["y"].max()
        candidates, truth = {}, {}
        for u, group in pairs.groupby("u", sort=False):
            u = int(u)
            candidates[u] = group["i"].to_numpy(dtype=np.int64)
            truth[u] = set(group.loc[group["y"] > 0, "i"].astype(int))
        return candidates, truth

    val_candidates, val_ground_truth = make_pool(df_val)
    test_candidates, test_ground_truth = make_pool(df_test)
    print(f"[Data Pipeline] Random validation: {len(val_candidates):,} users; held-out test: {len(test_candidates):,} users")

    data_payload = {
        "data_dir": data_dir,
        "num_users": num_users,
        "n_users": num_users,
        "num_items": num_items,
        "n_items": num_items,
        "train_u": train_u,
        "train_i": train_i,
        "train_y": train_y,
        "train_c": train_c,
        "train_r": train_r,
        "train_sparse": train_sparse,
        "train_user_pos_map": train_user_pos_map,
        "val_candidates": val_candidates,
        "val_ground_truth": val_ground_truth,
        "test_candidates": test_candidates,
        "test_ground_truth": test_ground_truth,
        "eval_users": np.array(sorted(test_candidates), dtype=np.int64),
        "train_item_exposure_counts": np.bincount(train_i, minlength=num_items),
        "evaluation_include_zero_positive_users": False,
        "split_info": {
            "protocol": "paper_idcf",
            "split_strategy": "seeded_by_user_1234",
            "random_val_fraction": 0.3,
            "val_rows": len(df_val), "test_rows": len(df_test),
            "train_rows": len(df_train),
            "val_users": df_val["user_id"].nunique(), "test_users": df_test["user_id"].nunique(),
            "val_max_time_ms": None,
            "test_min_time_ms": None,
            "label": signature["label"],
        },
        "cache_signature": signature,
        "user_mapping": user_mapping,
        "video_mapping": video_mapping
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Data Pipeline] Đã lưu bộ đệm KuaiRand is_click tại: {cache_file}")
    except Exception as e:
        print(f"[Data Pipeline] Lưu bộ đệm thất bại ({e}), bỏ qua.")

    return data_payload


# =============================================================================
# 4. TRÍCH XUẤT ĐẶC TRƯNG NGOẠI SINH CHO IViDR (INSTRUMENTS & PROXIES)
# =============================================================================

def load_kuairand_features(n_users, n_items, user_mapping, video_mapping, feat_dim=32, data_dir=None):
    """
    Trích xuất ma trận đặc trưng người dùng (Z_u) và biến Proxy (W_u) cho KuaiRand.
    Được chuẩn hóa Z-score an toàn, phục vụ kiến trúc mạng iVAE trong IViDR.
    """
    data_dir = resolve_dataset_dir("kuairand_pure", data_dir)
    user_feat_file = os.path.join(data_dir, "user_features_pure.csv")

    u_features = np.zeros((n_users, feat_dim), dtype=np.float32)
    proxy_w = np.zeros((n_users, 8), dtype=np.float32)

    raw_to_uidx = user_mapping  # load_kuairand_processed always maps raw user ID -> embedding index

    if os.path.exists(user_feat_file):
        try:
            udf = pd.read_csv(user_feat_file)
            udf_mapped = udf[udf["user_id"].isin(raw_to_uidx)].copy()
            if udf_mapped["user_id"].duplicated().any() or set(udf_mapped["user_id"]) != set(raw_to_uidx):
                raise ValueError("User features must cover each mapped user exactly once")
            udf_mapped["u_idx"] = udf_mapped["user_id"].map(raw_to_uidx)

            candidate_cols = [
                "user_active_degree", "is_lowactive_period", "is_live_streamer",
                "is_video_author", "follow_user_num_range", "fans_user_num_range",
                "friend_user_num_range", "register_days_range"
            ]
            avail_cols = [c for c in candidate_cols if c in udf_mapped.columns]
            if not avail_cols:
                raise ValueError("No supported user feature columns found")

            if avail_cols:
                num_features = []
                for col in avail_cols:
                    s = udf_mapped[col]
                    if not pd.api.types.is_numeric_dtype(s):
                        codes = pd.Categorical(s).codes.astype(np.float32)
                        num_features.append(codes)
                    else:
                        num_features.append(s.fillna(0).values.astype(np.float32))

                feat_vals = np.column_stack(num_features)
                mean = np.mean(feat_vals, axis=0, keepdims=True)
                std = np.std(feat_vals, axis=0, keepdims=True) + 1e-6
                norm_vals = (feat_vals - mean) / std

                u_indices = udf_mapped["u_idx"].values.astype(int)
                valid_mask = u_indices < n_users
                u_indices = u_indices[valid_mask]
                norm_vals = norm_vals[valid_mask]

                dim_fill = min(norm_vals.shape[1], feat_dim)
                u_features[u_indices, :dim_fill] = norm_vals[:, :dim_fill]
                proxy_w[u_indices, :min(dim_fill, 8)] = norm_vals[:, :min(dim_fill, 8)]

                print(f"  -> Nạp thành công đặc trưng KuaiRand: {feat_vals.shape[1]} thuộc tính cho {len(u_indices):,} users.")
        except Exception as e:
            raise ValueError(f"Unable to load KuaiRand user features: {user_feat_file}") from e
    else:
        raise FileNotFoundError(user_feat_file)

    return u_features, proxy_w


def subset_kuairand_users(data, max_users, seed=42):
    """Small, explicitly marked smoke subset; never a paper benchmark."""
    if max_users < 2:
        raise ValueError("Smoke subset requires at least two users")
    users = np.sort(np.random.default_rng(seed).choice(
        np.unique(data["train_u"]), min(max_users, data["n_users"]), replace=False))
    mask = np.isin(data["train_u"], users)
    items = np.unique(data["train_i"][mask])
    umap = {int(u): idx for idx, u in enumerate(users)}
    imap = {int(i): idx for idx, i in enumerate(items)}
    out = dict(data)
    out.update(n_users=len(users), num_users=len(users), n_items=len(items), num_items=len(items))
    for key in ("train_u", "train_i", "train_y", "train_c", "train_r"):
        out[key] = data[key][mask].copy()
    out["train_u"] = np.array([umap[int(u)] for u in out["train_u"]], dtype=np.int64)
    out["train_i"] = np.array([imap[int(i)] for i in out["train_i"]], dtype=np.int64)
    for split in ("val", "test"):
        out[f"{split}_candidates"] = {umap[u]: np.array([imap[int(i)] for i in pool if int(i) in imap], dtype=np.int64)
                                        for u, pool in data[f"{split}_candidates"].items() if u in umap}
        out[f"{split}_ground_truth"] = {umap[u]: {imap[int(i)] for i in pool if int(i) in imap}
                                        for u, pool in data[f"{split}_ground_truth"].items() if u in umap}
    out["user_mapping"] = {raw: umap[idx] for raw, idx in data["user_mapping"].items() if idx in umap}
    out["video_mapping"] = {raw: imap[idx] for raw, idx in data["video_mapping"].items() if idx in imap}
    out["train_item_exposure_counts"] = np.bincount(out["train_i"], minlength=len(items))
    out["train_user_pos_map"] = {u: set(out["train_i"][(out["train_u"] == u) & (out["train_y"] > 0)]) for u in range(len(users))}
    pos = out["train_y"] > 0
    out["train_sparse"] = csr_matrix((np.ones(pos.sum()), (out["train_u"][pos], out["train_i"][pos])), shape=(len(users),len(items)))
    out["train_sparse"].data[:] = 1
    out["eval_users"] = np.array(sorted(out["test_candidates"]), dtype=np.int64)
    out["split_info"] = dict(data["split_info"], smoke_only=True, smoke_seed=seed,
                             train_rows=len(out["train_u"]), val_users=len(out["val_candidates"]),
                             test_users=len(out["test_candidates"]),
                             val_rows=sum(map(len,out["val_candidates"].values())),
                             test_rows=sum(map(len,out["test_candidates"].values())))
    return out
