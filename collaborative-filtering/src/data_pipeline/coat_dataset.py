# -*- coding: utf-8 -*-
"""
Module nạp và quản lý dữ liệu Coat Dataset cho cả Baseline, IViDR và DR-BIAS + CDR
Project: TopTop (agri-behavioral-recsys)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix

from src.utils.helpers import PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_dataset_dir


def load_coat_processed(pos_threshold=4.0, data_dir=None):
    """
    Nạp dữ liệu Coat phục vụ huấn luyện và kiểm thử.
    Theo giao thức đánh giá iDCF (KDD '23):
    Rating > 3.0 (tức >= 4.0: 4 và 5 sao) được định nghĩa là tương tác tích cực (y_ui = 1).
    Rating <= 3.0 (1, 2, 3 sao) là tương tác tiêu cực (y_ui = 0).
    Không gộp 3 sao (mức trung tính Neutral) vào positive để tránh làm sai lệch phân phối
    và thổi phồng điểm số Baseline MF.
    """
    raw = load_coat_raw_features(data_dir)
    train_matrix = raw["train_matrix"]
    test_matrix = raw["test_matrix"]
    n_users, n_items = train_matrix.shape

    train_u, train_i = np.where(train_matrix > 0)
    train_y = (train_matrix[train_u, train_i] >= pos_threshold).astype(np.float32)
    # iDCF splits the random-exposure log by user, with 70% held out for test.
    # Keep the original user order (0..n_users-1) and RandomState seed 1234
    # to match utils.split_by_user in the authors' implementation.
    random_users = np.flatnonzero(np.any(test_matrix > 0, axis=1))
    test_users = np.random.RandomState(1234).choice(
        random_users, size=int(len(random_users) * 0.7), replace=False
    )
    test_user_set = set(map(int, test_users))
    val_candidates = {}
    val_ground_truth = {}
    test_candidates = {}
    test_ground_truth = {}
    for u in random_users:
        cands = np.where(test_matrix[u] > 0)[0]
        if len(cands) > 0:
            candidates = test_candidates if u in test_user_set else val_candidates
            ground_truth = test_ground_truth if u in test_user_set else val_ground_truth
            candidates[int(u)] = cands
            ground_truth[int(u)] = set(cands[test_matrix[u, cands] >= pos_threshold])

    positive = train_y > 0
    overlap = (train_matrix > 0) & (test_matrix > 0)
    test_overlap = sum(int(overlap[u, items].sum()) for u, items in test_candidates.items())
    validation_overlap = sum(int(overlap[u, items].sum()) for u, items in val_candidates.items())
    test_overlap_positive = sum(
        int(np.logical_and(overlap[u, items], test_matrix[u, items] >= pos_threshold).sum())
        for u, items in test_candidates.items()
    )
    train_sparse = csr_matrix((train_y[positive], (train_u[positive], train_i[positive])), shape=(n_users, n_items))
    return {
        "data_dir": raw["data_dir"],
        "cache_signature": raw["source_signature"],
        "train_u": train_u,
        "train_i": train_i,
        "train_y": train_y,
        "train_sparse": train_sparse,
        "train_matrix_for_features": train_matrix,
        "val_candidates": val_candidates,
        "val_ground_truth": val_ground_truth,
        "test_candidates": test_candidates,
        "test_ground_truth": test_ground_truth,
        "evaluation_include_zero_positive_users": False,
        "split_info": {
            "protocol": "paper_idcf",
            "split_unit": "user",
            "seed": 1234,
            "validation_users": len(val_candidates),
            "test_users": len(test_candidates),
            "validation_rows": sum(map(len, val_candidates.values())),
            "test_rows": sum(map(len, test_candidates.values())),
            "train_random_overlap_pairs": int(overlap.sum()),
            "validation_overlap_pairs": validation_overlap,
            "test_overlap_pairs": test_overlap,
            "test_overlap_positive_pairs": test_overlap_positive,
        },
        "n_users": n_users,
        "n_items": n_items
    }


def load_coat_raw_features(data_dir=None):
    """
    Nạp đặc trưng gốc người dùng, vật phẩm và xác suất phơi bày (propensities) từ data/raw/coat/
    """
    coat_raw_dir = resolve_dataset_dir("coat", data_dir)
    u_feat_path = os.path.join(coat_raw_dir, "user_item_features", "user_features.ascii")
    i_feat_path = os.path.join(coat_raw_dir, "user_item_features", "item_features.ascii")
    prop_path = os.path.join(coat_raw_dir, "propensities.ascii")
    train_path = os.path.join(coat_raw_dir, "train.ascii")
    test_path = os.path.join(coat_raw_dir, "test.ascii")

    user_features = np.loadtxt(u_feat_path, dtype=np.float32)  # (290, 14)
    item_features = np.loadtxt(i_feat_path, dtype=np.float32)  # (300, 33)
    propensities = np.loadtxt(prop_path, dtype=np.float32)    # (290, 300)
    train_matrix = np.loadtxt(train_path, dtype=np.float32)    # (290, 300)
    test_matrix = np.loadtxt(test_path, dtype=np.float32)      # (290, 300)

    # Đảm bảo propensities không bị số 0 để tránh chia cho 0
    propensities = np.clip(propensities, 1e-4, 1.0)

    return {
        "data_dir": coat_raw_dir,
        "source_signature": {"files": [(p, os.path.getsize(p), os.stat(p).st_mtime_ns)
                                      for p in (u_feat_path, i_feat_path, prop_path, train_path, test_path)]},
        "user_features": user_features,
        "item_features": item_features,
        "propensities": propensities,
        "train_matrix": train_matrix,
        "test_matrix": test_matrix
    }


class CoatCausalDataset(Dataset):
    """
    Dataset phục vụ huấn luyện Causal Debiasing (IViDR & DR-BIAS + CDR).
    Chứa toàn bộ các cặp (u, i), nhãn nhị phân r_ui, chỉ số phơi bày o_ui,
    xác suất phơi bày p_ui, đặc trưng người dùng Z_u, đặc trưng vật phẩm T_i,
    và đặc trưng proxy W_u.
    """
    def __init__(self, mode="train", pos_threshold=4.0, train_matrix_override=None, data_dir=None):
        raw = load_coat_raw_features(data_dir)
        train_matrix = train_matrix_override if train_matrix_override is not None else raw["train_matrix"]
        propensities = raw["propensities"]
        user_features = raw["user_features"]
        item_features = raw["item_features"]

        n_users, n_items = train_matrix.shape
        self.n_users = n_users
        self.n_items = n_items
        self.pos_threshold = pos_threshold

        # Đặc trưng proxy W: mức độ tương tác và đặc trưng nhân khẩu học trung bình
        interaction_counts = (train_matrix > 0).sum(axis=1, keepdims=True).astype(np.float32)
        proxy_w = np.hstack([user_features, interaction_counts])  # (290, 15)

        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.item_features = torch.tensor(item_features, dtype=torch.float32)
        self.proxy_w = torch.tensor(proxy_w, dtype=torch.float32)

        # Tạo danh sách các cặp quan sát (o_ui = 1) và không quan sát (o_ui = 0)
        # Trong Coat, train_matrix > 0 là observed ratings (6960 cặp)
        users = []
        items = []
        ratings = []
        observed = []
        props = []

        for u in range(n_users):
            for i in range(n_items):
                r = train_matrix[u, i]
                p = propensities[u, i]
                is_obs = 1.0 if r > 0 else 0.0
                bin_r = 1.0 if r >= pos_threshold else 0.0

                users.append(u)
                items.append(i)
                ratings.append(bin_r)
                observed.append(is_obs)
                props.append(p)

        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.observed = torch.tensor(observed, dtype=torch.float32)
        self.props = torch.tensor(props, dtype=torch.float32)

        # Lọc danh sách chỉ các mẫu đã quan sát cho các mô hình tối ưu trên quan sát
        obs_mask = self.observed == 1.0
        self.obs_users = self.users[obs_mask]
        self.obs_items = self.items[obs_mask]
        self.obs_ratings = self.ratings[obs_mask]
        self.obs_props = self.props[obs_mask]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
            "observed": self.observed[idx],
            "propensity": self.props[idx],
            "user_feat": self.user_features[self.users[idx]],
            "item_feat": self.item_features[self.items[idx]],
            "proxy_w": self.proxy_w[self.users[idx]]
        }
