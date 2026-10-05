# -*- coding: utf-8 -*-
"""
Bộ điều phối đánh giá Unbiased Top-K Re-ranking chuẩn hóa cho toàn bộ các mô hình
Project: TopTop (agri-behavioral-recsys)
"""

import time
from numbers import Integral
import numpy as np
import torch
from src.evaluation.protocol import METRIC_REVISION
from src.evaluation.metrics import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_hitrate_at_k,
    compute_auc
)


def evaluate_unbiased_model(model, test_candidates, test_ground_truth, k=5, device=None, compute_user_auc=False, return_user_ndcg=False, include_zero_positive_users=True):
    """
    Đánh giá mô hình trên tập ứng viên ngẫu nhiên không thiên kiến (Unbiased Candidate Pool).
    Tương thích với mọi mô hình PyTorch (có phương thức predict hoặc get_evaluation_factors).
    """
    if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be positive")
    for user, items in test_candidates.items():
        if not isinstance(user, Integral) or user < 0:
            raise ValueError("User IDs must be nonnegative integers")
        if any(not isinstance(item, Integral) or item < 0 for item in items):
            raise ValueError("Item IDs must be nonnegative integers")
        if len(set(items)) != len(items):
            raise ValueError(f"Duplicate candidates for user {user}; aggregate exposures in the loader")
        if not set(test_ground_truth.get(user, ())).issubset(items):
            raise ValueError(f"Ground truth outside candidate pool for user {user}")
    if any(positives and user not in test_candidates for user, positives in test_ground_truth.items()):
        raise ValueError("Ground truth user missing from candidate pool")
    if hasattr(model, "eval"):
        model.eval()

    # Kiểm tra xem mô hình có trích xuất được user_factors và item_factors dạng ma trận không
    # (giúp tính toán nhanh gấp 100 lần)
    u_factors = None
    i_factors = None
    if hasattr(model, "get_evaluation_factors"):
        u_factors, i_factors = model.get_evaluation_factors()
        if u_factors.ndim != 2 or i_factors.ndim != 2 or u_factors.shape[1] != i_factors.shape[1]:
            raise ValueError("Invalid evaluation factor shapes")
        for user, items in test_candidates.items():
            if user >= len(u_factors) or any(item >= len(i_factors) for item in items):
                raise ValueError("Candidate ID exceeds model dimensions")

    # Match the chosen protocol's rule for zero-positive users.
    eval_users = [u for u, items in test_candidates.items()
                  if len(items) > 0 and (include_zero_positive_users or test_ground_truth.get(u))]
    total_users = len(eval_users)

    hit_rates = []
    precisions = []
    recalls = []
    recalls_bounded = []
    ndcgs = []
    aucs = []

    t0 = time.time()
    with torch.no_grad():
        for u_idx in eval_users:
            cand_items = np.array(test_candidates[u_idx], dtype=np.int64)
            if len(cand_items) == 0:
                continue
            pos_items = set(test_ground_truth.get(u_idx, ()))

            # Tính điểm số dự đoán cho danh sách candidates
            if u_factors is not None and i_factors is not None:
                # Tính nhanh bằng tích vô hướng
                scores = np.dot(i_factors[cand_items], u_factors[u_idx])
            elif hasattr(model, "predict"):
                scores = model.predict(u_idx, cand_items, device=device)
            else:
                # Fallback qua forward PyTorch
                u_tensor = torch.tensor([u_idx] * len(cand_items), dtype=torch.long, device=device)
                i_tensor = torch.tensor(cand_items, dtype=torch.long, device=device)
                scores = model(u_tensor, i_tensor).cpu().numpy().flatten()

            if torch.is_tensor(scores):
                scores = scores.detach().cpu().numpy()
            scores = np.asarray(scores).reshape(-1)
            if len(scores) != len(cand_items) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid prediction scores for user {u_idx}")
            # Explicit local tie rule, independent of candidate input ordering.
            top_items = cand_items[np.lexsort((cand_items, -scores))[:k]]

            hits = [1 if it in pos_items else 0 for it in top_items]
            num_hits = sum(hits)
            ideal_hits = min(len(pos_items), k)

            hit_rates.append(compute_hitrate_at_k(num_hits))
            precisions.append(compute_precision_at_k(num_hits, k=k))
            recalls.append(compute_recall_at_k(num_hits, len(pos_items), k=k, bounded=False))
            recalls_bounded.append(compute_recall_at_k(num_hits, len(pos_items), k=k, bounded=True))
            ndcgs.append(compute_ndcg_at_k(hits, ideal_hits, k=k))

            if compute_user_auc:
                pos_mask = np.array([1 if it in pos_items else 0 for it in cand_items], dtype=bool)
                pos_scores = scores[pos_mask]
                neg_scores = scores[~pos_mask]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    aucs.append(compute_auc(pos_scores, neg_scores))

    elapsed = time.time() - t0
    results = {
        "metric_revision": METRIC_REVISION,
        "candidate_users": len(test_candidates),
        "candidate_pairs": sum(map(len, test_candidates.values())),
        "empty_candidate_users": sum(len(items) == 0 for items in test_candidates.values()),
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"Recall@{k}_bounded": float(np.mean(recalls_bounded)) if recalls_bounded else 0.0,
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "eval_users": total_users,
        "zero_positive_users": sum(
            not test_ground_truth.get(u) for u, items in test_candidates.items() if len(items) > 0
        ),
        f"NDCG@{k}_CI95_halfwidth": float(1.96 * np.std(ndcgs, ddof=1) / np.sqrt(len(ndcgs))) if len(ndcgs) > 1 else 0.0,
        "elapsed_seconds": round(elapsed, 2)
    }
    if compute_user_auc:
        # AUC is auxiliary macro-user AUC, not a pooled CDR reproduction metric.
        results["AUC"] = float(np.mean(aucs)) if aucs else None
        results["auc_users"] = len(aucs)
    if return_user_ndcg:
        results["user_ids"] = [int(u) for u in eval_users]
        results["user_ndcg"] = ndcgs

    return results
