# -*- coding: utf-8 -*-
"""
Các chỉ số đo lường hiệu năng xếp hạng và giải trừ thiên kiến (Ranking & Debiasing Metrics)
Project: TopTop (agri-behavioral-recsys)
"""

import numpy as np


def compute_ndcg_at_k(hits, ideal_hits, k=5):
    """
    Tính Normalized Discounted Cumulative Gain tại Top-K:
      DCG@K = sum_{r=0}^{K-1} hits[r] / log2(r + 2)
      IDCG@K = sum_{r=0}^{ideal_hits-1} 1 / log2(r + 2)
    """
    if ideal_hits <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    dcg = np.sum(np.array(hits[:k]) * discounts[:len(hits[:k])])
    idcg = np.sum(discounts[:min(ideal_hits, k)])
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_recall_at_k(num_hits, total_positives, k=5, bounded=False):
    """
    Recall@K:
      - bounded=False (iDCF released utils.py: Recall_at_k_batch):
          Recall@K = num_hits / total_positives
      - bounded=True (Chuẩn Top-K Bounded):
          Recall@K = num_hits / min(total_positives, k)
    """
    denom = min(total_positives, k) if bounded else total_positives
    if denom <= 0:
        return 0.0
    return float(num_hits / float(denom))


def compute_precision_at_k(num_hits, k=5):
    """Precision@K = num_hits / K"""
    return float(num_hits / float(k))


def compute_hitrate_at_k(num_hits):
    """HitRate@K = 1 nếu num_hits > 0 else 0"""
    return 1.0 if num_hits > 0 else 0.0


def compute_auc(pos_scores, neg_scores):
    """
    Tính AUC (Area Under Curve) giữa tập điểm dương và tập điểm âm.
    AUC = P(score_pos > score_neg)
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # Sử dụng vectorization
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    diff = pos_scores[:, None] - neg_scores[None, :]
    num_greater = np.sum(diff > 0)
    num_equal = np.sum(diff == 0)
    return float((num_greater + 0.5 * num_equal) / (n_pos * n_neg))
