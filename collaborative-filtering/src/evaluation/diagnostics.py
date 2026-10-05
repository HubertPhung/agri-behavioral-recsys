"""Candidate-pool baselines and diagnostics for logged-exposure ranking."""

import numpy as np

from src.evaluation.evaluator import evaluate_unbiased_model


class PopularityModel:
    def __init__(self, train_items, num_users, num_items):
        counts = np.bincount(train_items, minlength=num_items).astype(np.float32)
        self.user_factors = np.ones((num_users, 1), dtype=np.float32)
        self.item_factors = np.log1p(counts)[:, None]

    def eval(self):
        return self

    def get_evaluation_factors(self):
        return self.user_factors, self.item_factors


def evaluate_popularity(data, split="test", k=5):
    model = PopularityModel(data["train_i"], data["n_users"], data["n_items"])
    return evaluate_unbiased_model(
        model, data[f"{split}_candidates"], data[f"{split}_ground_truth"],
        k=k, compute_user_auc=True,
        include_zero_positive_users=data.get("evaluation_include_zero_positive_users", True),
    )


def candidate_pool_diagnostics(data, split="test", k=5):
    if k < 1:
        raise ValueError("k must be positive")
    candidates = data[f"{split}_candidates"]
    truth = data[f"{split}_ground_truth"]
    users = [u for u, items in candidates.items() if len(items)]
    sizes = np.array([len(candidates[u]) for u in users], dtype=np.int64)
    positives = np.array([len(truth.get(u, ())) for u in users], dtype=np.int64)
    fractions = positives / sizes
    discounts = 1 / np.log2(np.arange(k) + 2)
    expected = {name: [] for name in ("ndcg", "recall", "precision", "hitrate")}
    for n, m, p in zip(sizes, positives, fractions):
        top = min(k, n)
        idcg = discounts[:min(k, m)].sum()
        expected["ndcg"].append(float(p * discounts[:top].sum() / idcg) if m else 0.0)
        expected["recall"].append(float(top / n) if m else 0.0)
        expected["precision"].append(float(p * top / k))
        miss = np.prod([(n - m - j) / (n - j) for j in range(top)])
        expected["hitrate"].append(float(1 - miss))
    train_users = set(map(int, data["train_u"]))
    train_items = set(map(int, data["train_i"]))
    cold_candidates = sum(int(i) not in train_items for u in users for i in candidates[u])
    eval_mask = positives > 0 if not data.get("evaluation_include_zero_positive_users", True) else np.ones(len(users), dtype=bool)
    return {
        "users": len(users),
        "zero_positive_users": int(np.sum(positives == 0)),
        "evaluated_users": int(eval_mask.sum()),
        "candidate_pairs": int(sizes.sum()),
        "positive_pairs": int(positives.sum()),
        "pooled_positive_rate": float(positives.sum() / sizes.sum()) if sizes.sum() else 0.0,
        "mean_user_positive_rate": float(fractions.mean()) if len(fractions) else 0.0,
        "candidate_size_p10_p50_p90": np.quantile(sizes, [0.1, 0.5, 0.9]).tolist() if len(sizes) else [0.0, 0.0, 0.0],
        "users_with_fewer_than_k": int(np.sum(sizes < k)),
        "cold_user_fraction": float(np.mean([u not in train_users for u in users])) if users else 0.0,
        "cold_candidate_fraction": float(cold_candidates / sizes.sum()) if sizes.sum() else 0.0,
        "random_expected": {
            f"NDCG@{k}": float(np.mean(np.asarray(expected["ndcg"])[eval_mask])) if eval_mask.any() else 0.0,
            f"Recall@{k}": float(np.mean(np.asarray(expected["recall"])[eval_mask])) if eval_mask.any() else 0.0,
            f"Precision@{k}": float(np.mean(np.asarray(expected["precision"])[eval_mask])) if eval_mask.any() else 0.0,
            f"HitRate@{k}": float(np.mean(np.asarray(expected["hitrate"])[eval_mask])) if eval_mask.any() else 0.0,
            "AUC": 0.5,
        },
    }


def paired_ndcg_difference(model_metrics, baseline_metrics):
    """User-paired normal-approximation interval for NDCG difference."""
    if model_metrics["user_ids"] != baseline_metrics["user_ids"]:
        raise ValueError("Model and baseline must evaluate the same users in the same order")
    diff = np.asarray(model_metrics["user_ndcg"]) - np.asarray(baseline_metrics["user_ndcg"])
    halfwidth = float(1.96 * np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"mean_delta_ndcg": float(np.mean(diff)), "CI95_halfwidth": halfwidth,
            "users": len(diff)}
