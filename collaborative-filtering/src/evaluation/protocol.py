"""One documented ranking contract shared by every benchmark workflow."""
TOP_K = 5
METRIC_REVISION = "logged_binary_v1"
PRIMARY_METRICS = ("NDCG@5", "Recall@5")
DEFAULT_SEEDS = tuple(range(42, 52))


def protocol_metadata(data=None):
    data = data or {}
    return {
        "metric_revision": METRIC_REVISION,
        "reference": "iDCF_KDD2023",
        "implementation": "idcf_aligned_local",
        "k": TOP_K,
        "primary_metrics": list(PRIMARY_METRICS),
        "candidates": "logged_exposures",
        "relevance": "binary_any_positive_per_user_item",
        "ndcg": "sum(hit_r/log2(r+1))/sum(1/log2(r+1),r=1..min(k,n_positive))",
        "recall_denominator": "all_positive_items_in_candidate_pool",
        "aggregation": "macro_by_user",
        "include_zero_positive_users": data.get("evaluation_include_zero_positive_users", True) if data else False,
        "tie_break": "score_desc_then_item_id_asc",
        "train_overlap": "retain_and_disclose",
        "selection": "validation_ndcg",
        "auc": "auxiliary_macro_by_user_with_both_classes_half_credit_for_ties",
        "seed_std_ddof": 1,
        "split": data.get("split_info", {}),
        "paper_scores_directly_comparable": False,
        "limitations": [
            "CDR paper uses 10/90 validation/test; this benchmark uses iDCF 30/70 user split.",
            "Binary duplicate aggregation and deterministic score ties are local conventions.",
            "Local model architectures, loss and tuning differ from original implementations.",
        ],
    }


def validate_paper_split(data):
    """Reject accidental protocol drift when using the declared iDCF split."""
    if data.get("split_info", {}).get("protocol") != "paper_idcf":
        return
    if data.get("evaluation_include_zero_positive_users") is not False:
        raise ValueError("iDCF ranking excludes zero-positive users")
    if set(data["val_candidates"]) & set(data["test_candidates"]):
        raise ValueError("iDCF validation/test users must be disjoint")


def summarize_seeds(rows):
    """Primary metrics only; sample std is undefined for a single seed."""
    import numpy as np
    if not rows:
        raise ValueError("No seed results to summarize")
    return {metric: {
        "mean": float(np.mean([row[metric] for row in rows])),
        "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else None,
        "n_seeds": len(rows),
    } for metric in PRIMARY_METRICS}
