"""Versioned experiment identity and strict checkpoint loading."""
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from src.evaluation.protocol import METRIC_REVISION, protocol_metadata, validate_paper_split

VERSION = 4
ARCHITECTURES = {"mf": "mf_v4", "cdr": "cdr_v4", "ividr": "ividr_v5"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()[:16]


def validate_training(**values):
    for name, value in values.items():
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive, got {value}")


def data_signature(data):
    """Hash actual train arrays and evaluation pools, including small/subset runs."""
    h = hashlib.sha256()
    for key in ("train_u", "train_i", "train_y", "train_c"):
        if key in data:
            a = np.ascontiguousarray(data[key])
            h.update(key.encode())
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
    for split in ("val", "test"):
        for kind in ("candidates", "ground_truth"):
            pool = data[f"{split}_{kind}"]
            for user in sorted(pool):
                a = np.asarray(sorted(pool[user]), dtype=np.int64)
                h.update(f"{split}:{kind}:{user}:{len(a)}".encode())
                h.update(a.tobytes())
    return {"sha256": h.hexdigest(), "source": data.get("cache_signature", {}),
            "split": data.get("split_info", {}),
            "data_dir": data.get("data_dir"),
            "n_users": int(data.get("n_users", data.get("num_users"))),
            "n_items": int(data.get("n_items", data.get("num_items")))}


def source_signature():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for folder in ("src", "experiments"):
        for path in sorted((root / folder).rglob("*.py")):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def make_metadata(model_name, dataset, data, config, model_kwargs):
    validate_paper_split(data)
    if not any(data["val_ground_truth"].get(user) for user, items in data["val_candidates"].items() if len(items)):
        raise ValueError("Validation must contain a positive item for ranking-based checkpoint selection")
    return {"artifact_version": VERSION, "architecture_version": ARCHITECTURES[model_name],
            "model": model_name, "dataset": dataset, "config": config,
            "model_kwargs": model_kwargs, "data": data_signature(data),
            "code_sha256": source_signature(),
            "protocol": protocol_metadata(data),
            "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__)}}


def checkpoint_path(directory, name, metadata):
    path = Path(directory) / f"{name}_v4_{digest(metadata)}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_checkpoint(model, path, metadata, epoch, validation):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save({"metadata": metadata, "model_state_dict": model.state_dict(),
                    "epoch": epoch, "validation": validation,
                    "embedding_dim": metadata["model_kwargs"]["embedding_dim"]}, stream)
    replace_with_retry(temporary, path)


def replace_with_retry(source, destination):
    """Keep the old file intact when Windows briefly locks an atomic replace."""
    for attempt in range(8):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def load_checkpoint(path, device="cpu", expected_metadata=None):
    state = torch.load(path, map_location=device, weights_only=True)
    metadata = state.get("metadata", {})
    if metadata.get("artifact_version") != VERSION:
        raise ValueError(f"Checkpoint incompatible with v{VERSION}: {path}. Train a new checkpoint; old files are retained.")
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError(f"Checkpoint configuration/data mismatch: {path}")
    return state


def evaluate_checkpoint(path, data, device=None):
    """Evaluate the selected checkpoint without training or reselecting an epoch."""
    from src.models.matrix_factorization import MatrixFactorization
    from src.causal.cdr.cdr_model import CDRFramework
    from src.causal.ividr.ividr_model import IViDRModel
    from src.evaluation.evaluator import evaluate_unbiased_model

    device = device or torch.device("cpu")
    state = load_checkpoint(path, device)
    meta = state["metadata"]
    if meta.get("protocol", {}).get("metric_revision") != METRIC_REVISION:
        raise ValueError("Checkpoint uses an older evaluation protocol; train a new checkpoint")
    validate_paper_split(data)
    if meta["data"] != data_signature(data):
        raise ValueError("Selected checkpoint does not match evaluation data")
    classes = {"mf": MatrixFactorization, "cdr": CDRFramework, "ividr": IViDRModel}
    if meta.get("architecture_version") != ARCHITECTURES[meta['model']]:
        raise ValueError("Incompatible architecture version")
    model = classes[meta["model"]](**meta["model_kwargs"]).to(device)
    if meta["model"] == "ividr":
        for name in ("context_features", "history_items", "history_offsets"):
            setattr(model, name, state["model_state_dict"][name])
        if meta["model_kwargs"].get("projection_mode", "item_pinv") == "item_pinv":
            for name in ("bases", "ranks", "owners", "means"):
                setattr(model.iv_recon, name, state["model_state_dict"][f"iv_recon.{name}"])
    model.load_state_dict(state["model_state_dict"])
    result = evaluate_unbiased_model(
        model, data["test_candidates"], data["test_ground_truth"], device=device,
        compute_user_auc=True, k=meta["protocol"]["k"],
        include_zero_positive_users=meta["protocol"]["include_zero_positive_users"],
    )
    result.update(checkpoint_path=str(path), best_epoch=state["epoch"],
                  best_val_ndcg=state["validation"]["NDCG@5"], metadata=meta)
    return result
