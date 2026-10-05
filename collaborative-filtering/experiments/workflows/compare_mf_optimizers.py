"""Compare Adam/SGD on the SAME MF objective; select configurations on validation."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.baselines.train_mf import train_baseline_mf
from src.data_pipeline.coat_dataset import load_coat_processed
from src.data_pipeline.kuairand_dataset import load_kuairand_processed
from src.utils.artifacts import data_signature, digest, evaluate_checkpoint, source_signature, validate_training
from src.utils.helpers import RESULTS_DIR, get_device
from src.evaluation.protocol import DEFAULT_SEEDS, protocol_metadata, summarize_seeds


def compare(data, dataset, *, epochs=50, dim=64, batch_size=512, patience=10,
            loss="bce", decay_scope="embeddings", scheduler="none",
            adam_lrs=(.001, .005, .01), sgd_lrs=(.01, .1, 1.),
            weight_decays=(1e-5, 1e-4), seeds=DEFAULT_SEEDS, device=None):
    validate_training(epochs=epochs, dim=dim, batch_size=batch_size, patience=patience)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not adam_lrs or not sgd_lrs or len(adam_lrs) != len(sgd_lrs) or not weight_decays:
        raise ValueError("Use nonempty grids with the same number of learning rates per optimizer")
    for lr in (*adam_lrs, *sgd_lrs):
        validate_training(lr=lr)
    if any(not np.isfinite(wd) or wd < 0 for wd in weight_decays):
        raise ValueError("weight decays must be finite and nonnegative")
    device = device or get_device()
    common = dict(epochs=epochs, embedding_dim=dim, batch_size=batch_size,
                  patience=patience, loss_name=loss, decay_scope=decay_scope,
                  scheduler_name=scheduler)
    identity = dict(data=data_signature(data), code_sha256=source_signature(), dataset=dataset,
                    common=common, adam_lrs=adam_lrs, sgd_lrs=sgd_lrs,
                    weight_decays=weight_decays, seeds=seeds, device=str(device),
                    torch=str(torch.__version__), numpy=str(np.__version__))
    report = dict(identity=identity, status="tuning", trials={}, selected={}, repeats={},
                  evaluation_protocol=protocol_metadata(data),
                  protocol="tune_each_optimizer_on_first_seed_validation_then_freeze_config",
                  paper_reproduction=False, summary={})
    output = Path(RESULTS_DIR) / f"mf_optimizers_{dataset}_{digest(identity)}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    # Finish ALL hyperparameter selection before touching the test pools.
    for optimizer, rates in (("adam", adam_lrs), ("sgd", sgd_lrs)):
        trials = []
        for lr in rates:
            for wd in weight_decays:
                start = time.perf_counter()
                trial = train_baseline_mf(
                    dataset, data_dict=data, **common, optimizer_name=optimizer,
                    lr=lr, weight_decay=wd, seed=seeds[0], device=device,
                    evaluate_test=False, checkpoint_tag="optimizer_comparison")
                trials.append(dict(lr=lr, weight_decay=wd, seconds=time.perf_counter()-start,
                                   **trial))
        report["trials"][optimizer] = trials
        report["selected"][optimizer] = max(trials, key=lambda t: t["best_val_ndcg"])
        save()
    report["recommended_by_validation"] = max(
        report["selected"], key=lambda name: report["selected"][name]["best_val_ndcg"])
    report["status"] = "evaluating_frozen_configs"
    save()
    for optimizer, selected in report["selected"].items():
        repeats = []
        for seed in seeds:
            if seed == seeds[0]:
                path = selected["checkpoint_path"]
            else:
                trial = train_baseline_mf(
                    dataset, data_dict=data, **common, optimizer_name=optimizer,
                    lr=selected["lr"], weight_decay=selected["weight_decay"],
                    seed=seed, device=device, evaluate_test=False,
                    checkpoint_tag="optimizer_comparison")
                path = trial["checkpoint_path"]
            repeats.append(dict(seed=seed, test=evaluate_checkpoint(path, data, device)))
        report["repeats"][optimizer] = repeats
        report["summary"][optimizer] = summarize_seeds([r["test"] for r in repeats])
        save()
    report["status"] = "complete"
    report["limits"] = ["Equal trial/epoch budgets do not guarantee equal convergence or compute.",
                        "First seed participates in tuning; repeats share a fixed data split.",
                        "This is a local MF comparison, not a reproduction of paper scores."]
    save()
    print(f"Report: {output}")
    print(json.dumps(report["summary"], indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["coat", "kuairand"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--loss", choices=["bce", "mse"], default="bce")
    p.add_argument("--decay-scope", choices=["embeddings", "all"], default="embeddings")
    p.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    p.add_argument("--adam-lrs", nargs="+", type=float, default=[.001, .005, .01])
    p.add_argument("--sgd-lrs", nargs="+", type=float, default=[.01, .1, 1.])
    p.add_argument("--weight-decays", nargs="+", type=float, default=[1e-5, 1e-4])
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = p.parse_args(argv)
    kwargs = vars(args).copy()
    kwargs["batch_size"] = args.batch_size or (512 if args.dataset == "coat" else 4096)
    # Catch simple invalid runs before the potentially expensive loader.
    validate_training(epochs=args.epochs, dim=args.dim, batch_size=kwargs["batch_size"], patience=args.patience)
    data = load_coat_processed() if args.dataset == "coat" else load_kuairand_processed()
    compare(data, **kwargs)


