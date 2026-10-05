"""Compare Coat CDR inverse-propensity caps using validation only."""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from src.baselines.train_mf import train_baseline_mf
from src.causal.cdr.train_dr_bias_cdr import train_dr_bias_cdr
from src.data_pipeline.coat_dataset import load_coat_processed
from src.utils.artifacts import data_signature, digest, source_signature
from src.utils.helpers import RESULTS_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--caps", nargs="+", default=["none", "10", "20"])
    parser.add_argument("--weighting", choices=["none", "population", "both"], default="none")
    parser.add_argument("--dims", nargs="+", type=int, default=[8])
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1.0])
    parser.add_argument("--warm-start-mf", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive epochs and distinct seeds")
    try:
        caps = [None if cap.lower() == "none" else float(cap) for cap in args.caps]
    except ValueError:
        parser.error("Caps must be 'none' or positive numbers")
    if any(cap is not None and (not np.isfinite(cap) or cap < 1) for cap in caps):
        parser.error("Numeric caps must be finite and at least 1")
    if any(dim < 1 for dim in args.dims) or any(not np.isfinite(lr) or lr <= 0 for lr in args.lrs):
        parser.error("Dims and learning rates must be positive")
    if any(not np.isfinite(threshold) or threshold < 0 for threshold in args.thresholds):
        parser.error("Thresholds must be finite and nonnegative")

    data = load_coat_processed()
    weighting_options = ([False, True] if args.weighting == "both" else
                         [args.weighting == "population"])
    configs = [{"embedding_dim": dim, "lr": lr, "un_thres": threshold,
                "max_inverse_propensity": cap,
                "population_counterfactual_weight": weighting}
               for dim in args.dims for lr in args.lrs for threshold in args.thresholds
               for cap in caps
               for weighting in weighting_options]
    identity = {"dataset": "coat", "model": "cdr", "epochs": args.epochs,
                "seeds": args.seeds, "configs": configs,
                "warm_start_mf": args.warm_start_mf,
                "data": data_signature(data), "code_sha256": source_signature()}
    path = Path(RESULTS_DIR) / f"cdr_coat_propensity_tune_{digest(identity)}.json"
    log_path = path.with_suffix(".log")
    report = {"identity": identity, "selection": "mean_validation_ndcg_only",
              "test_evaluated": False, "mf_pretraining": [], "trials": [], "summary": []}

    mf_checkpoints = {}
    if args.warm_start_mf:
        for dim in sorted(set(args.dims)):
            for seed in args.seeds:
                tag = f"cdr_warm_start_{digest({'dim': dim, 'seed': seed, 'identity': identity})}"
                with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    mf_result = train_baseline_mf(
                        dataset_name="coat", data_dict=data, embedding_dim=dim,
                        epochs=50, lr=0.005, batch_size=512, patience=7,
                        evaluate_test=False, seed=seed, checkpoint_tag=tag)
                mf_checkpoints[(dim, seed)] = mf_result["checkpoint_path"]
                report["mf_pretraining"].append({"dim": dim, "seed": seed,
                    "best_val_ndcg": mf_result["best_val_ndcg"],
                    "checkpoint_path": mf_result["checkpoint_path"]})
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"MF dim={dim} seed={seed}: validation NDCG@5={mf_result['best_val_ndcg']:.4f}", flush=True)

    for config in configs:
        scores = []
        for seed in args.seeds:
            tag = f"propensity_tune_{digest({'config': config, 'seed': seed, 'epochs': args.epochs})}"
            initial_rec_checkpoint = (mf_checkpoints[(config["embedding_dim"], seed)]
                                      if args.warm_start_mf else None)
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                result = train_dr_bias_cdr(
                    dataset_name="coat", data_dict=data,
                    epochs=args.epochs, patience=7,
                    evaluate_test=False, seed=seed, checkpoint_tag=tag,
                    initial_rec_checkpoint=initial_rec_checkpoint, **config)
            score = float(result["best_val_ndcg"])
            scores.append(score)
            report["trials"].append({"config": config, "seed": seed,
                                     "initial_rec_checkpoint": initial_rec_checkpoint,
                                     "best_val_ndcg": score,
                                     "best_epoch": result["best_epoch"],
                                     "checkpoint_path": result["checkpoint_path"]})
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{config} seed={seed}: validation NDCG@5={score:.4f}", flush=True)
        report["summary"].append({"config": config,
                                  "mean_validation_ndcg": float(np.mean(scores)),
                                  "std_validation_ndcg": float(np.std(scores, ddof=1))
                                  if len(scores) > 1 else None})
    report["selected"] = max(report["summary"], key=lambda row: row["mean_validation_ndcg"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected on validation: {report['selected']}")
    print(f"Report: {path}")
