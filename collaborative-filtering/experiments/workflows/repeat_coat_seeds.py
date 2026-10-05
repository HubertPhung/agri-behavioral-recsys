"""Repeat validation-selected Coat configurations over a fixed seed list."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.train_mf import train_baseline_mf
from src.causal.ividr.train_ividr import train_ividr
from src.causal.cdr.train_dr_bias_cdr import train_dr_bias_cdr
from src.data_pipeline.coat_dataset import load_coat_processed
from src.utils.artifacts import data_signature, digest, evaluate_checkpoint, source_signature
from src.utils.helpers import RESULTS_DIR
from src.evaluation.protocol import DEFAULT_SEEDS, protocol_metadata, summarize_seeds


def summarize(runs):
    summary = {}
    for model in sorted({r["model"] for r in runs}):
        rows = [r for r in runs if r["model"] == model]
        summary[model] = {"seeds": [r["seed"] for r in rows],
                          **summarize_seeds([r["test"] for r in rows])}
    return summary


def write_report(payload, path):
    payload["summary"] = summarize(payload["runs"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Coat: cấu hình cố định, lặp nhiều seed", "",
             "Chọn siêu tham số bằng validation ở seed 42; giữ nguyên cho mọi seed sau đó. "
             "Mỗi seed chọn epoch bằng validation. Không chọn seed theo test.", "",
             f"Seeds đã định trước: {payload['identity']['seeds']}. "
             f"Trạng thái: {payload['status']}.", "",
             "| Mô hình | Số seed hoàn tất | NDCG@5 mean ± std | Recall@5 mean ± std |",
             "| :-- | --: | --: | --: |"]
    for name, summary in payload["summary"].items():
        ndcg, recall = summary["NDCG@5"], summary["Recall@5"]
        ndcg_std = f"{ndcg['std']:.4f}" if ndcg['std'] is not None else "N/A"
        recall_std = f"{recall['std']:.4f}" if recall['std'] is not None else "N/A"
        lines.append(f"| {name} | {len(summary['seeds'])} | {ndcg['mean']:.4f} ± {ndcg_std} | "
                     f"{recall['mean']:.4f} ± {recall_std} |")
    lines += ["", "Std là độ lệch chuẩn mẫu giữa seed trên cùng split; không phải khoảng tin cậy "
              "cho dữ liệu mới. Seed 42 đã tham gia chọn cấu hình. Đây là kiểm tra độ ổn định, "
              "chưa phải nested cross-validation hay tái lập nguyên bản bài báo.", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-report", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    tuning = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    if set(tuning["selected"]) != {"mf", "ividr", "cdr"}:
        parser.error("Provide a completed tuning report for all three models")
    data = load_coat_processed()
    source = source_signature()
    if tuning["identity"]["code_sha256"] != source:
        parser.error("Source changed since tuning; use a report matching the current code")
    if digest(tuning["identity"]["data"]) != digest(data_signature(data)):
        parser.error("Data changed since tuning")
    identity = {"version": 4, "seeds": args.seeds, "epochs": tuning["epochs"],
                "configs": {name: s["config"] for name, s in tuning["selected"].items()},
                "data": data_signature(data), "code_sha256": source,
                "tuning_identity": digest(tuning["identity"])}
    path = Path(RESULTS_DIR) / f"coat_multiseed_{digest(identity)}_v4.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if digest(payload["identity"]) != digest(identity):
            raise ValueError("Existing report identity mismatch")
    else:
        payload = {"identity": identity, "tuning_report": str(args.tuning_report.resolve()),
                   "evaluation_protocol": protocol_metadata(data),
                   "status": "running", "runs": []}
    trainers = {"mf": train_baseline_mf, "ividr": train_ividr, "cdr": train_dr_bias_cdr}
    done = {(r["model"], r["seed"]) for r in payload["runs"]}
    for name, config in identity["configs"].items():
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            if seed == 42:
                # Reuse the already evaluated selected checkpoint, including its test result.
                test = tuning["selected"][name]["test"]
            else:
                with path.with_suffix(".log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    result = trainers[name](dataset_name="coat", epochs=tuning["epochs"],
                                            seed=seed, evaluate_test=False, data_dict=data,
                                            checkpoint_tag=f"repeat_seed{seed}", **config)
                    test = evaluate_checkpoint(result["checkpoint_path"], data)
            payload["runs"].append({"model": name, "seed": seed, "test": test})
            write_report(payload, path)
            print(f"{name} seed={seed}: NDCG@5={test['NDCG@5']:.4f}, Recall@5={test['Recall@5']:.4f}", flush=True)
    payload["status"] = "complete"
    write_report(payload, path)
    print(f"Report: {path}")


