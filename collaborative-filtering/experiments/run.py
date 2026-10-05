"""Single entrypoint for tests, experiments, evaluation and cleanup.

Run from collaborative-filtering: python experiments/run.py COMMAND --help
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKFLOWS = {
    "benchmark": "run_all_benchmarks",
    "tune": "tune_coat_models",
    "tune-coat": "tune_coat_models",
    "tune-kr": "tune_kuairand_models",
    "repeat": "repeat_coat_seeds",
    "optimizers": "compare_mf_optimizers",
    "table3": "table3",
    "cdr-coat-tune": "tune_cdr_coat_propensity",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["test", "smoke", "mf", *WORKFLOWS, "evaluate", "protocol", "clean"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "mf":
        from src.baselines.train_mf import main as train_mf
        train_mf(args.arguments)
        return 0
    if args.command in WORKFLOWS:
        workflow = importlib.import_module(f"experiments.workflows.{WORKFLOWS[args.command]}")
        workflow.main(args.arguments)
        return 0
    p = argparse.ArgumentParser(prog=f"{parser.prog} {args.command}")
    if args.command == "test":
        p.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2)
        options = p.parse_args(args.arguments)
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_suite.py")
        result = unittest.TextTestRunner(verbosity=options.verbosity).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.command == "smoke":
        p.add_argument("--dataset", choices=["all", "coat", "kuairand"], default="all")
        p.add_argument("--model", choices=["all", "mf", "ividr", "cdr"], default="all")
        p.add_argument("--smoke-users", type=int, default=64)
        options = p.parse_args(args.arguments)
        if options.smoke_users < 2:
            p.error("--smoke-users must be at least 2")
        from experiments.workflows.run_all_benchmarks import run_all_benchmarks
        datasets = ["coat", "kuairand"] if options.dataset == "all" else [options.dataset]
        for dataset in datasets:
            run_all_benchmarks(quick_run=True, epochs_coat=1, epochs_kr=1,
                               dataset_filter=dataset, model_filter=options.model,
                               dim=16, batch_size=256,
                               smoke_users=options.smoke_users if dataset == "kuairand" else None)
        return 0
    if args.command == "protocol":
        p.parse_args(args.arguments)
        from src.evaluation.protocol import protocol_metadata
        print(json.dumps(protocol_metadata(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "clean":
        p.add_argument("--apply", action="store_true", help="Delete generated files; default is preview only")
        options = p.parse_args(args.arguments)
        command = ["powershell.exe", "-NoProfile", "-File", str(ROOT / "experiments" / "clean_generated.ps1")]
        if options.apply:
            command.append("-Apply")
        return subprocess.run(command, check=False).returncode
    if args.command == "evaluate":
        p.add_argument("--dataset", choices=["coat", "kuairand"], required=True)
        p.add_argument("--checkpoint", type=Path, required=True)
        options = p.parse_args(args.arguments)
        if not options.checkpoint.is_file():
            p.error("Checkpoint does not exist")
        from src.data_pipeline.coat_dataset import load_coat_processed
        from src.data_pipeline.kuairand_dataset import load_kuairand_processed
        from src.utils.artifacts import evaluate_checkpoint
        data = load_coat_processed() if options.dataset == "coat" else load_kuairand_processed()
        print(json.dumps(evaluate_checkpoint(options.checkpoint, data), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
