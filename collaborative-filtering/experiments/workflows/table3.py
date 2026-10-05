"""Full-data MF and local IViDR reference with Table 3's published Adam grid.

Local assumptions are recorded; completing this run is not an exact reproduction.
"""
import argparse
import contextlib
import json
import hashlib
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from src.baselines.train_mf import train_baseline_mf
from src.causal.ividr.train_ividr import train_ividr
from src.data_pipeline.coat_dataset import load_coat_processed
from src.data_pipeline.kuairand_dataset import load_kuairand_processed
from src.evaluation.protocol import protocol_metadata, summarize_seeds
from src.utils.artifacts import data_signature, digest, evaluate_checkpoint, source_signature, replace_with_retry
from src.utils.helpers import RESULTS_DIR
from src.utils.experiment_lock import experiment_lock

LEARNING_RATES = (1e-3, 5e-4, 1e-4, 5e-5, 1e-5)
WEIGHT_DECAYS = (1e-5, 1e-6)
SEEDS = tuple(range(42, 52))
PAPER = "https://arxiv.org/html/2410.12451v1#S5.SS1"


def feature_signature(data, dataset):
    root = Path(data['data_dir'])
    path = root / ('user_item_features/user_features.ascii' if dataset == 'coat' else 'user_features_pure.csv')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_job(data, dataset, common, threads, job):
    """Isolated process/seed and log for a single training run."""
    torch.set_num_threads(threads)
    started = time.perf_counter()
    settings = dict(common)
    model_name = settings.pop("model_name", "mf")
    expected_source = settings.pop("expected_source", None)
    expected_features = settings.pop("expected_features", None)
    if expected_source is not None and source_signature() != expected_source:
        raise RuntimeError("Source changed during experiment; refusing to mix versions")
    if expected_features is not None and feature_signature(data, dataset) != expected_features:
        raise RuntimeError("User features changed during experiment")
    trainer = train_baseline_mf if model_name == "mf" else train_ividr
    with Path(job["log"]).open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log):
        result = trainer(dataset, data_dict=data, **settings, **job["config"],
                         seed=job["seed"], evaluate_test=False, checkpoint_tag=f"table3_{model_name}")
    if expected_source is not None and source_signature() != expected_source:
        raise RuntimeError("Source changed during training; result is not accepted")
    if expected_features is not None and feature_signature(data, dataset) != expected_features:
        raise RuntimeError("User features changed during training; result is not accepted")
    return dict(config=job["config"], seed=job["seed"], seconds=time.perf_counter()-started, **result)


def train_jobs(data, dataset, common, threads, jobs, workers):
    if workers == 1:
        for job in jobs:
            yield train_job(data, dataset, common, threads, job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(train_job, data, dataset, common, threads, job) for job in jobs]
            errors = []
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except Exception as error:
                    errors.append(error)
                    for pending in futures:
                        pending.cancel()
                else:
                    yield result
            if errors:
                raise RuntimeError(f"Training job failed: {errors[0]}") from errors[0]


def save_report(payload, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)
    lines = [f"# Table 3 reference settings — {payload['identity']['model']} / {payload['identity']['dataset']}", "",
             f"Status: **{payload['status']}**. Trials: {len(payload['trials'])}/10; evaluation seeds: {len(payload['runs'])}/10.", "",
             "Published optimization settings are used; full IViDR Table 3 reproduction is NOT complete.", "",
             "IViDR reference uses item-specific pseudoinverse; joint training/proxies remain local assumptions.", "",
             f"Paper: {PAPER}", "", "## Explicit implementation choices", ""]
    lines.extend(f"- {s}" for s in payload["implementation_choices"])
    if payload.get("selected"):
        lines += ["", f"Validation-selected config: `{payload['selected']['config']}`."]
    if payload["runs"]:
        lines += ["", "| Metric | Mean | Sample std | Seeds |", "| :-- | --: | --: | --: |"]
        for metric, result in summarize_seeds([r["test"] for r in payload["runs"]]).items():
            std = f"{result['std']:.6f}" if result['std'] is not None else "N/A"
            lines.append(f"| {metric} | {result['mean']:.6f} | {std} | {result['n_seeds']} |")
        lines += ["", "Seed 42 participates in tuning. No test-based configuration/seed selection.",
                  "A t-test against IViDR requires real IViDR seed results; paper mean/std are not substituted."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data, dataset, *, epochs=100, patience=20, threads=2, workers=1, results_dir=None, model_name="mf"):
    if dataset not in ("coat", "kuairand") or model_name not in ("mf", "ividr"):
        raise ValueError("Invalid dataset/model")
    directory = Path(results_dir or RESULTS_DIR)
    with experiment_lock(directory / f"table3_{model_name}_{dataset}.lock"):
        state = {}
        try:
            return _run(data, dataset, epochs=epochs, patience=patience, threads=threads,
                        workers=workers, results_dir=directory, model_name=model_name, state=state)
        except Exception as error:
            if state:
                state['payload']['status'] = 'failed'
                state['payload']['error'] = repr(error)
                try:
                    save_report(state['payload'], state['path'])
                except OSError:
                    pass  # preserve the original training error if disk writing also fails
            raise


def _run(data, dataset, *, epochs, patience, threads, workers, results_dir, model_name, state):
    if dataset not in ("coat", "kuairand") or min(epochs, patience, threads, workers) < 1:
        raise ValueError("Invalid dataset or training budget")
    if data.get("split_info", {}).get("smoke_only"):
        raise ValueError("Table 3 run requires full data, not a smoke subset")
    if model_name not in ("mf", "ividr"):
        raise ValueError("model_name must be mf or ividr")
    torch.set_num_threads(threads)
    common = dict(embedding_dim=32 if dataset == "coat" else 128,
                  epochs=epochs, patience=patience, batch_size=512 if dataset == "coat" else 4096,
                  optimizer_name="adam", loss_name="bce", decay_scope="all",
                  scheduler_name="none", train_global_bias=False, adam_fused=True)
    if model_name == "ividr":
        for key in ("optimizer_name", "loss_name", "decay_scope", "train_global_bias"):
            common.pop(key)
        common.update(latent_dim=16 if dataset == "coat" else 32, phi=1.0,
                      projection_mode="item_pinv", elbo_reduction="sum")
    identity = dict(dataset=dataset, model=model_name, common=common,
                    grid=dict(lr=LEARNING_RATES, weight_decay=WEIGHT_DECAYS), seeds=SEEDS,
                    data=data_signature(data), code_sha256=source_signature(),
                    torch=str(torch.__version__), threads=threads, workers=workers)
    if model_name == 'ividr':
        identity['user_features_sha256'] = feature_signature(data, dataset)
    directory = Path(results_dir or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"table3_{model_name}_{dataset}_{digest(identity)}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if digest(payload["identity"]) != digest(identity):
            raise ValueError("Existing run identity mismatch")
        referenced = [t['checkpoint_path'] for t in payload['trials']]
        referenced += [r['test']['checkpoint_path'] for r in payload['runs']]
        if any(not Path(name).is_file() for name in referenced):
            raise FileNotFoundError("Report references missing checkpoints; restore artifacts or clean this run before restarting")
    else:
        payload = dict(identity=identity, status="tuning", trials=[], runs=[], selected=None,
                       evaluation_protocol=protocol_metadata(data), paper_reproduction_complete=False,
                       ividr_status="local_reference_not_author_implementation",
                       implementation_choices=[
                           "Only Adam, the LR/decay grid and 10 trials are explicitly fixed by section 5.1.",
                           f"Epoch cap {epochs}, patience {patience}, batch {common['batch_size']} are local choices; paper does not specify them.",
                           "MF dimension matches the dataset's published IViDR dimension; MF-specific dimension is not specified.",
                           "MF uses BCE, all-parameter L2, normal embedding initialization and zero user/item biases; no learned global intercept (Eq. 23).",
                           "Fixed seeds 42–51 and validation-only tuning on seed 42 are local reproducibility choices.",
                           "iDCF-based user split/candidate protocol, binary duplicate aggregation and deterministic ties remain local conventions."])
        if model_name == "ividr":
            payload["implementation_choices"] += [
                "IViDR: per-item Moore-Penrose projection; mean-pooled gate context, train-only user histories.",
                "IViDR: joint BCE + annealed ELBO, LayerNorm, latent dimension 16/32, proxies and sampled exposure are local assumptions; Appendix B staged training is not implemented.",
                "Gaussian KL and Bernoulli reconstruction are sums; uniform sampled items use an unbiased full-sum estimator."]
    training_common = dict(common, model_name=model_name, expected_source=identity['code_sha256'])
    if model_name == 'ividr':
        training_common['expected_features'] = identity['user_features_sha256']
    state.update(payload=payload, path=path)
    payload['status'] = 'tuning'
    payload.pop('error', None)
    save_report(payload, path)
    print(f"Report: {path}", flush=True)
    done = {(t["config"]["lr"], t["config"]["weight_decay"]) for t in payload["trials"]}
    jobs = [dict(config=dict(lr=lr, weight_decay=wd), seed=SEEDS[0],
                 log=str(path.with_name(path.stem + f"_lr{lr}_wd{wd}.log")))
            for lr in LEARNING_RATES for wd in WEIGHT_DECAYS if (lr, wd) not in done]
    for result in train_jobs(data, dataset, training_common, threads, jobs, workers):
        payload["trials"].append(result)
        save_report(payload, path)
        print(f"{dataset} {model_name} trial {len(payload['trials'])}/10: {result['config']}, val={result['best_val_ndcg']:.6f}, epochs={result['epochs_run']}", flush=True)
    if payload["selected"] is None:
        payload["trials"].sort(key=lambda t: (LEARNING_RATES.index(t['config']['lr']),
                                             WEIGHT_DECAYS.index(t['config']['weight_decay'])))
        payload["selected"] = max(payload["trials"], key=lambda t: t["best_val_ndcg"])
    selected = payload["selected"]
    payload["status"] = "evaluating_frozen_config"
    save_report(payload, path)
    done_seeds = {r["seed"] for r in payload["runs"]}
    def evaluate_result(seed, result):
        if source_signature() != identity['code_sha256']:
            raise RuntimeError("Source changed before evaluation")
        test = evaluate_checkpoint(result["checkpoint_path"], data)
        payload["runs"].append(dict(seed=seed, test=test, epochs_run=result["epochs_run"],
                                    epoch_cap_reached=result["epoch_cap_reached"]))
        payload["runs"].sort(key=lambda r: r['seed'])
        save_report(payload, path)
        print(f"{dataset} {model_name} seed {seed}: test NDCG={test['NDCG@5']:.6f}, Recall={test['Recall@5']:.6f}", flush=True)
    if SEEDS[0] not in done_seeds:
        evaluate_result(SEEDS[0], selected)
    jobs = [dict(config=selected['config'], seed=seed,
                 log=str(path.with_name(path.stem + f"_seed{seed}.log")))
            for seed in SEEDS[1:] if seed not in done_seeds]
    for result in train_jobs(data, dataset, training_common, threads, jobs, workers):
        evaluate_result(result['seed'], result)
    payload["status"] = f"{model_name}_reference_complete"
    payload["summary"] = summarize_seeds([r["test"] for r in payload["runs"]])
    save_report(payload, path)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coat", "kuairand", "all"], default="all")
    parser.add_argument("--model", choices=["mf", "ividr", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.threads, args.workers) < 1:
        parser.error("epochs, patience and threads must be positive")
    for dataset in (["coat", "kuairand"] if args.dataset == "all" else [args.dataset]):
        data = load_coat_processed() if dataset == "coat" else load_kuairand_processed()
        for model_name in (["mf", "ividr"] if args.model == "all" else [args.model]):
            run(data, dataset, epochs=args.epochs, patience=args.patience, threads=args.threads,
                workers=args.workers, model_name=model_name)
