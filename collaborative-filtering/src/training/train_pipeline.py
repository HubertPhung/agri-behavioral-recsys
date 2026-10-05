"""Canonical MF entrypoint and validation-only hyperparameter search."""
import torch
from src.evaluation.table_formatter import format_grid_table


def train_single_run(
    data_dict: dict,
    embedding_dim: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    epochs: int = 25,
    patience: int = 7,
    checkpoint_dir: str = None,
    device: torch.device = None,
    evaluate_test: bool = True,
    seed: int = 42,
    optimizer_name: str = "adam",
    loss_name: str = "bce",
    decay_scope: str = "embeddings",
    scheduler_name: str = "plateau",
):
    """Compatibility entrypoint delegating to the canonical MF trainer."""
    from src.baselines.train_mf import train_baseline_mf
    from src.utils.artifacts import evaluate_checkpoint
    data = dict(data_dict)
    if "n_users" not in data:
        data["n_users"] = data["num_users"]
    if "n_items" not in data:
        data["n_items"] = data["num_items"]
    result = train_baseline_mf(
        dataset_name="kuairand", data_dict=data, embedding_dim=embedding_dim,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
        patience=patience, checkpoint_dir=checkpoint_dir, device=device,
        evaluate_test=False, seed=seed,
        optimizer_name=optimizer_name, loss_name=loss_name,
        decay_scope=decay_scope, scheduler_name=scheduler_name,
    )
    output = {"validation": result["validation"], "checkpoint": result["checkpoint_path"],
              "metadata": result["metadata"]}
    if evaluate_test:
        output["test"] = evaluate_checkpoint(result["checkpoint_path"], data, device)
    return output


def run_grid_search(
    data_dict: dict,
    embedding_dim: int = 128,
    lr_candidates: list = None,
    weight_decay_candidates: list = None,
    batch_size: int = 4096,
    epochs: int = 20,
    device: torch.device = None
):
    """
    Validation-only grid used by this local MF benchmark:
      - lr in [1e-3, 5e-4, 1e-4, 5e-5]
      - weight_decay in [1e-4, 1e-5, 1e-6]
    """
    from src.utils.artifacts import validate_training
    validate_training(embedding_dim=embedding_dim, batch_size=batch_size, epochs=epochs)
    if lr_candidates == [] or weight_decay_candidates == []:
        raise ValueError("Hyperparameter grids must not be empty")
    if lr_candidates is None:
        lr_candidates = [1e-3, 5e-4, 1e-4, 5e-5]
    if weight_decay_candidates is None:
        weight_decay_candidates = [1e-4, 1e-5, 1e-6]

    print("\n" + "=" * 75)
    print(" BẮT ĐẦU QUÉT SIÊU THAM SỐ GRID SEARCH (MATRIX FACTORIZATION)")
    print(f" Danh sách Learning Rates : {lr_candidates}")
    print(f" Danh sách Weight Decays  : {weight_decay_candidates}")
    print("=" * 75)

    all_trials = []
    best_overall_ndcg = -1.0
    best_config = None

    trial_idx = 0
    total_trials = len(lr_candidates) * len(weight_decay_candidates)

    for lr in lr_candidates:
        for wd in weight_decay_candidates:
            trial_idx += 1
            print(f"\n>>> [Trial {trial_idx}/{total_trials}] Thử nghiệm lr={lr}, weight_decay={wd} ...")
            metrics = train_single_run(
                data_dict=data_dict,
                embedding_dim=embedding_dim,
                lr=lr,
                weight_decay=wd,
                batch_size=batch_size,
                epochs=epochs,
                patience=5,
                device=device,
                evaluate_test=False,
            )

            trial_record = {
                "lr": lr,
                "weight_decay": wd,
                "embedding_dim": embedding_dim,
                "metrics": metrics["validation"],
                "checkpoint": metrics["checkpoint"],
                "metadata": metrics["metadata"],
            }
            all_trials.append(trial_record)

            if metrics["validation"].get("NDCG@5", 0.0) > best_overall_ndcg:
                best_overall_ndcg = metrics["validation"]["NDCG@5"]
                best_config = trial_record

    from src.evaluation.table_formatter import format_grid_table
    all_trials.sort(key=lambda x: x["metrics"].get("NDCG@5", 0.0), reverse=True)
    grid_headers = ["Hạng", "Learning Rate", "Weight Decay", "NDCG@5", "Recall@5", "Precision@5", "HitRate@5"]
    grid_rows = []
    for rank, t in enumerate(all_trials[:10], 1):
        m = t["metrics"]
        grid_rows.append([
            f"#{rank}",
            f"{t['lr']:.0e}",
            f"{t['weight_decay']:.0e}",
            f"{m.get('NDCG@5', 0.0):.4f}",
            f"{m.get('Recall@5', 0.0):.4f}",
            f"{m.get('Precision@5', 0.0):.4f}",
            f"{m.get('HitRate@5', 0.0):.4f}"
        ])

    print("\n" + format_grid_table(
        headers=grid_headers,
        rows=grid_rows,
        col_aligns=["center", "center", "center", "center", "center", "center", "center"],
        title="KẾT QUẢ TỔNG KẾT QUÉT SIÊU THAM SỐ (GRID SEARCH TOP PERFORMERS)",
        row_divider=True
    ))

    if best_config is not None:
        from src.utils.artifacts import evaluate_checkpoint
        best_config["held_out_test"] = evaluate_checkpoint(best_config["checkpoint"], data_dict, device)
    return best_config, all_trials
