# -*- coding: utf-8 -*-
"""
Script huấn luyện Matrix Factorization Baseline trên Coat và KuaiRand
Project: TopTop (agri-behavioral-recsys)
"""

import os
import sys

# Đảm bảo nhận diện đúng package src từ PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from src.data_pipeline.coat_dataset import load_coat_processed
from src.data_pipeline.kuairand_dataset import load_kuairand_processed
from src.models.matrix_factorization import MatrixFactorization
from src.evaluation.evaluator import evaluate_unbiased_model
from src.evaluation.protocol import TOP_K
from src.utils.artifacts import (validate_training, make_metadata, checkpoint_path,
                                 save_checkpoint, evaluate_checkpoint)


def train_baseline_mf(dataset_name="coat", embedding_dim=64, epochs=35, lr=None,
                      batch_size=None, patience=7, device=None, alpha=0.0,
                      checkpoint_tag="manual", data_dict=None,
                      weight_decay=1e-4, evaluate_test=True, seed=42, checkpoint_dir=None,
                      optimizer_name="adam", loss_name="bce", decay_scope="embeddings",
                      scheduler_name="plateau", train_global_bias=True, adam_fused=False):
    validate_training(epochs=epochs, embedding_dim=embedding_dim, lr=lr,
                      batch_size=batch_size, patience=patience)
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if optimizer_name not in ("adam", "sgd"):
        raise ValueError("optimizer_name must be adam or sgd; classical ALS does not optimize BCE")
    if adam_fused and optimizer_name != "adam":
        raise ValueError("adam_fused requires Adam")
    if loss_name not in ("bce", "mse"):
        raise ValueError("loss_name must be bce or mse")
    if decay_scope not in ("embeddings", "all"):
        raise ValueError("decay_scope must be embeddings or all")
    if scheduler_name not in ("plateau", "none"):
        raise ValueError("scheduler_name must be plateau or none")
    set_seed(seed)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    device = device or get_device()

    if dataset_name.lower() == "coat":
        data = data_dict if data_dict is not None else load_coat_processed()
        n_users = data["n_users"]
        n_items = data["n_items"]
        train_u = data["train_u"]
        train_i = data["train_i"]
        train_y = data["train_y"]
        train_c = np.ones_like(train_y)
        val_candidates = data["val_candidates"]
        val_ground_truth = data["val_ground_truth"]
        test_candidates = data["test_candidates"]
        test_ground_truth = data["test_ground_truth"]
        n_samples = len(train_u)
        batch_size = batch_size or 512
        lr = lr if lr is not None else (0.005 if optimizer_name == "adam" else 0.1)
    elif dataset_name.lower() == "kuairand":
        data = data_dict if data_dict is not None else load_kuairand_processed(alpha=alpha)
        n_users = data["n_users"]
        n_items = data["n_items"]
        train_u = data["train_u"]
        train_i = data["train_i"]
        train_y = data["train_y"]
        train_c = data["train_c"]
        val_candidates = data["val_candidates"]
        val_ground_truth = data["val_ground_truth"]
        test_candidates = data["test_candidates"]
        test_ground_truth = data["test_ground_truth"]
        n_samples = len(train_u)
        batch_size = batch_size or 4096
        lr = lr if lr is not None else (0.001 if optimizer_name == "adam" else 0.1)
    else:
        raise ValueError(f"Tập dữ liệu không hợp lệ: '{dataset_name}'. Chỉ hỗ trợ 'coat' hoặc 'kuairand'.")

    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN MATRIX FACTORIZATION BASELINE: {dataset_name.upper()}")
    print(f" Thiết bị: {device}, Dim: {embedding_dim}, LR: {lr}, Batch: {batch_size}, Optimizer: {optimizer_name}, Loss: {loss_name}")
    print(f"=======================================================")

    model = MatrixFactorization(n_users, n_items, embedding_dim=embedding_dim, use_bias=True,
                                train_global_bias=train_global_bias).to(device)
    
    # Tách biệt L2 regularization: chỉ phạt ma trận nhúng, không phạt bias và global_bias
    weight_params = [model.user_embedding.weight, model.item_embedding.weight]
    bias_params = [model.user_bias.weight, model.item_bias.weight, model.global_bias]
    optimizer_class = optim.Adam if optimizer_name == "adam" else optim.SGD
    optimizer = optimizer_class([
        {"params": weight_params, "weight_decay": weight_decay},
        {"params": bias_params, "weight_decay": weight_decay if decay_scope == "all" else 0.0}
    ], lr=lr, **({"fused": True} if adam_fused else {}))

    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)
        if scheduler_name == "plateau" else None)
    loss_fn = (nn.BCEWithLogitsLoss(reduction="none") if loss_name == "bce"
               else nn.MSELoss(reduction="none"))

    if n_samples == 0:
        raise ValueError("Training data is empty")
    best_val_ndcg = -1.0
    best_epoch = 1
    patience_cnt = 0
    metadata = make_metadata("mf", dataset_name, data,
        dict(epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
             alpha=alpha, weight_decay=weight_decay, seed=seed, device=str(device),
             optimizer=optimizer_name, loss=loss_name, decay_scope=decay_scope,
             scheduler=scheduler_name, adam_fused=adam_fused,
             momentum=0.0 if optimizer_name == "sgd" else None),
        dict(num_users=n_users, num_items=n_items, embedding_dim=embedding_dim, use_bias=True,
             train_global_bias=train_global_bias))
    save_path = checkpoint_path(checkpoint_dir or os.path.join(CHECKPOINT_DIR, "baselines"),
                                f"{dataset_name}_mf_{checkpoint_tag}", metadata)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            b_idx = perm[start_idx:end_idx]
            u_t = torch.tensor(train_u[b_idx], dtype=torch.long, device=device)
            i_t = torch.tensor(train_i[b_idx], dtype=torch.long, device=device)
            y_t = torch.tensor(train_y[b_idx], dtype=torch.float32, device=device)
            c_t = torch.tensor(train_c[b_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            preds = model(u_t, i_t)
            loss = (loss_fn(preds, y_t) * c_t).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_idx)
            n_batches += len(b_idx)

        avg_train_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate_unbiased_model(model, val_candidates, val_ground_truth, k=TOP_K, device=device,
                                               include_zero_positive_users=data.get("evaluation_include_zero_positive_users", True))
        val_ndcg = val_metrics["NDCG@5"]
        if not np.isfinite(val_ndcg):
            raise FloatingPointError("Non-finite validation score")
        val_rec = val_metrics["Recall@5"]
        val_hr = val_metrics["HitRate@5"]

        if scheduler is not None:
            scheduler.step(val_ndcg)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val NDCG@5: {val_ndcg:.4f} | Recall@5: {val_rec:.4f} | HR@5: {val_hr:.4f}")

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            patience_cnt = 0
            save_checkpoint(model, save_path, metadata, epoch, val_metrics)
            best_metrics = val_metrics
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  -> Early stopping kích hoạt tại Epoch {epoch} (Best NDCG@5: {best_val_ndcg:.4f} tại Epoch {best_epoch})")
                break

    if not evaluate_test:
        return {"best_val_ndcg": best_val_ndcg, "best_epoch": best_epoch,
                "checkpoint_path": save_path, "validation": best_metrics, "metadata": metadata,
                "epochs_run": epoch, "epoch_cap_reached": epoch == epochs}

    return evaluate_checkpoint(save_path, data, device)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coat", choices=["coat", "kuairand"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--loss", choices=["bce", "mse"], default="bce")
    parser.add_argument("--decay-scope", choices=["embeddings", "all"], default="embeddings")
    parser.add_argument("--scheduler", choices=["plateau", "none"], default="plateau")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = train_baseline_mf(
        dataset_name=args.dataset, embedding_dim=args.dim, epochs=args.epochs, lr=args.lr,
        optimizer_name=args.optimizer, loss_name=args.loss, decay_scope=args.decay_scope,
        scheduler_name=args.scheduler, weight_decay=args.weight_decay,
        batch_size=args.batch_size, seed=args.seed)
    print({key: result[key] for key in ("NDCG@5", "Recall@5", "best_epoch", "checkpoint_path")})


if __name__ == "__main__":
    main()
