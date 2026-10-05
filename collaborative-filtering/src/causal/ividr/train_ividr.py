# -*- coding: utf-8 -*-
"""
Script huấn luyện mô hình Causal Debiasing: IViDR trên Coat và KuaiRand
Project: TopTop (agri-behavioral-recsys)
"""

import os
import sys

# Đảm bảo nhận diện đúng package src từ PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
from scipy.sparse import csr_matrix

from src.utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from src.data_pipeline.coat_dataset import load_coat_processed, CoatCausalDataset
from src.data_pipeline.kuairand_dataset import load_kuairand_processed, load_kuairand_features
from src.causal.ividr.ividr_model import IViDRModel
from src.evaluation.evaluator import evaluate_unbiased_model
from src.evaluation.protocol import TOP_K
from src.utils.artifacts import (validate_training, make_metadata, checkpoint_path,
                                 save_checkpoint, evaluate_checkpoint)


def make_exposure_batch(exposure_matrix, user_ids, num_items, sample_size, rng, device):
    """Build exposure reconstruction targets from biased training interactions only."""
    if sample_size >= num_items:
        labels = exposure_matrix[user_ids].toarray().astype(np.float32)
        return torch.as_tensor(labels, device=device), None
    item_ids = rng.integers(0, num_items, size=(len(user_ids), sample_size), dtype=np.int64)
    labels = exposure_matrix[user_ids[:, None], item_ids].toarray().astype(np.float32)
    return torch.as_tensor(labels, device=device), torch.as_tensor(item_ids, device=device)


def train_ividr(dataset_name="coat", embedding_dim=None, latent_dim=None, epochs=35, lr=None,
                batch_size=None, phi=1.0, beta_elbo=None, patience=None, device=None,
                alpha=0.0, checkpoint_tag="manual", data_dict=None,
                exposure_sample_size=128, evaluate_test=True, seed=42,
                weight_decay=1e-5, scheduler_name="none", adam_fused=False,
                projection_mode="item_pinv", elbo_reduction="sum"):
    validate_training(epochs=epochs, embedding_dim=embedding_dim, lr=lr,
                      batch_size=batch_size, patience=patience)
    if dataset_name.lower() not in ("coat", "kuairand"):
        raise ValueError("Unsupported dataset")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if scheduler_name not in ("none", "plateau"):
        raise ValueError("Invalid scheduler")
    set_seed(seed)
    validate_training(latent_dim=latent_dim, exposure_sample_size=exposure_sample_size)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    device = device or get_device()

    if dataset_name.lower() == "coat":
        embedding_dim = embedding_dim or 64
        latent_dim = latent_dim or 32
        lr = lr or 0.003
        batch_size = batch_size or 512
        beta_elbo = beta_elbo if beta_elbo is not None else 0.01
        patience = patience or 7
        kl_warmup_epochs = 3
    elif dataset_name.lower() == "kuairand":
        embedding_dim = embedding_dim or 128
        latent_dim = latent_dim or 32
        lr = lr or 0.002
        batch_size = batch_size or 4096
        beta_elbo = beta_elbo if beta_elbo is not None else 0.001
        patience = patience or 7
        kl_warmup_epochs = 5
    else:
        raise ValueError(f"Tập dữ liệu không hợp lệ: '{dataset_name}'. Chỉ hỗ trợ 'coat' hoặc 'kuairand'.")

    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN MÔ HÌNH CAUSAL DEBIASING: IViDR [{dataset_name.upper()}]")
    print(f" Thiết bị: {device} | Dim: {embedding_dim} | Latent C: {latent_dim} | LR: {lr} | Phi: {phi}")
    print(f" Beta ELBO: {beta_elbo} | KL Warmup: {kl_warmup_epochs} epochs | Patience: {patience}")
    print(f"=======================================================")

    if dataset_name.lower() == "coat":
        proc_data = data_dict if data_dict is not None else load_coat_processed()
        n_users = proc_data["n_users"]
        n_items = proc_data["n_items"]
        train_u = proc_data["train_u"]
        train_i = proc_data["train_i"]
        train_y = proc_data["train_y"]
        train_c = np.ones_like(train_y)
        val_candidates = proc_data["val_candidates"]
        val_ground_truth = proc_data["val_ground_truth"]
        test_candidates = proc_data["test_candidates"]
        test_ground_truth = proc_data["test_ground_truth"]

        dataset = CoatCausalDataset(train_matrix_override=proc_data["train_matrix_for_features"], data_dir=proc_data.get("data_dir"))
        user_feat_tensor = dataset.user_features.to(device)
        item_feat_tensor = dataset.item_features.to(device)
        proxy_w_tensor = dataset.proxy_w.to(device)

        iv_dim = user_feat_tensor.shape[1]
        proxy_dim = proxy_w_tensor.shape[1]
        n_samples = len(train_u)
    else:
        proc_data = data_dict if data_dict is not None else load_kuairand_processed(alpha=alpha)
        n_users = proc_data["n_users"]
        n_items = proc_data["n_items"]
        val_candidates = proc_data["val_candidates"]
        val_ground_truth = proc_data["val_ground_truth"]
        test_candidates = proc_data["test_candidates"]
        test_ground_truth = proc_data["test_ground_truth"]

        u_feat_np, proxy_w_np = load_kuairand_features(n_users, n_items, proc_data["user_mapping"], proc_data["video_mapping"], data_dir=proc_data.get("data_dir"))
        user_feat_tensor = torch.tensor(u_feat_np, dtype=torch.float32, device=device)
        proxy_w_tensor = torch.tensor(proxy_w_np, dtype=torch.float32, device=device)
        iv_dim = user_feat_tensor.shape[1]
        proxy_dim = proxy_w_tensor.shape[1]

        train_u = proc_data["train_u"]
        train_i = proc_data["train_i"]
        train_y = proc_data["train_y"]
        train_c = proc_data["train_c"]
        n_samples = len(train_u)

    if not np.isfinite(phi) or not np.isfinite(beta_elbo) or beta_elbo < 0:
        raise ValueError("phi must be finite; beta_elbo must be finite and nonnegative")
    model = IViDRModel(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        iv_dim=iv_dim,
        proxy_dim=proxy_dim,
        latent_dim=latent_dim,
        phi=phi,
        lam=1.0,
        rho=0.9,
        tau=0.9,
        projection_mode=projection_mode, elbo_reduction=elbo_reduction,
    ).to(device)

    exposure_matrix = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (train_u, train_i)),
        shape=(n_users, n_items), dtype=np.float32,
    )
    exposure_matrix.data[:] = 1.0
    exposure_rng = np.random.default_rng(seed)
    model.set_training_context(user_feat_tensor, exposure_matrix)
    reconstruction_items = n_items if dataset_name.lower() == "coat" else exposure_sample_size

    # Explicit all-parameter L2; the fixed zero global intercept has no gradient.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                           **({"fused": True} if adam_fused else {}))

    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)
                 if scheduler_name == "plateau" else None)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    import hashlib
    feature_hash = hashlib.sha256(user_feat_tensor.cpu().numpy().tobytes() + proxy_w_tensor.cpu().numpy().tobytes()).hexdigest()
    metadata = make_metadata("ividr", dataset_name, proc_data,
        dict(epochs=epochs, lr=lr, batch_size=batch_size, patience=patience, alpha=alpha,
             seed=seed, device=str(device), weight_decay=weight_decay, beta_elbo=beta_elbo,
             kl_warmup_epochs=kl_warmup_epochs, exposure_sample_size=reconstruction_items,
             feature_sha256=feature_hash, ridge=1e-4 if projection_mode == "global_ridge" else None,
             optimizer="Adam", adam_fused=adam_fused, gradient_clip=1.0,
             scheduler=scheduler_name, training_schedule="joint_local_reference",
             elbo_reduction=elbo_reduction, paper_reproduction_complete=False),
        dict(num_users=n_users, num_items=n_items, embedding_dim=embedding_dim,
             iv_dim=iv_dim, proxy_dim=proxy_dim, latent_dim=latent_dim,
             phi=phi, lam=1.0, rho=0.9, tau=0.9,
             projection_mode=projection_mode, elbo_reduction=elbo_reduction))
    save_path = checkpoint_path(os.path.join(CHECKPOINT_DIR, "ividr"),
                                f"{dataset_name}_ividr_{checkpoint_tag}", metadata)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    if n_samples == 0:
        raise ValueError("Training data is empty")
    best_val_ndcg = -1.0
    best_epoch = 1
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        # KL Annealing: tuyến tính tăng beta từ 0 lên beta_elbo trong kl_warmup_epochs đầu
        if epoch <= kl_warmup_epochs:
            current_beta = beta_elbo * (epoch / kl_warmup_epochs)
        else:
            current_beta = beta_elbo

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            b_idx = perm[start_idx:end_idx]
            u_t = torch.tensor(train_u[b_idx], dtype=torch.long, device=device)
            i_t = torch.tensor(train_i[b_idx], dtype=torch.long, device=device)
            y_t = torch.tensor(train_y[b_idx], dtype=torch.float32, device=device)
            c_t = torch.tensor(train_c[b_idx], dtype=torch.float32, device=device)

            u_feat_batch = user_feat_tensor[u_t]
            proxy_w_batch = proxy_w_tensor[u_t]
            exposure_labels, exposure_item_ids = make_exposure_batch(
                exposure_matrix, train_u[b_idx], n_items, reconstruction_items,
                exposure_rng, device,
            )

            optimizer.zero_grad()
            preds, elbo = model(
                u_t, i_t, u_feat_batch, proxy_w_batch,
                a_target=exposure_labels, exposure_item_ids=exposure_item_ids,
            )
            rec_loss = (bce_loss_fn(preds, y_t) * c_t).mean()
            loss = rec_loss + current_beta * elbo

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            # Gradient clipping để tránh loss spikes do KL explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(b_idx)
            n_batches += len(b_idx)

        avg_loss = total_loss / max(n_batches, 1)

        # Cập nhật cache người dùng và đánh giá Validation NDCG@5
        model.update_full_user_cache(user_feat_tensor, proxy_w_tensor, device=device)
        val_metrics = evaluate_unbiased_model(model, val_candidates, val_ground_truth, k=TOP_K, device=device,
                                               include_zero_positive_users=proc_data.get("evaluation_include_zero_positive_users", True))
        val_ndcg = val_metrics["NDCG@5"]
        if not np.isfinite(val_ndcg):
            raise FloatingPointError("Non-finite validation score")
        val_rec = val_metrics["Recall@5"]
        val_hr = val_metrics["HitRate@5"]

        if scheduler is not None:
            scheduler.step(val_ndcg)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | β={current_beta:.4f} | "
              f"Val NDCG@5: {val_ndcg:.4f} | Recall@5: {val_rec:.4f} | HR@5: {val_hr:.4f}", flush=True)

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
                "checkpoint_path": save_path, "metadata": metadata,
                "validation": best_metrics, "epochs_run": epoch, "epoch_cap_reached": epoch == epochs}

    return evaluate_checkpoint(save_path, proc_data, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coat", choices=["coat", "kuairand"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_ividr(dataset_name=args.dataset, embedding_dim=args.dim, latent_dim=args.latent, epochs=args.epochs, lr=args.lr)
