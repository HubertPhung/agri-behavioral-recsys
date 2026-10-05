# -*- coding: utf-8 -*-
"""
Script huấn luyện mô hình Causal Debiasing: DR-BIAS + CDR trên Coat và KuaiRand
Tích hợp lọc Poisonous Imputation bằng Monte Carlo Dropout và đối sánh Unbiased Top-5
Biến thể địa phương tham khảo CDR CIKM 2023 (Song et al.)
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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.helpers import set_seed, get_device, CHECKPOINT_DIR
from src.data_pipeline.coat_dataset import load_coat_raw_features, load_coat_processed
from src.data_pipeline.kuairand_dataset import load_kuairand_processed
from src.causal.cdr.cdr_model import CDRFramework
from src.evaluation.evaluator import evaluate_unbiased_model
from src.evaluation.protocol import TOP_K
from src.utils.artifacts import (validate_training, make_metadata, checkpoint_path,
                                 save_checkpoint, evaluate_checkpoint,
                                 load_checkpoint, data_signature)


def train_dr_bias_cdr(dataset_name="coat", embedding_dim=None, un_thres=10.0,
                      epochs=None, lr=None, batch_size=None, patience=7, device=None,
                      alpha=0.0, checkpoint_tag="manual", data_dict=None,
                      evaluate_test=True, seed=42, max_inverse_propensity=None,
                      population_counterfactual_weight=False,
                      initial_rec_checkpoint=None):
    validate_training(epochs=epochs, embedding_dim=embedding_dim, lr=lr,
                      batch_size=batch_size, patience=patience)
    if dataset_name.lower() not in ("coat", "kuairand"):
        raise ValueError("Unsupported dataset")
    if max_inverse_propensity is not None:
        if not np.isfinite(max_inverse_propensity) or max_inverse_propensity < 1:
            raise ValueError("max_inverse_propensity must be finite and at least 1")
        if dataset_name.lower() != "coat":
            raise ValueError("max_inverse_propensity is only supported for Coat")
    if population_counterfactual_weight and dataset_name.lower() != "coat":
        raise ValueError("population_counterfactual_weight is only supported for Coat")
    if initial_rec_checkpoint is not None and dataset_name.lower() != "coat":
        raise ValueError("initial_rec_checkpoint is only supported for Coat")
    set_seed(seed)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    device = device or get_device()

    if dataset_name.lower() == "coat":
        embedding_dim = embedding_dim or 32
        lr = lr or 0.002
        epochs = epochs or 35
        batch_size = batch_size or 128
        weight_decay = 1e-3
        G = 4
    else:
        embedding_dim = embedding_dim or 128
        lr = lr or 0.001
        epochs = epochs or 25
        batch_size = batch_size or 8192
        weight_decay = 1e-4
        G = 1

    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN MÔ HÌNH CAUSAL DEBIASING: DR-BIAS + CDR [{dataset_name.upper()}]")
    print(f" Thiết bị: {device} | Dim: {embedding_dim} | Unc Threshold: {un_thres} | LR: {lr}")
    print(f"=======================================================")

    if dataset_name.lower() == "coat":
        raw = load_coat_raw_features(data_dict.get("data_dir") if data_dict else None)
        train_matrix = raw["train_matrix"]
        test_matrix = raw["test_matrix"]
        propensities = raw["propensities"]
        n_users, n_items = train_matrix.shape

        proc_data = data_dict if data_dict is not None else load_coat_processed()
        val_candidates = proc_data["val_candidates"]
        val_ground_truth = proc_data["val_ground_truth"]
        test_candidates = proc_data["test_candidates"]
        test_ground_truth = proc_data["test_ground_truth"]

        train_u, train_i = proc_data["train_u"], proc_data["train_i"]
        train_r = proc_data["train_y"]
        inv_props = 1.0 / np.clip(propensities[train_u, train_i], 1e-4, 1.0)
        clipped_propensity_fraction = (float(np.mean(inv_props > max_inverse_propensity))
                                       if max_inverse_propensity is not None else 0.0)
        if max_inverse_propensity is not None:
            inv_props = np.minimum(inv_props, max_inverse_propensity)
        n_samples = len(train_u)

        # Counterfactuals pool
        all_pairs = np.argwhere(train_matrix == 0)
        n_all_pairs = len(all_pairs)
        counterfactual_scale = (n_all_pairs / (G * n_samples)
                                if population_counterfactual_weight else 1.0)
    else:
        proc_data = data_dict if data_dict is not None else load_kuairand_processed(alpha=alpha)
        n_users = proc_data["n_users"]
        n_items = proc_data["n_items"]
        val_candidates = proc_data["val_candidates"]
        val_ground_truth = proc_data["val_ground_truth"]
        test_candidates = proc_data["test_candidates"]
        test_ground_truth = proc_data["test_ground_truth"]

        train_u = proc_data["train_u"]
        train_i = proc_data["train_i"]
        train_r = proc_data["train_y"]

        # Exposure frequency is only an item-level proxy, not a logged propensity.
        item_pop = proc_data["train_item_exposure_counts"].astype(np.float32)
        item_prop = np.clip(item_pop / (item_pop.max() + 1e-6), 0.05, 1.0)
        inv_props = (1.0 / item_prop[train_i]).astype(np.float32)
        n_samples = len(train_u)
        n_all_pairs = n_samples
        clipped_propensity_fraction = 0.0
        counterfactual_scale = 1.0
        observed_users = torch.tensor(np.unique(train_u), dtype=torch.long, device=device)
        observed_items = torch.tensor(np.unique(train_i), dtype=torch.long, device=device)

    if not np.isfinite(un_thres) or un_thres < 0:
        raise ValueError("un_thres must be finite and nonnegative")
    model = CDRFramework(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        un_thres=un_thres,
        dropout_rate=0.5
    ).to(device)

    if initial_rec_checkpoint is not None:
        initial_state = load_checkpoint(initial_rec_checkpoint, device)
        initial_meta = initial_state["metadata"]
        if (initial_meta.get("model") != "mf" or
                initial_meta.get("dataset") != "coat" or
                initial_meta.get("data") != data_signature(proc_data) or
                initial_meta.get("model_kwargs", {}).get("embedding_dim") != embedding_dim or
                initial_meta.get("config", {}).get("seed") != seed):
            raise ValueError("MF warm-start checkpoint must match Coat data, dimension and seed")
        model.rec_model.load_state_dict(initial_state["model_state_dict"])

    # Tách biệt L2 regularization: không phạt bias và global_bias
    rec_w = [model.rec_model.user_embedding.weight, model.rec_model.item_embedding.weight]
    rec_b = [model.rec_model.user_bias.weight, model.rec_model.item_bias.weight, model.rec_model.global_bias]
    opt_rec = optim.Adam([
        {"params": rec_w, "weight_decay": weight_decay},
        {"params": rec_b, "weight_decay": 0.0}
    ], lr=lr)
    opt_imp = optim.Adam(model.imp_model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_rec = optim.lr_scheduler.ReduceLROnPlateau(opt_rec, mode="max", factor=0.5, patience=3, min_lr=1e-5)
    sched_imp = optim.lr_scheduler.ReduceLROnPlateau(opt_imp, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    import hashlib
    metadata = make_metadata("cdr", dataset_name, proc_data,
        dict(epochs=epochs, lr=lr, batch_size=batch_size, patience=patience, alpha=alpha,
             seed=seed, device=str(device), weight_decay=weight_decay, G=G,
             mc_samples=5 if dataset_name.lower() == "kuairand" else 10,
             propensity_sha256=hashlib.sha256(inv_props.tobytes()).hexdigest(),
             max_inverse_propensity=max_inverse_propensity,
             clipped_propensity_fraction=clipped_propensity_fraction,
             population_counterfactual_weight=population_counterfactual_weight,
             counterfactual_scale=counterfactual_scale,
             initial_rec_checkpoint=str(initial_rec_checkpoint) if initial_rec_checkpoint else None,
             propensity="provided_coat" if dataset_name.lower() == "coat" else "item_frequency_clip_0.05_1",
             counterfactuals="unobserved_pairs" if dataset_name.lower() == "coat" else "uniform_catalog_pairs",
             optimizer="Adam", scheduler="plateau_factor0.5_patience3_min1e-5"),
        dict(num_users=n_users, num_items=n_items, embedding_dim=embedding_dim,
             un_thres=un_thres, dropout_rate=0.5))
    save_path = checkpoint_path(os.path.join(CHECKPOINT_DIR, "cdr"),
                                f"{dataset_name}_cdr_{checkpoint_tag}", metadata)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if n_samples == 0:
        raise ValueError("Training data is empty")
    best_val_ndcg = -1.0
    best_epoch = 1
    patience_cnt = 0
    total_batches = math.ceil(n_samples / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        perm_obs = np.random.permutation(n_samples)
        if dataset_name.lower() == "coat":
            perm_all = np.random.permutation(n_all_pairs)

        total_rec_loss = 0.0
        total_imp_loss = 0.0
        filtered_ratios = []
        n_b = 0

        for b_idx in range(total_batches):
            sel_obs = perm_obs[b_idx * batch_size : (b_idx + 1) * batch_size]
            current_size = len(sel_obs)
            u_o = torch.tensor(train_u[sel_obs], dtype=torch.long, device=device)
            i_o = torch.tensor(train_i[sel_obs], dtype=torch.long, device=device)
            r_o = torch.tensor(train_r[sel_obs], dtype=torch.float32, device=device)
            p_o = torch.tensor(inv_props[sel_obs], dtype=torch.float32, device=device)

            if dataset_name.lower() == "coat":
                sel_unobs = perm_all[G * b_idx * batch_size : G * b_idx * batch_size + G * current_size]
                sampled_pairs = all_pairs[sel_unobs]
                u_un = torch.tensor(sampled_pairs[:, 0], dtype=torch.long, device=device)
                i_un = torch.tensor(sampled_pairs[:, 1], dtype=torch.long, device=device)
            else:
                u_un = observed_users[torch.randint(0, len(observed_users), (current_size * G,), device=device)]
                i_un = observed_items[torch.randint(0, len(observed_items), (current_size * G,), device=device)]

            # Cập nhật đồng thời Recommendation Model (CDR Loss) và Imputation Model (DR-BIAS Loss)
            opt_rec.zero_grad()
            opt_imp.zero_grad()
            loss_rec, loss_imp, f_ratio = model.compute_cdr_losses(
                u_obs=u_o, i_obs=i_o, r_obs=r_o, inv_p_obs=p_o,
                u_unobs=u_un, i_unobs=i_un, mc_samples=(5 if dataset_name.lower() == "kuairand" else 10),
                counterfactual_scale=counterfactual_scale,
            )
            if not (torch.isfinite(loss_rec) and torch.isfinite(loss_imp)):
                raise FloatingPointError("Non-finite CDR training loss")
            loss_rec.backward()
            loss_imp.backward()
            opt_rec.step()
            opt_imp.step()


            total_rec_loss += loss_rec.item()
            total_imp_loss += loss_imp.item()
            filtered_ratios.append(f_ratio)
            n_b += 1

        avg_rec_loss = total_rec_loss / max(n_b, 1)
        avg_imp_loss = total_imp_loss / max(n_b, 1)
        avg_filtered = np.mean(filtered_ratios) * 100.0 if filtered_ratios else 0.0

        # Đánh giá Unbiased Validation NDCG@5
        val_metrics = evaluate_unbiased_model(model, val_candidates, val_ground_truth, k=TOP_K, device=device,
                                               include_zero_positive_users=proc_data.get("evaluation_include_zero_positive_users", True))
        val_ndcg = val_metrics["NDCG@5"]
        if not np.isfinite(val_ndcg):
            raise FloatingPointError("Non-finite validation score")
        val_rec = val_metrics["Recall@5"]
        val_hr = val_metrics["HitRate@5"]

        sched_rec.step(val_ndcg)
        sched_imp.step(val_ndcg)

        print(f"Epoch {epoch:02d}/{epochs:02d} | CDR Rec Loss: {avg_rec_loss:.4f} | Imp Loss: {avg_imp_loss:.4f} | "
              f"Poisonous Filtered: {avg_filtered:.1f}% | Val NDCG@5: {val_ndcg:.4f} | Recall@5: {val_rec:.4f} | HR@5: {val_hr:.4f}",
              flush=True)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            patience_cnt = 0
            save_checkpoint(model, save_path, metadata, epoch, val_metrics)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  -> Early stopping kích hoạt tại Epoch {epoch} (Best NDCG@5: {best_val_ndcg:.4f} tại Epoch {best_epoch})")
                break

    if not evaluate_test:
        return {"best_val_ndcg": best_val_ndcg, "best_epoch": best_epoch,
                "checkpoint_path": save_path, "metadata": metadata}

    return evaluate_checkpoint(save_path, proc_data, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coat", choices=["coat", "kuairand"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--un_thres", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_dr_bias_cdr(dataset_name=args.dataset, embedding_dim=args.dim, un_thres=args.un_thres, epochs=args.epochs, lr=args.lr)
