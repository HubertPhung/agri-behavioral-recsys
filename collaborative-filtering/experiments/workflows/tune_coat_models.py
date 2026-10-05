# -*- coding: utf-8 -*-
"""
Workflow Tìm Kiếm & Tối Ưu Hóa Siêu Tham Số (Hyperparameter Tuning) Cho Coat Dataset.

Tuân thủ nghiêm ngặt nguyên tắc thực nghiệm khoa học:
1. Toàn bộ các trial trong Grid Search chỉ được lựa chọn dựa trên điểm số Validation NDCG@5 (30% Random split).
2. Checkpoint tối ưu nhất của mỗi mô hình mới được đánh giá DUY NHẤT 1 LẦN trên tập Held-Out Test (70% Random split).
3. Hỗ trợ Warm-start MF cho CDR, hiển thị tiến độ thời gian thực (TeeLogger) và xuất bảng đối sánh khoa học.

TopTop Project (agri-behavioral-recsys)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.train_mf import train_baseline_mf
from src.causal.ividr.train_ividr import train_ividr
from src.causal.cdr.train_dr_bias_cdr import train_dr_bias_cdr
from src.utils.helpers import RESULTS_DIR
from src.utils.artifacts import evaluate_checkpoint, data_signature, source_signature, digest
from src.data_pipeline.coat_dataset import load_coat_processed
from src.evaluation.protocol import protocol_metadata
from src.evaluation.table_formatter import format_grid_table


class TeeLogger:
    """Ghi đồng thời log ra file và hiển thị thời gian thực lên màn hình console."""
    def __init__(self, file_path: Path):
        self.file = open(file_path, "a", encoding="utf-8")
        self.terminal = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.terminal.write(message)
        self.terminal.flush()

    def flush(self):
        self.file.flush()
        self.terminal.flush()

    def close(self):
        self.file.close()


# =============================================================================
# LƯỚI TÌM KIẾM SIÊU THAM SỐ (SEARCH GRIDS) ĐÃ TINH CHỈNH CHO COAT DATASET
# =============================================================================
GRIDS = {
    "mf": [
        {"embedding_dim": 16, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512},
        {"embedding_dim": 32, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512},
        {"embedding_dim": 32, "lr": 5e-4, "weight_decay": 1e-5, "batch_size": 512},
        {"embedding_dim": 64, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512},
        {"embedding_dim": 64, "lr": 2e-3, "weight_decay": 1e-5, "batch_size": 512},
        {"embedding_dim": 128, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 512},
    ],
    "ividr": [
        {"embedding_dim": 32, "latent_dim": 16, "lr": 1e-3, "phi": 1.0, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 32, "latent_dim": 16, "lr": 3e-3, "phi": 0.5, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 32, "latent_dim": 16, "lr": 3e-3, "phi": 1.0, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 64, "latent_dim": 32, "lr": 2e-3, "phi": 1.0, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 64, "latent_dim": 32, "lr": 3e-3, "phi": 0.5, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 64, "latent_dim": 32, "lr": 3e-3, "phi": 1.0, "beta_elbo": 0.01, "batch_size": 512},
        {"embedding_dim": 64, "latent_dim": 32, "lr": 3e-3, "phi": 2.0, "beta_elbo": 0.01, "batch_size": 512},
    ],
    "cdr": [
        {"embedding_dim": 32, "lr": 0.001, "un_thres": 0.05, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 32, "lr": 0.002, "un_thres": 0.01, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 32, "lr": 0.002, "un_thres": 0.05, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 32, "lr": 0.002, "un_thres": 0.10, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 32, "lr": 0.005, "un_thres": 0.05, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 64, "lr": 0.001, "un_thres": 0.05, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 64, "lr": 0.002, "un_thres": 0.05, "max_inverse_propensity": 20.0, "batch_size": 128},
        {"embedding_dim": 64, "lr": 0.003, "un_thres": 0.05, "max_inverse_propensity": 10.0, "batch_size": 128},
    ],
}


def train_model(name: str, config: dict, epochs: int, evaluate_test: bool, tag: str,
                data_dict: dict = None, patience: int = 7):
    common = {
        "dataset_name": "coat",
        "epochs": epochs,
        "patience": patience,
        "evaluate_test": evaluate_test,
        "checkpoint_tag": tag,
    }
    if data_dict is not None:
        common["data_dict"] = data_dict
    if name == "mf":
        return train_baseline_mf(**common, **config)
    if name == "ividr":
        return train_ividr(**common, **config)
    if name == "cdr":
        cfg = dict(config)
        dim = cfg.get("embedding_dim", 32)
        initial_ckpt = cfg.get("initial_rec_checkpoint")
        if initial_ckpt is None:
            mf_tag = f"coat_mf_warm_{dim}_e{epochs}"
            mf_res = train_baseline_mf(
                dataset_name="coat",
                embedding_dim=dim,
                epochs=epochs,
                lr=1e-3,
                weight_decay=1e-5,
                batch_size=512,
                patience=patience,
                evaluate_test=False,
                checkpoint_tag=mf_tag,
                **({"data_dict": data_dict} if data_dict is not None else {})
            )
            initial_ckpt = mf_res["checkpoint_path"]
            cfg["initial_rec_checkpoint"] = initial_ckpt
        return train_dr_bias_cdr(**common, **cfg)
    raise ValueError(f"Mô hình không được hỗ trợ: {name}")


def write_report(payload: dict, json_path: Path):
    """Ghi file kết quả định dạng JSON và Markdown kèm bảng đối sánh khoa học."""
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    lines = [
        "# Coat Dataset: Báo Cáo Tinh Chỉnh Siêu Tham Số (Validation Tuning)",
        "",
        f"- **Số Epoch Tối Đa**: {payload['epochs']} epochs (kèm Early Stopping)",
        f"- **Giao thức thực nghiệm**: Chọn cấu hình tốt nhất hoàn toàn bằng tập Validation (30% Random log).",
        f"- **Đánh giá khách quan**: Chỉ chấm điểm Held-Out Test DUY NHẤT 1 LẦN cho mô hình được chọn.",
        "",
        "| Mô hình | Cấu hình tối ưu | Val NDCG@5 | Test NDCG@5 | Test Recall@5 | Test Precision@5 | Test HitRate@5 | Test AUC |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: |"
    ]
    for name, selected in payload["selected"].items():
        test = selected["test"]
        cfg_str = json.dumps(selected['config'], sort_keys=True)
        auc_str = f"{test['AUC']:.4f}" if test.get("AUC") is not None else "N/A"
        prec_str = f"{test.get('Precision@5', 0.0):.4f}"
        hr_str = f"{test.get('HitRate@5', 0.0):.4f}"
        lines.append(
            f"| {name.upper()} | `{cfg_str}` | {selected['validation_ndcg']:.4f} | "
            f"{test['NDCG@5']:.4f} | {test['Recall@5']:.4f} | {prec_str} | {hr_str} | {auc_str} |"
        )
    
    lines.extend([
        "",
        "### Chi Tiết Toàn Bộ Các Trials Trong Grid Search",
        "",
        "| Mô hình | Cấu hình thử nghiệm | Best Epoch | Val NDCG@5 | Thời gian (s) |",
        "| :-- | :-- | --: | --: | --: |"
    ])
    for t in payload.get("trials", []):
        cfg_str = json.dumps(t['config'], sort_keys=True)
        lines.append(f"| {t['model'].upper()} | `{cfg_str}` | {t['best_epoch']} | {t['best_val_ndcg']:.4f} | {t['elapsed_seconds']:.1f}s |")

    lines.extend(["", "*(Báo cáo tự động tạo bởi TopTop Collaborative Filtering Pipeline)*", ""])
    json_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def print_summary_table(selected_results: dict):
    """In bảng tổng kết kết quả đẹp mắt ra Console bằng format_grid_table."""
    headers = [
        "Mô Hình",
        "Cấu Hình Tối Ưu",
        "Val NDCG@5",
        "Test NDCG@5",
        "Test Recall@5",
        "Test AUC"
    ]
    rows = []
    for model_name, sel in selected_results.items():
        t = sel["test"]
        cfg = json.dumps(sel["config"])
        if len(cfg) > 42:
            cfg = cfg[:39] + "..."
        auc_str = f"{t['AUC']:.4f}" if t.get("AUC") is not None else "N/A"
        rows.append([
            model_name.upper(),
            cfg,
            f"{sel['validation_ndcg']:.4f}",
            f"{t['NDCG@5']:.4f}",
            f"{t['Recall@5']:.4f}",
            auc_str
        ])

    table_str = format_grid_table(
        headers=headers,
        rows=rows,
        col_aligns=["left", "left", "center", "center", "center", "center"],
        title="KẾT QUẢ CUỐI CÙNG SAU KHI TUNING TRÊN COAT DATASET",
        row_divider=True
    )
    print("\n" + table_str + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "mf", "ividr", "cdr"), default="all",
                        help="Chọn mô hình cần tuning (mặc định: all)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Số epoch tối đa cho mỗi trial (mặc định: 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Số epoch Early Stopping patience (mặc định: 10)")
    parser.add_argument("--quick", action="store_true",
                        help="Chế độ chạy nhanh (lấy mẫu nhỏ grid để kiểm tra pipeline)")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs phải lớn hơn 0")

    names = list(GRIDS) if args.model == "all" else [args.model]
    
    effective_grids = {}
    for name in names:
        if args.quick:
            effective_grids[name] = GRIDS[name][:1]
        else:
            effective_grids[name] = GRIDS[name]

    print("=" * 85)
    print(" BẮT ĐẦU QUY TRÌNH TINH CHỈNH SIÊU THAM SỐ (TUNING) TRÊN COAT DATASET")
    print(f" Mô hình: {names} | Epochs: {args.epochs} | Patience: {args.patience}")
    print("=" * 85)
    
    print(" Đang nạp dữ liệu Coat...")
    data = load_coat_processed()
    identity = {
        "models": names,
        "epochs": args.epochs,
        "seed": 42,
        "version": 4,
        "grids": effective_grids,
        "data": data_signature(data),
        "code_sha256": source_signature()
    }
    run_digest = digest(identity)
    json_path = Path(RESULTS_DIR) / f"tuning_coat_{args.model}_e{args.epochs}_{run_digest}_v4.json"
    log_path = json_path.with_suffix(".log")
    
    payload = {
        "identity": identity,
        "protocol": "paper_idcf_30_70_user_seed1234",
        "evaluation_protocol": protocol_metadata(data),
        "selection": "validation_ndcg_only",
        "epochs": args.epochs,
        "grids": effective_grids,
        "trials": [],
        "selected": {}
    }
    write_report(payload, json_path)

    original_stdout = sys.stdout
    tee = TeeLogger(log_path)
    sys.stdout = tee

    try:
        mf_checkpoints = {}
        # Nếu có tuning CDR, tiền huấn luyện MF checkpoints làm warm-start
        if "cdr" in names:
            print("\n>>> [TIỀN HUẤN LUYỆN MF ĐỂ WARM-START CHO CDR]...")
            needed_dims = sorted(list(set(c["embedding_dim"] for c in effective_grids["cdr"])))
            for dim in needed_dims:
                mf_tag = f"coat_mf_warm_{dim}_e{args.epochs}"
                mf_res = train_baseline_mf(
                    dataset_name="coat",
                    embedding_dim=dim,
                    epochs=args.epochs,
                    lr=1e-3,
                    weight_decay=1e-5,
                    batch_size=512,
                    patience=args.patience,
                    evaluate_test=False,
                    data_dict=data,
                    checkpoint_tag=mf_tag
                )
                mf_checkpoints[dim] = mf_res["checkpoint_path"]
                print(f"    -> Đã tạo MF Warm-start checkpoint cho dim={dim} (Val NDCG={mf_res['best_val_ndcg']:.4f})")

        for name in names:
            candidates = []
            grid_list = effective_grids[name]
            total_trials = len(grid_list)
            print(f"\n" + "#" * 85)
            print(f" >>> BẮT ĐẦU TINH CHỈNH MÔ HÌNH: {name.upper()} (Tổng cộng: {total_trials} trials)")
            print("#" * 85)

            for idx, config in enumerate(grid_list, 1):
                config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
                tag = f"tune_coat_e{args.epochs}_{config_digest}"
                
                print(f"\n--- [{name.upper()} Trial {idx}/{total_trials}] Cấu hình: {config} ---")
                started = time.perf_counter()
                
                result = train_model(name, config, args.epochs, False, tag)
                
                elapsed = time.perf_counter() - started
                trial = {
                    "model": name,
                    "config": config,
                    "best_val_ndcg": result["best_val_ndcg"],
                    "best_epoch": result["best_epoch"],
                    "checkpoint_path": result["checkpoint_path"],
                    "metadata": result["metadata"],
                    "elapsed_seconds": round(elapsed, 2)
                }
                payload["trials"].append(trial)
                candidates.append(trial)
                write_report(payload, json_path)
                
                print(f" [KẾT QUẢ TRIAL {idx}] Best Val NDCG@5 = {trial['best_val_ndcg']:.4f} (tại epoch {trial['best_epoch']}) | Thời gian: {elapsed:.1f}s")

            best = max(candidates, key=lambda trial: trial["best_val_ndcg"])
            best_config = best["config"]
            
            print(f"\n>>> [LỰA CHỌN TỐI ƯU CHO {name.upper()}] Cấu hình: {best_config}")
            print(f"    Điểm Validation NDCG@5 cao nhất: {best['best_val_ndcg']:.4f} (epoch {best['best_epoch']})")
            print(f"    -> Đang tiến hành đánh giá DUY NHẤT 1 LẦN trên tập Held-Out Test...")
            
            test_metrics = evaluate_checkpoint(best["checkpoint_path"], data)
            payload["selected"][name] = {
                "config": best_config,
                "validation_ndcg": best["best_val_ndcg"],
                "best_epoch": best["best_epoch"],
                "test": test_metrics
            }
            write_report(payload, json_path)
            
            auc_str = f"{test_metrics['AUC']:.4f}" if test_metrics.get("AUC") is not None else "N/A"
            print(f"    -> KẾT QUẢ TEST: NDCG@5 = {test_metrics['NDCG@5']:.4f} | Recall@5 = {test_metrics['Recall@5']:.4f} | AUC = {auc_str}")

        print_summary_table(payload["selected"])
        print(f"[HOÀN THÀNH] Báo cáo chi tiết JSON : {json_path}")
        print(f"[HOÀN THÀNH] Báo cáo khoa học Markdown: {json_path.with_suffix('.md')}")
        print(f"[HOÀN THÀNH] Toàn bộ nhật ký Log    : {log_path}\n")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
