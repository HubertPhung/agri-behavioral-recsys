# -*- coding: utf-8 -*-
"""
Master Benchmark Runner: Thực hiện toàn diện thực nghiệm đối sánh
3 mô hình (Baseline MF, IViDR, DR-BIAS + CDR) trên 2 dataset (Coat, KuaiRand-Pure)
và hỗ trợ huấn luyện riêng lẻ, quét siêu tham số Grid Search.
Project: TopTop (agri-behavioral-recsys)
"""

import os
import sys
import json
import time
import argparse
import hashlib

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# Thêm thư mục gốc dự án vào PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import set_seed, get_device, RESULTS_DIR
from src.utils.artifacts import validate_training, data_signature, source_signature, digest
from src.data_pipeline.kuairand_dataset import subset_kuairand_users
from src.baselines.train_mf import train_baseline_mf
from src.causal.ividr.train_ividr import train_ividr
from src.causal.cdr.train_dr_bias_cdr import train_dr_bias_cdr
from src.evaluation.diagnostics import evaluate_popularity, candidate_pool_diagnostics
from src.data_pipeline.coat_dataset import load_coat_processed
from src.data_pipeline.kuairand_dataset import load_kuairand_processed
from src.evaluation.protocol import protocol_metadata


def run_all_benchmarks(
    quick_run: bool = False,
    epochs_coat: int = None,
    epochs_kr: int = None,
    dataset_filter: str = "all",
    model_filter: str = "all",
    dim: int = None,
    lr: float = None,
    batch_size: int = None,
    alpha: float = 0.0,
    grid_search: bool = False,
    smoke_users: int = None,
):
    validate_training(epochs_coat=epochs_coat, epochs_kr=epochs_kr, dim=dim,
                      lr=lr, batch_size=batch_size, smoke_users=smoke_users)
    if dataset_filter not in ("all", "coat", "kuairand") or model_filter not in ("all", "mf", "ividr", "cdr"):
        raise ValueError("Unsupported dataset/model")
    import math
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if grid_search and (dataset_filter != "kuairand" or model_filter != "mf"):
        raise ValueError("--grid-search requires --dataset kuairand --model mf; use tune_coat_models.py for Coat")
    if grid_search and (lr is not None or quick_run or smoke_users is not None):
        raise ValueError("Grid search does not accept --lr, --quick or --smoke-users")
    if smoke_users is not None and (not quick_run or dataset_filter != "kuairand"):
        raise ValueError("--smoke-users requires --quick --dataset kuairand")
    set_seed(42)
    device = get_device()

    def checkpoint_tag(name, epoch_count):
        config = {"model": name, "epochs": epoch_count, "quick": quick_run,
                  "dim": dim, "lr": lr, "batch_size": batch_size, "alpha": alpha,
                  "kuairand_protocol": "paper_idcf", "coat_protocol": "paper_idcf",
                  "ividr_exposure_reconstruction": "full_coat_sampled_128_kuairand"}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        return f"v4_e{epoch_count}_{'quick' if quick_run else 'full'}_{digest}"

    # Xử lý chế độ Quét Siêu Tham Số (Grid Search) nếu được yêu cầu
    if grid_search:
        from src.training.train_pipeline import run_grid_search
        data_dict = load_kuairand_processed(alpha=alpha)
        best_cfg, trials = run_grid_search(
            data_dict=data_dict,
            embedding_dim=dim or 128,
            batch_size=batch_size or 4096,
            epochs=epochs_kr or 20,
            device=device
        )
        grid_config = {"data": data_signature(data_dict), "code_sha256": source_signature(),
                       "artifact_version": 4, "seed": 42,
                       "trial_metadata": [t["metadata"] for t in trials], "alpha": alpha, "kuairand_protocol": "paper_idcf", "dim": dim or 128, "batch_size": batch_size or 4096,
                       "epochs": epochs_kr or 20}
        grid_digest = hashlib.sha256(json.dumps(grid_config, sort_keys=True).encode()).hexdigest()[:10]
        summary_path = os.path.join(RESULTS_DIR, f"grid_search_v4_{grid_digest}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"protocol": "validation_tuning_then_one_held_out_test", "config": grid_config,
                       "evaluation_protocol": protocol_metadata(data_dict),
                       "best_config": best_cfg, "all_trials": trials}, f, indent=2)
        print(f"[OK] Đã lưu kết quả Grid Search tại: {summary_path}")
        return

    print("=" * 80)
    print(" BẮT ĐẦU CHẠY THỰC NGHIỆM ĐỐI SÁNH CAUSAL DEBIASING TRÊN COAT & KUAIRAND")
    print(f" Thiết bị: {device} | Dataset: {dataset_filter} | Model: {model_filter}")
    print("=" * 80)

    # Cấu hình số epoch linh hoạt
    if quick_run:
        epochs_coat = 5 if epochs_coat is None else epochs_coat
        epochs_kr = 3 if epochs_kr is None else epochs_kr
    else:
        epochs_coat = 100 if epochs_coat is None else epochs_coat
        epochs_kr = 25 if epochs_kr is None else epochs_kr

    print(f" [CẤU HÌNH EPOCHS] Coat: {epochs_coat} epochs | KuaiRand-Pure: {epochs_kr} epochs")

    all_results = {
        "artifact_version": 4,
        "coat": {},
        "kuairand": {},
        "diagnostics": {},
        "evaluation_protocol": {},
        "protocol": {
            "selection": "validation_only",
            "test": "held_out_final_evaluation",
            "candidate_pool": "logged_exposures_only",
            "include_zero_positive_users": {"coat": False, "kuairand": False},
            "coat_protocol": "paper_idcf",
            "ividr_exposure_reconstruction": "full_coat_sampled_128_kuairand",
            "ividr_phi": 1.0,
            "kuairand_protocol": "paper_idcf",
            "kuairand_random_val_fraction": 0.3,
            "paper_scores_directly_comparable": False,
            "coat_random_split_seed": 1234,
            "alpha": alpha,
            "alpha_applies_to": ["MF", "IViDR"],
            "kuairand_cdr_propensity": "item_exposure_frequency_proxy_clipped_0.05_to_1.0",
            "quick_run": quick_run,
            "seed": 42,
            "epochs_coat": epochs_coat,
            "epochs_kuairand": epochs_kr,
            "dimension_override": dim,
            "learning_rate_override": lr,
            "batch_size_override": batch_size,
        },
    }
    errors = []

    # =========================================================================
    # 1. THỰC NGHIỆM TRÊN COAT DATASET
    # =========================================================================
    if dataset_filter in ["all", "coat"]:
        print("\n" + "#" * 80)
        print(" PHẦN 1: THỰC NGHIỆM TRÊN TẬP DỮ LIỆU COAT (290 Users, 300 Items)")
        print("#" * 80)
        coat_data = load_coat_processed()
        all_results["evaluation_protocol"]["coat"] = protocol_metadata(coat_data)
        all_results["protocol"]["coat_split_info"] = coat_data["split_info"]
        coat_diagnostics = candidate_pool_diagnostics(coat_data)
        all_results["coat"]["Random (expected)"] = coat_diagnostics["random_expected"]
        all_results["coat"]["Popularity"] = evaluate_popularity(coat_data)
        all_results["diagnostics"]["coat"] = coat_diagnostics
        all_results["diagnostics"]["coat_validation"] = candidate_pool_diagnostics(coat_data, split="val")

        if model_filter in ["all", "mf"]:
            print("\n>>> [1/3] Huấn luyện Baseline PyTorch MF (Coat)...")
            try:
                all_results["coat"]["MF"] = train_baseline_mf(
                    dataset_name="coat",
                    embedding_dim=dim or 64,
                    epochs=epochs_coat,
                    lr=lr or 1e-3,
                    weight_decay=1e-5,
                    batch_size=batch_size or 512,
                    device=device,
                    data_dict=coat_data,
                    checkpoint_tag=checkpoint_tag("coat_mf", epochs_coat),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện MF trên Coat thất bại: {e}")
                errors.append(("coat", "MF", str(e)))

        if model_filter in ["all", "ividr"]:
            print("\n>>> [2/3] Huấn luyện IViDR (Coat)...")
            try:
                all_results["coat"]["IViDR"] = train_ividr(
                    dataset_name="coat",
                    embedding_dim=dim or 64,
                    latent_dim=32,
                    epochs=epochs_coat,
                    patience=20,
                    lr=lr or 3e-3,
                    phi=2.0,
                    beta_elbo=0.01,
                    batch_size=batch_size or 512,
                    device=device,
                    data_dict=coat_data,
                    checkpoint_tag=checkpoint_tag("coat_ividr", epochs_coat),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện IViDR trên Coat thất bại: {e}")
                errors.append(("coat", "IViDR", str(e)))

        if model_filter in ["all", "cdr"]:
            print("\n>>> [3/3] Huấn luyện DR-BIAS + CDR (Coat)...")
            try:
                cdr_dim = dim or 32
                mf_warm = all_results["coat"].get("MF")
                initial_mf_checkpoint = None
                if mf_warm and mf_warm.get("metadata", {}).get("embedding_dim") == cdr_dim:
                    initial_mf_checkpoint = mf_warm.get("checkpoint_path")
                if initial_mf_checkpoint is None:
                    mf_warm_start = train_baseline_mf(
                        dataset_name="coat", embedding_dim=cdr_dim,
                        epochs=epochs_coat, lr=1e-3,
                        weight_decay=1e-5,
                        batch_size=batch_size or 512, device=device,
                        data_dict=coat_data, evaluate_test=False,
                        checkpoint_tag=checkpoint_tag(f"coat_cdr_warm_mf_{cdr_dim}", epochs_coat),
                    )
                    initial_mf_checkpoint = mf_warm_start["checkpoint_path"]
                all_results["coat"]["DR-BIAS + CDR"] = train_dr_bias_cdr(
                    dataset_name="coat",
                    embedding_dim=cdr_dim,
                    un_thres=0.01,
                    max_inverse_propensity=20.0,
                    epochs=epochs_coat,
                    lr=lr or 0.002,
                    batch_size=batch_size or 128,
                    device=device,
                    data_dict=coat_data,
                    initial_rec_checkpoint=initial_mf_checkpoint,
                    checkpoint_tag=checkpoint_tag("coat_cdr", epochs_coat),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện DR-BIAS + CDR trên Coat thất bại: {e}")
                errors.append(("coat", "DR-BIAS + CDR", str(e)))

    # =========================================================================
    # 2. THỰC NGHIỆM TRÊN KUAIRAND-PURE DATASET (IS_CLICK)
    # =========================================================================
    if dataset_filter in ["all", "kuairand"]:
        print("\n" + "#" * 80)
        print(" PHẦN 2: THỰC NGHIỆM KUAIRAND-PURE (IS_CLICK, IDCF DATA SPLIT)")
        print("#" * 80)
        kr_data = load_kuairand_processed(alpha=alpha)
        if smoke_users is not None:
            kr_data = subset_kuairand_users(kr_data, smoke_users)
        all_results["evaluation_protocol"]["kuairand"] = protocol_metadata(kr_data)
        kr_diagnostics = candidate_pool_diagnostics(kr_data)
        all_results["kuairand"]["Random (expected)"] = kr_diagnostics["random_expected"]
        all_results["kuairand"]["Popularity"] = evaluate_popularity(kr_data)
        all_results["diagnostics"]["kuairand"] = kr_diagnostics
        all_results["diagnostics"]["kuairand_validation"] = candidate_pool_diagnostics(kr_data, split="val")
        all_results["protocol"]["kuairand_split_info"] = kr_data["split_info"]

        if model_filter in ["all", "mf"]:
            print("\n>>> [1/3] Huấn luyện Baseline PyTorch MF (KuaiRand)...")
            try:
                all_results["kuairand"]["MF"] = train_baseline_mf(
                    dataset_name="kuairand",
                    embedding_dim=dim or 128,
                    epochs=epochs_kr,
                    lr=lr or 5e-4,
                    weight_decay=1e-5,
                    batch_size=batch_size or 4096,
                    device=device,
                    alpha=alpha,
                    data_dict=kr_data,
                    checkpoint_tag=checkpoint_tag("kuairand_mf", epochs_kr),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện MF trên KuaiRand thất bại: {e}")
                errors.append(("kuairand", "MF", str(e)))

        if model_filter in ["all", "ividr"]:
            print("\n>>> [2/3] Huấn luyện IViDR (KuaiRand)...")
            try:
                all_results["kuairand"]["IViDR"] = train_ividr(
                    dataset_name="kuairand",
                    embedding_dim=dim or 128,
                    latent_dim=32,
                    epochs=epochs_kr,
                    lr=lr or 2e-3,
                    phi=0.1,
                    beta_elbo=0.001,
                    batch_size=batch_size or 4096,
                    device=device,
                    alpha=alpha,
                    data_dict=kr_data,
                    checkpoint_tag=checkpoint_tag("kuairand_ividr", epochs_kr),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện IViDR trên KuaiRand thất bại: {e}")
                errors.append(("kuairand", "IViDR", str(e)))

        if model_filter in ["all", "cdr"]:
            print("\n>>> [3/3] Huấn luyện DR-BIAS + CDR (KuaiRand)...")
            try:
                all_results["kuairand"]["DR-BIAS + CDR"] = train_dr_bias_cdr(
                    dataset_name="kuairand",
                    embedding_dim=dim or 128,
                    un_thres=0.005,
                    epochs=epochs_kr,
                    lr=lr or 1e-3,
                    batch_size=batch_size or 8192,
                    device=device,
                    alpha=alpha,
                    data_dict=kr_data,
                    checkpoint_tag=checkpoint_tag("kuairand_cdr", epochs_kr),
                )
            except Exception as e:
                print(f"[LỖI] Huấn luyện DR-BIAS + CDR trên KuaiRand thất bại: {e}")
                errors.append(("kuairand", "DR-BIAS + CDR", str(e)))

    # =========================================================================
    # 3. LƯU KẾT QUẢ VÀ TẠO BÁO CÁO TỔNG HỢP
    # =========================================================================
    if errors:
        raise RuntimeError(f"Benchmark incomplete; no report written. Failures: {errors}")
    run_tag = "all" if dataset_filter == model_filter == "all" and not quick_run else f"{dataset_filter}_{model_filter}{'_quick' if quick_run else ''}"
    model_metadata = {d: {name: value["metadata"] for name, value in all_results[d].items()
                          if "metadata" in value} for d in ("coat", "kuairand")}
    all_results["protocol"]["smoke_users"] = smoke_users
    report_config = {"effective_runs": model_metadata, "version": 4, "smoke_users": smoke_users, "dataset": dataset_filter, "model": model_filter, "epochs_coat": epochs_coat,
                     "epochs_kr": epochs_kr, "dim": dim, "lr": lr, "batch_size": batch_size, "alpha": alpha,
                     "kuairand_protocol": "paper_idcf", "coat_protocol": "paper_idcf",
                     "ividr_exposure_reconstruction": "full_coat_sampled_128_kuairand"}
    run_tag += "_" + hashlib.sha256(json.dumps(report_config, sort_keys=True).encode()).hexdigest()[:10]
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{run_tag}_v4.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Đã lưu kết quả chi tiết JSON tại: {json_path}")

    # Tạo bảng Markdown chuẩn duy nhất
    table_md = generate_markdown_report(all_results)
    table_path = os.path.join(RESULTS_DIR, f"benchmark_{run_tag}_v4.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"[OK] Đã lưu báo cáo khoa học Markdown tại: {table_path}")

    # In bảng lưới đóng khung chuyên nghiệp ra Console (có khoảng chắn giữa các hàng/cột)
    console_report = generate_console_grid_report(all_results)
    print(console_report)
    return all_results


def get_benchmark_tables_data(results):
    headers_local = [
        "Tập Dữ Liệu",
        "Mô Hình Thuật Toán",
        "NDCG@5",
        "Recall@5",
        "Precision@5",
        "HitRate@5",
        "AUC (macro-user)"
    ]
    rows_local = []
    for dname, dtitle in [("coat", "Coat"), ("kuairand", "KuaiRand-Pure")]:
        d_res = results.get(dname, {})
        for mname in ["MF", "DR-BIAS + CDR", "IViDR"]:
            if mname in d_res:
                m = d_res[mname]
                ndcg = f"{m.get('NDCG@5', 0.0):.4f}"
                rec = f"{m.get('Recall@5', 0.0):.4f}"
                prec = f"{m.get('Precision@5', 0.0):.4f}"
                hr = f"{m.get('HitRate@5', 0.0):.4f}"
                auc = f"{m['AUC']:.4f}" if m.get("AUC") is not None else "N/A"
                rows_local.append([dtitle, mname, ndcg, rec, prec, hr, auc])

    return headers_local, rows_local


def generate_console_grid_report(results):
    from src.evaluation.table_formatter import format_grid_table
    headers_local, rows_local = get_benchmark_tables_data(results)

    grid_local = format_grid_table(
        headers=headers_local,
        rows=rows_local,
        col_aligns=["left", "left", "center", "center", "center", "center", "center"],
        title="KẾT QUẢ TRÊN HELD-OUT RANDOM-LOG CANDIDATE POOL",
        row_divider=True
    )

    return f"\n{grid_local}\n"


def generate_markdown_report(results):
    from src.evaluation.table_formatter import format_markdown_table
    headers_local, rows_local = get_benchmark_tables_data(results)

    r_local_md = [[f"**{row[0]}**"] + row[1:] for row in rows_local]
    md_local = format_markdown_table(
        headers=headers_local,
        rows=r_local_md,
        col_aligns=["left", "left", "center", "center", "center", "center", "center"]
    )

    lines = [
        "# Benchmark v4: validation và test độc lập",
        "",
        (f"**SMOKE TEST — mẫu {results['protocol']['smoke_users']} user KuaiRand; candidate pool đã thu nhỏ, không dùng so điểm bài báo.**"
         if results["protocol"].get("smoke_users") else
         "**SMOKE TEST — kiểm tra vận hành, chưa phải kết quả hội tụ.**" if results["protocol"].get("quick_run") else
         "**Benchmark đầy đủ dữ liệu; một seed huấn luyện, chọn checkpoint bằng validation.**"),
        "",
        f"**Thời gian cập nhật:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Chỉ xếp hạng các exposure đã ghi log. Chọn epoch trên validation; test giữ riêng.",
        "Chỉ số chính: NDCG@5 và Recall@5. AUC là chỉ số phụ lấy trung bình theo user có hai lớp; không mặc định tương đương AUC trong bài CDR.",
        "Coat và KuaiRand chia random log theo user, 30% validation / 70% test; macro metric chỉ chấm user có nhãn dương.",
        "Kết quả địa phương chưa thể so trực tiếp với bảng bài báo: mô hình, siêu tham số, seed và giao thức giữa các bài báo còn khác. Bảng tham chiếu có chú giải riêng tại `docs/paper_reference_comparison.md`.",
        "",
        "### Kết quả trên held-out test:",
        "",
        md_local,
        ""
    ]
    for name in ("coat", "kuairand"):
        d = results.get("diagnostics", {}).get(name)
        if d:
            val_d = results["diagnostics"].get(f"{name}_validation", {})
            lines.extend([
                f"### Candidate pool {name}",
                "",
                f"- Users: {d['users']:,}; zero-positive users: {d['zero_positive_users']:,}; candidates: {d['candidate_pairs']:,}.",
                f"- Positive prevalence: {d['pooled_positive_rate']:.4f}; candidate size p10/p50/p90: {d['candidate_size_p10_p50_p90']}.",
                f"- Cold users: {d['cold_user_fraction']:.2%}; cold candidates: {d['cold_candidate_fraction']:.2%}.",
                f"- Validation positive prevalence: {val_d.get('pooled_positive_rate', 0.0):.4f}; validation candidate p50: {val_d.get('candidate_size_p10_p50_p90', [0, 0, 0])[1]:.0f}.",
                "",
            ])
            popularity = results[name]["Popularity"]["NDCG@5"]
            for model_name in ("MF", "IViDR", "DR-BIAS + CDR"):
                if model_name in results[name]:
                    metric = results[name][model_name]
                    lines.append(
                        f"- {model_name}: ΔNDCG@5 so với Popularity = "
                        f"{metric['NDCG@5'] - popularity:+.4f}; "
                        f"CI95 NDCG của mô hình, theo user ≈ ±{metric['NDCG@5_CI95_halfwidth']:.4f}."
                    )
            lines.append("")
    coat_split = results["protocol"].get("coat_split_info")
    if coat_split:
        lines.extend([
            f"Coat train/random overlap: {coat_split['train_random_overlap_pairs']} cặp; "
            f"test overlap: {coat_split['test_overlap_pairs']} cặp, "
            f"gồm {coat_split['test_overlap_positive_pairs']} cặp dương.",
            "",
        ])
    lines.extend([
        "### Giới hạn sai số",
        "",
        "- Khoảng tin cậy trong JSON là xấp xỉ theo người dùng của một lần chạy; chưa bao gồm biến thiên do seed hoặc chọn cấu hình.",
        "- Random (expected) là kỳ vọng xếp hạng ngẫu nhiên có điều kiện trên candidate pool hiện có; Popularity dùng tần suất exposure trong train.",
        "- KuaiRand dùng duy nhất is_click, ghép hai standard log, lọc user/item tối thiểu 10 exposure một lượt và chia random log theo user. CDR dùng tần suất exposure theo item làm proxy propensity.",
        "- Coat dùng rating ≥ 4 là dương; 6.960 biased ratings để train và 4.640 random ratings chia theo user seed 1234. Các cặp trùng train/test giữ nguyên để đối chiếu giao thức bài báo.",
        "- IViDR huấn luyện decoder tái tạo exposure từ biased train: toàn bộ item ở Coat và 128 item lấy mẫu đều mỗi batch ở KuaiRand.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Master Benchmark Runner: Huấn luyện và đánh giá Causal Debiasing trên nhãn is_click")
    parser.add_argument("--quick", action="store_true", help="Chạy nhanh kiểm thử (ít epoch)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "coat", "kuairand"], help="Chọn dataset thực thi")
    parser.add_argument("--model", type=str, default="all", choices=["all", "mf", "ividr", "cdr"], help="Chọn mô hình thực thi (mặc định all)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch chung cho các dataset")
    parser.add_argument("--epochs-coat", type=int, default=None, help="Số epoch cho Coat (mặc định 100)")
    parser.add_argument("--epochs-kr", type=int, default=None, help="Số epoch cho KuaiRand (mặc định 25)")
    parser.add_argument("--dim", type=int, default=None, help="Số chiều embedding (mặc định theo từng mô hình)")
    parser.add_argument("--lr", type=float, default=None, help="Tốc độ học Adam")
    parser.add_argument("--batch-size", type=int, default=None, help="Kích thước batch")
    parser.add_argument("--alpha", type=float, default=0.0, help="Trọng số positive cho MF/IViDR (mặc định 0; không áp dụng vào loss CDR)")
    parser.add_argument("--grid-search", action="store_true", help="Kích hoạt quét siêu tham số Grid Search trên KuaiRand")
    parser.add_argument("--smoke-users", type=int, default=None, help="Subset of KuaiRand users for smoke testing; requires --quick")
    args = parser.parse_args(argv)

    ep_coat = args.epochs if args.epochs is not None else args.epochs_coat
    ep_kr = args.epochs if args.epochs is not None else args.epochs_kr

    run_all_benchmarks(
        quick_run=args.quick,
        epochs_coat=ep_coat,
        epochs_kr=ep_kr,
        dataset_filter=args.dataset,
        model_filter=args.model,
        dim=args.dim,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        grid_search=args.grid_search,
        smoke_users=args.smoke_users,
    )
