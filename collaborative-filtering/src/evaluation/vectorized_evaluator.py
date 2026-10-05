# -*- coding: utf-8 -*-
"""
Module 3: Động Cơ Đánh Giá Vector Hóa Toàn Diện Trên GPU (GPU Vectorized Top-5 Evaluation Engine)
Đặc tả kỹ thuật & ràng buộc tính toán:
  - Không gian xếp hạng toàn diện (Unsampled Full Sort): Đánh giá trên TOÀN BỘ M items.
  - Mặt nạ huấn luyện (Masking): Gán -inf cho tất cả các item người dùng ĐÃ TƯƠNG TÁC trong tập Train.
  - Xếp hạng GPU song song theo chunks (User Chunking) với torch.matmul và torch.topk(..., k=5).
  - So khớp ground-truth bằng phép toán gather trên GPU (tuyệt đối không dùng vòng lặp for user).
  - 4 chỉ số chuẩn hóa Top-5:
      1. Recall@5 = hits / |D_test^u|
      2. NDCG@5 = DCG@5 / IDCG@5 với IDCG tính trên min(|D_test^u|, 5)
      3. Precision@5 = hits / 5.0
      4. HitRate@5 = I(hits >= 1)
Project: TopTop (agri-behavioral-recsys)
"""

import time
import numpy as np
import torch
import torch.nn as nn


class GPUVectorizedEvaluator:
    """
    Động cơ đánh giá Unbiased Top-5 vector hóa 100% trên GPU.
    Áp dụng User Chunking + GPU Gather + Matrix Operations siêu tốc.
    """
    def __init__(
        self,
        num_items: int,
        test_ground_truth: dict,
        train_user_pos_map: dict = None,
        eval_users: np.ndarray = None,
        k: int = 5,
        chunk_size: int = 512,
        device: torch.device = None
    ):
        if k < 1 or chunk_size < 1 or num_items < 0:
            raise ValueError("k/chunk_size must be positive and num_items nonnegative")
        self.num_items = num_items
        self.test_ground_truth = test_ground_truth
        self.train_user_pos_map = train_user_pos_map or {}
        self.k = k
        self.chunk_size = chunk_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Danh sách người dùng cần đánh giá (có ít nhất 1 item dương trong tập test)
        if eval_users is not None:
            self.eval_users = eval_users
        else:
            self.eval_users = np.array(sorted(list(test_ground_truth.keys())), dtype=np.int64)

        # 1. Vector hệ số chiết khấu vị trí DCG trên GPU: 1 / log2(rank + 1) với rank = 1..k
        ranks = torch.arange(1, self.k + 1, dtype=torch.float32, device=self.device)
        self.discounts = 1.0 / torch.log2(ranks + 1.0)  # Shape (k,)

        # 2. Bảng tra cứu IDCG lý tưởng cho số lượng item dương từ 0 đến k
        # idcg[n] = sum_{r=1}^n 1 / log2(r + 1)
        idcg_values = [0.0]
        curr_idcg = 0.0
        for r in range(1, self.k + 1):
            curr_idcg += 1.0 / np.log2(r + 1.0)
            idcg_values.append(curr_idcg)
        self.idcg_table = torch.tensor(idcg_values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> dict:
        """
        Thực thi đánh giá Top-5 toàn diện trên toàn bộ eval_users.
        Trả về dictionary chứa: NDCG@5, Recall@5, Precision@5, HitRate@5 và thời gian thực thi.
        """
        model.eval()
        t0 = time.time()

        total_users = len(self.eval_users)
        if total_users == 0 or self.num_items == 0:
            return {f"NDCG@{self.k}": 0.0, f"Recall@{self.k}": 0.0, f"Precision@{self.k}": 0.0, f"HitRate@{self.k}": 0.0, "eval_users": 0}

        all_recall = []
        all_ndcg = []
        all_precision = []
        all_hitrate = []

        # Duyệt qua các khối người dùng (User Chunking) để kiểm soát bộ nhớ GPU
        for start_idx in range(0, total_users, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_users)
            chunk_u_np = self.eval_users[start_idx:end_idx]
            B = len(chunk_u_np)
            chunk_u_tensor = torch.tensor(chunk_u_np, dtype=torch.long, device=self.device)

            # 1. Dự đoán điểm số trên TOÀN BỘ M items bằng GPU Matrix Multiplication
            # scores có kích thước (B, num_items)
            if hasattr(model, "predict_user_all"):
                scores = model.predict_user_all(chunk_u_tensor)
            else:
                # Fallback nếu model là nn.Module thông thường
                scores = model(chunk_u_tensor)

            # 2. MẶT NẠ HUẤN LUYỆN (Train Masking):
            # Gán -inf cho tất cả các item người dùng ĐÃ TƯƠNG TÁC trong tập Train để tránh rò rỉ dữ liệu
            for b_idx, u in enumerate(chunk_u_np):
                train_items = self.train_user_pos_map.get(u)
                if train_items:
                    train_idx_tensor = torch.tensor(list(train_items), dtype=torch.long, device=self.device)
                    scores[b_idx, train_idx_tensor] = -float('inf')

            # 3. LẤY TOP-5 XẾP HẠNG CỰC NHANH TRÊN GPU
            # topk_indices có kích thước (B, 5)
            _, topk_indices = torch.topk(scores, k=min(self.k, self.num_items), dim=-1)

            # 4. SO KHỚP GROUND-TRUTH TRÊN GPU BẰNG PHÉP GATHER (Không dùng vòng lặp for user)
            # Tạo ma trận nhị phân đánh dấu các item dương trong tập test của chunk hiện tại
            is_pos_chunk = torch.zeros((B, self.num_items), dtype=torch.bool, device=self.device)
            pos_counts_list = []

            for b_idx, u in enumerate(chunk_u_np):
                test_pos_items = set(self.test_ground_truth.get(u, ())) - set(self.train_user_pos_map.get(u, ()))
                pos_counts_list.append(len(test_pos_items))
                if test_pos_items:
                    is_pos_chunk[b_idx, list(test_pos_items)] = True


            pos_counts = torch.tensor(pos_counts_list, dtype=torch.float32, device=self.device)
            # Tránh chia cho 0 nếu user không có item dương
            pos_counts_safe = torch.clamp(pos_counts, min=1.0)

            # Trích xuất ma trận hits (B, 5): 1.0 nếu trúng ground-truth, 0.0 nếu trượt
            hits = is_pos_chunk.gather(1, topk_indices).float()  # (B, 5)
            hit_counts = hits.sum(dim=-1)                       # (B,)

            # 5. TÍNH TOÁN CÁC CHỈ SỐ VECTOR HÓA TRÊN GPU THEO ĐÚNG CÔNG THỨC:
            # A. Recall@5 = (Số item dương nằm trong Top-5) / |D_test^u|
            recalls_b = hit_counts / pos_counts_safe

            # B. Precision@5 = (Số item dương nằm trong Top-5) / 5.0
            precisions_b = hit_counts / float(self.k)

            # C. HitRate@5 = 1 nếu có ít nhất 1 item dương trong Top-5, ngược lại = 0
            hitrates_b = (hit_counts > 0.0).float()

            # D. NDCG@5:
            # DCG@5 = sum_{r=1}^5 hits[r] / log2(r + 1)
            dcg_b = torch.sum(hits * self.discounts[:min(self.k, self.num_items)].unsqueeze(0), dim=-1)  # (B,)
            # IDCG@5 tính trên min(|D_test^u|, 5)
            ideal_hits = torch.clamp(pos_counts, max=float(self.k)).long()
            idcg_b = self.idcg_table[ideal_hits]
            idcg_safe = torch.clamp(idcg_b, min=1e-8)
            ndcg_b = dcg_b / idcg_safe

            # Chuyển kết quả chunk sang CPU list để lưu trữ
            all_recall.extend(recalls_b.cpu().tolist())
            all_ndcg.extend(ndcg_b.cpu().tolist())
            all_precision.extend(precisions_b.cpu().tolist())
            all_hitrate.extend(hitrates_b.cpu().tolist())

        elapsed = time.time() - t0

        results = {
            f"NDCG@{self.k}": float(np.mean(all_ndcg)),
            f"Recall@{self.k}": float(np.mean(all_recall)),
            f"Precision@{self.k}": float(np.mean(all_precision)),
            f"HitRate@{self.k}": float(np.mean(all_hitrate)),
            "eval_users": total_users,
            "elapsed_seconds": round(elapsed, 2)
        }
        return results

