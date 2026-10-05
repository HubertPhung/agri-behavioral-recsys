# -*- coding: utf-8 -*-
"""
Module 2A: Mô hình Gợi Ý Baseline Matrix Factorization (MF) với Implicit Feedback
Áp dụng phân tích nhân tử ma trận có bias và khởi tạo chuẩn hoá:
  - user_embedding: (num_users, embedding_dim) ~ N(0, 0.01^2)
  - item_embedding: (num_items, embedding_dim) ~ N(0, 0.01^2)
  - user_bias: (num_users, 1) ~ 0
  - item_bias: (num_items, 1) ~ 0
  - Forward: logit_ui = <p_u, q_i> + b_u + b_i
Hỗ trợ trích xuất toàn bộ nhân tử phục vụ GPU Vectorized Top-5 Engine.
Project: TopTop (agri-behavioral-recsys)
"""

import numpy as np
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Mô hình Matrix Factorization (MF) chuẩn mực cho bài toán Implicit Feedback và Causal Debiasing.
    Công thức: logit(u, i) = <p_u, q_i> + b_u + b_i + mu
    Hỗ trợ cả 2 giao thức đánh giá:
      1. predict_user_all(): Nhân ma trận GPU toàn phần cho Unsampled Full-Sort (vectorized_evaluator.py)
      2. get_evaluation_factors(): Mở rộng factor ma trận (D+2) cho Candidate Re-ranking (evaluator.py)
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128, use_bias: bool = True,
                 train_global_bias: bool = True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # 1. Các ma trận nhúng người dùng và vật phẩm
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 2. Các số hạng chệch (Bias terms) & Global Bias
        if self.use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=train_global_bias)

        # 3. Khởi tạo tham số chuẩn mực
        self.reset_parameters()

    def reset_parameters(self):
        """Khởi tạo trọng số bằng phân phối chuẩn N(0, 0.01^2) và bias bằng 0."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            nn.init.zeros_(self.global_bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến tính toán logits:
          logit_ui = sum(p_u * q_i, dim=-1) + b_u + b_i + mu
        """
        p_u = self.user_embedding(user_indices)  # (B, D)
        q_i = self.item_embedding(item_indices)  # (B, D)
        logits = torch.sum(p_u * q_i, dim=-1)    # (B,)

        if self.use_bias:
            b_u = self.user_bias(user_indices).squeeze(-1)
            b_i = self.item_bias(item_indices).squeeze(-1)
            logits = logits + b_u + b_i + self.global_bias

        return logits

    def predict_user_all(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán logits của một nhóm users trên TOÀN BỘ M items bằng nhân ma trận GPU:
          Scores = P @ Q.T + b_u + (b_i + mu).T
        user_indices: Tensor (B,)
        Trả về: Tensor logits (B, num_items)
        """
        user_indices = torch.as_tensor(user_indices, device=self.user_embedding.weight.device).view(-1)
        p_u = self.user_embedding(user_indices)       # (B, D)
        q_all = self.item_embedding.weight            # (M, D)

        scores = torch.matmul(p_u, q_all.t())         # (B, M)

        if self.use_bias:
            b_u = self.user_bias(user_indices)        # (B, 1)
            b_all = self.item_bias.weight.squeeze(-1) + self.global_bias  # (M,)
            scores = scores + b_u + b_all.unsqueeze(0)

        return scores

    def get_evaluation_factors(self):
        """
        Trích xuất u_factors và i_factors dạng NumPy mở rộng (D+2) tích hợp bias:
          scores = np.dot(i_factors[cand_items], u_factors[u])
        """
        with torch.no_grad():
            p = self.user_embedding.weight.detach().cpu().numpy()
            q = self.item_embedding.weight.detach().cpu().numpy()

            if self.use_bias:
                mu_half = (self.global_bias.item() * 0.5)
                b_u = self.user_bias.weight.detach().cpu().numpy() + mu_half
                b_i = self.item_bias.weight.detach().cpu().numpy() + mu_half

                ones_u = np.ones((self.num_users, 1), dtype=np.float32)
                ones_i = np.ones((self.num_items, 1), dtype=np.float32)

                u_factors = np.hstack([p, b_u, ones_u])
                i_factors = np.hstack([q, ones_i, b_i])
            else:
                u_factors = p
                i_factors = q

            return u_factors, i_factors

    def get_all_factors(self):
        """Trích xuất tham số an toàn (đã detach) cho tensor operations."""
        with torch.no_grad():
            if self.use_bias:
                return (
                    self.user_embedding.weight.detach(),
                    self.item_embedding.weight.detach(),
                    self.user_bias.weight.detach(),
                    self.item_bias.weight.detach(),
                    self.global_bias.detach()
                )
            return (self.user_embedding.weight.detach(), self.item_embedding.weight.detach())
