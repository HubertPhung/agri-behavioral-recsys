# -*- coding: utf-8 -*-
"""
Legacy full-sort example; not used by run_all_benchmarks.
Module 2B: Khung Khử Lệch Tổng Quát (Abstract Debiasing Framework)
Cấu trúc sẵn sàng mở rộng và tích hợp:
  1. CDR - Conservative Doubly Robust (CIKM '23):
     - Lọc Poisonous Imputation bằng Monte Carlo Dropout (10 mẫu, p=0.5)
     - Ngưỡng lọc bất định eta = 5.0
  2. IViDR - Instrumental Variable Disentangled Debiasing (2024):
     - Tái tạo Treatment bằng 2SLS qua biến công cụ Z
     - Suy diễn biến ẩn C qua iVAE với LayerNorm và hàm mục tiêu cộng tính
     - Tham số kết hợp: phi = 1.0, lambda = 1.0, rho = 0.9, tau = 0.9
Project: TopTop (agri-behavioral-recsys)
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.matrix_factorization import MatrixFactorization


class BaseDebiasModel(nn.Module, abc.ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các mô hình khử lệch nhân quả (Causal Debiasing).
    Cung cấp giao diện thống nhất cho việc tính toán Loss khử lệch và xếp hạng Full-Sort.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        # Siêu tham số chuẩn CDR (CIKM '23)
        cdr_eta: float = 5.0,
        cdr_mc_samples: int = 10,
        cdr_dropout_rate: float = 0.5,
        # Siêu tham số chuẩn IViDR (2024)
        ividr_phi: float = 1.0,
        ividr_lam: float = 1.0,
        ividr_rho: float = 0.9,
        ividr_tau: float = 0.9
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 1. Khung tham số chuẩn CDR (Song et al., CIKM '23)
        self.cdr_eta = cdr_eta
        self.cdr_mc_samples = cdr_mc_samples
        self.cdr_dropout_rate = cdr_dropout_rate

        # 2. Khung tham số chuẩn IViDR (Deng et al., 2024)
        self.ividr_phi = ividr_phi
        self.ividr_lam = ividr_lam
        self.ividr_rho = ividr_rho
        self.ividr_tau = ividr_tau

        # Backbone cốt lõi (Matrix Factorization)
        self.backbone = MatrixFactorization(num_users, num_items, embedding_dim=embedding_dim)

    @abc.abstractmethod
    def compute_debiased_loss(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                              targets: torch.Tensor, confidences: torch.Tensor,
                              **kwargs) -> dict:
        """
        Tính toán hàm mất mát khử lệch có trọng số.
        Trả về dict chứa: 'loss' (Tensor dùng để backward) và các thành phần loss chi tiết.
        """
        pass

    @abc.abstractmethod
    def predict_user_all(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán điểm xếp hạng của một nhóm users trên TOÀN BỘ M items (Full Ranking Space).
        Trả về: Tensor điểm số (B, num_items)
        """
        pass


class CDRReadyDebiasWrapper(BaseDebiasModel):
    """
    Legacy demonstration wrapper, not the benchmark CDR or IViDR implementation.
    Có thể hoạt động ở chế độ:
      - 'baseline_mf': Matrix Factorization chuẩn hóa với Weighted BCE Loss
      - 'cdr_mode': Kích hoạt Monte Carlo Dropout Imputation và lọc thận trọng eta
      - 'ividr_mode': Kích hoạt điều chỉnh biến ẩn C cộng tính
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        mode: str = "baseline_mf",
        cdr_eta: float = 5.0,
        ividr_phi: float = 1.0,
        ividr_lam: float = 1.0
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            cdr_eta=cdr_eta,
            ividr_phi=ividr_phi,
            ividr_lam=ividr_lam
        )
        self.mode = mode

        # Hook mở rộng CDR: Imputation Model với Monte Carlo Dropout
        self.imp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.imp_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mc_dropout = nn.Dropout(p=self.cdr_dropout_rate)
        nn.init.normal_(self.imp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.imp_item_embedding.weight, std=0.01)

        # Hook mở rộng IViDR: Item Confounder Projection với LayerNorm
        self.item_c_proj = nn.Embedding(num_items, embedding_dim)
        self.c_norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.item_c_proj.weight, std=0.01)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass mặc định qua backbone Matrix Factorization."""
        return self.backbone(user_indices, item_indices)

    def compute_debiased_loss(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        targets: torch.Tensor,
        confidences: torch.Tensor,
        inv_propensities: torch.Tensor = None,
        **kwargs
    ) -> dict:
        """
        Tính toán hàm mất mát có trọng số mẫu:
          Loss = (1 / B) * sum c_ui * BCE(logit_ui, y_ui)
        Nếu ở chế độ 'cdr_mode': Tích hợp kiểm tra độ bất định phương sai MC Dropout và lọc theo eta.
        """
        logits = self.backbone(user_indices, item_indices)

        # 1. Hàm mất mát BCE có trọng số tin cậy mẫu c_ui
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = bce_raw * confidences

        if self.mode == "cdr_mode":
            # --- CDR CIKM '23 Hook: Đánh giá độ bất định phương sai Imputation từng mẫu ---
            with torch.no_grad():
                preds_list = []
                for _ in range(self.cdr_mc_samples):
                    # Unscaled dropout training=True cho Monte Carlo
                    u_drop = F.dropout(self.imp_user_embedding(user_indices), p=self.cdr_dropout_rate, training=True)
                    i_drop = F.dropout(self.imp_item_embedding(item_indices), p=self.cdr_dropout_rate, training=True)
                    preds_list.append(torch.sigmoid((u_drop * i_drop).sum(-1)))
                
                preds_stack = torch.stack(preds_list, dim=0)  # (S, B)
                mean_pred = torch.mean(preds_stack, dim=0).clamp(min=1e-4)  # (B,)
                var_pred = torch.var(preds_stack, dim=0)                   # (B,)
                std_pred = torch.sqrt(var_pred.clamp(min=1e-8))            # (B,)
            
            # Bộ lọc thận trọng CDR theo hệ số biến thiên: std / mean < eta
            cdr_filter = (std_pred / mean_pred < self.cdr_eta).float()     # (B,)
            rec_loss = (weighted_bce * cdr_filter).mean()
            return {
                "loss": rec_loss,
                "bce_loss": weighted_bce.mean().item(),
                "filtered_ratio": (1.0 - cdr_filter).mean().item()
            }

        elif self.mode == "ividr_mode":
            # --- IViDR 2024 Hook: Kết hợp mô hình cộng tính phi * L_iVAE + lambda * L_MF ---
            item_c = self.c_norm(self.item_c_proj(item_indices))
            u_c = self.c_norm(self.backbone.user_embedding(user_indices))
            c_score = torch.sum(u_c * item_c, dim=-1)
            
            # Điểm cộng tính f(u, i, C) = lam * MF + phi * C
            additive_logits = self.ividr_lam * logits + self.ividr_phi * c_score
            loss = (F.binary_cross_entropy_with_logits(additive_logits, targets, reduction="none") * confidences).mean()
            return {"loss": loss, "bce_loss": loss.item()}

        else:
            # Chế độ Baseline MF: Weighted BCE chuẩn mực
            total_loss = weighted_bce.mean()
            return {"loss": total_loss, "bce_loss": total_loss.item()}

    def predict_user_all(self, user_indices: torch.Tensor) -> torch.Tensor:
        """Dự đoán xếp hạng toàn diện trên toàn bộ M items bằng GPU matrix multiplication."""
        if self.mode == "ividr_mode":
            # Tích hợp cả MF backbone lẫn biến ẩn C
            mf_scores = self.backbone.predict_user_all(user_indices)
            u_c = self.c_norm(self.backbone.user_embedding(user_indices))
            i_c = self.c_norm(self.item_c_proj.weight)
            c_scores = torch.matmul(u_c, i_c.t())
            return self.ividr_lam * mf_scores + self.ividr_phi * c_scores
        else:
            return self.backbone.predict_user_all(user_indices)


