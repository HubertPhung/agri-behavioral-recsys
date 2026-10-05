# -*- coding: utf-8 -*-
"""
Mô hình IViDR hoàn chỉnh (PyTorch nn.Module)
Tích hợp:
  1. Phân rã & tái tạo Treatment bằng biến công cụ (ridge approximation)
  2. Mạng biến phân có thể định danh (iVAE) suy diễn biến ẩn C
  3. Mô hình gợi ý cộng tính (Additive Debiased Recommendation)
Project: TopTop (agri-behavioral-recsys)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.matrix_factorization import MatrixFactorization
from src.causal.ividr.iv_reconstruction import IVTreatmentReconstruction, ItemIVTreatmentReconstruction
from src.causal.ividr.ivae import IdentifiableVAE


class IViDRModel(nn.Module):
    """
    Mô hình IViDR:
      - Backbone: Matrix Factorization (P_u, Q_i, b_u, b_i, mu)
      - IV Module: T_re = IV_Reconstruction(Q, Z)
      - iVAE Module: C = rho * C_1(X_re) + tau * C_2(X)
      - Scoring: score(u, i) = phi * <C_u, H_i> + lambda * score_MF(u, i)
    """
    def __init__(self, num_users, num_items, embedding_dim=32, iv_dim=14, proxy_dim=15,
                 latent_dim=16, phi=1.0, lam=1.0, rho=0.9, tau=0.9,
                 projection_mode="item_pinv", elbo_reduction="sum"):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.phi = phi
        self.lam = lam
        self.rho = rho
        self.tau = tau

        # 1. Backbone MF
        self.mf = MatrixFactorization(num_users, num_items, embedding_dim=embedding_dim, use_bias=True,
                                      train_global_bias=False)

        # 2. IV Treatment Reconstruction
        if projection_mode not in ("item_pinv", "global_ridge"):
            raise ValueError("Unknown IV projection mode")
        reconstruction = ItemIVTreatmentReconstruction if projection_mode == "item_pinv" else IVTreatmentReconstruction
        self.iv_recon = reconstruction(item_dim=embedding_dim, iv_dim=iv_dim, proj_dim=embedding_dim)

        # 3. iVAE Submodules: iVAE_1 trên X_re và iVAE_2 trên X
        self.ivae_1 = IdentifiableVAE(input_dim=embedding_dim, proxy_dim=proxy_dim, latent_dim=latent_dim, num_items=num_items, elbo_reduction=elbo_reduction)
        self.ivae_2 = IdentifiableVAE(input_dim=embedding_dim, proxy_dim=proxy_dim, latent_dim=latent_dim, num_items=num_items, elbo_reduction=elbo_reduction)

        # 4. Latent Confounder Item Projection: Chiếu item sang không gian C
        self.item_c_proj = nn.Embedding(num_items, latent_dim)
        nn.init.normal_(self.item_c_proj.weight, std=0.01)

        # Chuẩn hóa LayerNorm cho biểu diễn biến ẩn C để đồng bộ thang đo với MF (~1.0)
        self.c_norm = nn.LayerNorm(latent_dim)

        # Bộ đệm lưu trữ biểu diễn biến ẩn C của người dùng sau khi encode
        self.register_buffer("user_c_cache", torch.zeros(num_users, latent_dim))
        self.register_buffer("context_features", torch.empty(0, iv_dim))
        self.register_buffer("history_items", torch.empty(0, dtype=torch.long))
        self.register_buffer("history_offsets", torch.zeros(num_users + 1, dtype=torch.long))

    def set_training_context(self, user_features, exposure_matrix):
        """Register fixed train-only context and unique CSR exposure histories."""
        exposure = exposure_matrix.tocsr(copy=True)
        exposure.sum_duplicates()
        exposure.eliminate_zeros()
        if exposure.shape != (self.num_users, self.num_items):
            raise ValueError("Exposure history shape does not match model")
        device = self.mf.user_embedding.weight.device
        active = np.flatnonzero(np.diff(exposure.indptr))
        self.context_features = user_features[torch.as_tensor(active, device=user_features.device)].detach().to(device)
        self.history_items = torch.as_tensor(exposure.indices.copy(), dtype=torch.long, device=device)
        self.history_offsets = torch.as_tensor(exposure.indptr.copy(), dtype=torch.long, device=device)
        if isinstance(self.iv_recon, ItemIVTreatmentReconstruction):
            self.iv_recon.set_context(user_features, exposure)

    def aggregate_history(self, item_embeddings, user_indices):
        # Select CSR bags for unique users, without a dense user-item matrix.
        users, inverse = user_indices.unique(return_inverse=True)
        starts = self.history_offsets[users]
        lengths = self.history_offsets[users + 1] - starts
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        bag_ids = torch.repeat_interleave(torch.arange(len(users), device=users.device), lengths)
        positions = torch.arange(offsets[-1], device=users.device) - offsets[bag_ids] + starts[bag_ids]
        rows = F.embedding_bag(self.history_items[positions], item_embeddings,
                               offsets, mode="mean", include_last_offset=True)
        return rows[inverse]

    def compute_latent_confounders(self, user_indices, user_features, proxy_w,
                                   a_target=None, exposure_item_ids=None):
        """Encode each user using train-only context and their exposure history."""
        # Biểu diễn người dùng từ MF
        u_emb = self.mf.user_embedding(user_indices)

        t_re = self.iv_recon(self.mf.item_embedding.weight, self.context_features)
        x_re = u_emb + self.aggregate_history(t_re, user_indices)
        x_orig = u_emb

        c1, mu1, elbo1 = self.ivae_1(x_re, proxy_w, a_target, exposure_item_ids)
        c2, mu2, elbo2 = self.ivae_2(x_orig, proxy_w, a_target, exposure_item_ids)

        # Dung hợp biến ẩn theo Eq. (15) trong paper IViDR:
        # - Khi training: dùng c_sample (reparameterization) để tối ưu ELBO
        # - Khi eval / inference: dùng kỳ vọng tất định mu1, mu2 theo Định lý Identifiability
        if self.training:
            c_fused = self.rho * c1 + self.tau * c2
        else:
            c_fused = self.rho * mu1 + self.tau * mu2
        total_elbo = elbo1 + elbo2

        # Chuẩn hóa LayerNorm để đồng bộ thang đo với MF backbone
        c_fused = self.c_norm(c_fused)

        return c_fused, total_elbo

    def forward(self, user_indices, item_indices, user_features, proxy_w,
                a_target=None, exposure_item_ids=None):
        """
        Lan truyền tiến dự đoán điểm số và tính ELBO loss của iVAE
        """
        c_fused, elbo = self.compute_latent_confounders(
            user_indices, user_features, proxy_w,
            a_target=a_target, exposure_item_ids=exposure_item_ids,
        )

        # Cập nhật cache cho đánh giá
        with torch.no_grad():
            self.user_c_cache[user_indices] = c_fused.detach()

        # 1. Điểm số từ MF backbone
        mf_scores = self.mf(user_indices, item_indices)

        # 2. Điểm số điều chỉnh biến ẩn C
        item_c = self.item_c_proj(item_indices)
        c_scores = (c_fused * item_c).sum(dim=-1)

        # 3. Kết hợp mô hình cộng tính
        final_scores = self.lam * mf_scores + self.phi * c_scores

        return final_scores, elbo

    def update_full_user_cache(self, user_features, proxy_w, device=None):
        """Mã hóa toàn bộ người dùng để sẵn sàng cho Re-ranking Top-5 nhanh."""
        self.eval()
        target_device = device or self.mf.user_embedding.weight.device
        with torch.no_grad():
            all_u = torch.arange(self.num_users, device=target_device)
            c_fused, _ = self.compute_latent_confounders(
                all_u,
                user_features.to(target_device),
                proxy_w.to(target_device)
            )
            self.user_c_cache.copy_(c_fused)


    def get_evaluation_factors(self):
        """
        Trích xuất u_factors và i_factors tích hợp cả MF lẫn biểu diễn biến ẩn C
        để phục vụ Unbiased Top-K Re-ranking siêu tốc:
          u_factors = [ u_mf_factors, phi * user_c ]
          i_factors = [ i_mf_factors, item_c ]
        """
        with torch.no_grad():
            u_mf, i_mf = self.mf.get_evaluation_factors()
            u_c = self.user_c_cache.cpu().numpy() * self.phi
            i_c = self.item_c_proj.weight.detach().cpu().numpy()

            u_factors = np.hstack([u_mf * self.lam, u_c])
            i_factors = np.hstack([i_mf, i_c])

            return u_factors, i_factors
