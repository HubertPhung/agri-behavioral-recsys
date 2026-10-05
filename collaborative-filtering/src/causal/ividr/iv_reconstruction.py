# -*- coding: utf-8 -*-
"""
Module Tái Tạo Treatment bằng Biến Công Cụ (IV Treatment Reconstruction) trong IViDR
Xấp xỉ ridge khả vi trong không gian đặc trưng (không phải phép chiếu trực giao)
và cơ chế cổng học động (Dynamic Gating)
Project: TopTop (agri-behavioral-recsys)
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class IVTreatmentReconstruction(nn.Module):
    """Ridge fitted/residual components with learned dynamic gates.

    Uses fixed biased-train user context. This is a local IViDR variant,
    not an exact implementation of the paper's 2SLS estimator.
    """
    def __init__(self, item_dim, iv_dim, proj_dim=32):
        super().__init__()
        self.item_dim = item_dim
        self.iv_dim = iv_dim
        self.proj_dim = proj_dim

        # MLP_0: Chuyển đổi embedding vật phẩm sang không gian phép chiếu d_q
        self.mlp_0 = nn.Sequential(
            nn.Linear(item_dim, proj_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim, proj_dim)
        )

        # Mạng tuyến tính ánh xạ IV Z sang không gian d_q để xây dựng toán tử chiếu
        self.iv_proj = nn.Linear(iv_dim, proj_dim, bias=False)

        # MLP_1 và MLP_2: Ước lượng trọng số kết hợp alpha_1, alpha_2
        concat_dim = proj_dim + iv_dim
        self.mlp_1 = nn.Sequential(
            nn.Linear(concat_dim, proj_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim // 2, 1),
            nn.Sigmoid()
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(concat_dim, proj_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim // 2, 1),
            nn.Sigmoid()
        )

    def compute_projection_matrix(self, iv_features):
        """Differentiable ridge fit in feature space (a local approximation)."""
        if iv_features.shape[0] == 0:
            raise ValueError("IV reconstruction requires biased-train context")
        z = self.iv_proj(iv_features)
        # Double precision keeps the solve stable for rank-deficient feature matrices.
        g = z.double().T @ z.double() / z.shape[0]
        eye = torch.eye(self.proj_dim, device=g.device, dtype=g.dtype)
        return torch.linalg.solve(g + 1e-4 * eye, g).to(z.dtype)

    def forward(self, item_embeddings, iv_features):
        """
        item_embeddings: (M, item_dim)
        iv_features: (N, iv_dim)
        Trích xuất: T_re có kích thước (M, proj_dim)
        """
        # 1. Chuyển đổi qua MLP_0
        T_proj = self.mlp_0(item_embeddings)  # (M, proj_dim)

        # 2. Differentiable ridge fitted component
        P_Z = self.compute_projection_matrix(iv_features)  # (proj_dim, proj_dim)
        
        # 3. Phân rã fitted part và residual part
        T_hat = torch.matmul(T_proj, P_Z)                  # (M, proj_dim)
        T_tilde = T_proj - T_hat                           # (M, proj_dim)

        # 4. Dynamic gating theo phương trình (9): item context kết hợp phân phối IV
        z_mean = iv_features.mean(dim=0, keepdim=True)         # (1, iv_dim)
        z_expand = z_mean.expand(item_embeddings.size(0), -1)  # (M, iv_dim)
        gate_input = torch.cat([T_proj, z_expand], dim=-1)     # (M, proj_dim + iv_dim)
        
        alpha_1 = self.mlp_1(gate_input)                       # (M, 1)
        alpha_2 = self.mlp_2(gate_input)                       # (M, 1)

        # 5. Tái tạo Treatment theo phương trình (8)
        T_re = alpha_1 * T_hat + alpha_2 * T_tilde             # (M, proj_dim)
        return T_re


class ItemIVTreatmentReconstruction(IVTreatmentReconstruction):
    """Eq. 6 projection onto each item's train-user feature span.

    Raw feature bases are computed once, without gradients. The learned linear
    feature embedding and its pseudoinverse remain differentiable. Full-span
    items share one basis, avoiding thousands of identical SVDs per minibatch.
    Mean pooling for the variable-length gate input is an explicit local choice.
    """
    def __init__(self, item_dim, iv_dim, proj_dim=32):
        super().__init__(item_dim, iv_dim, proj_dim)
        self.register_buffer("bases", torch.empty(0, iv_dim, iv_dim, dtype=torch.float64))
        self.register_buffer("ranks", torch.empty(0, dtype=torch.long))
        self.register_buffer("owners", torch.empty(0, dtype=torch.long))
        self.register_buffer("means", torch.empty(0, iv_dim))

    def set_context(self, features, exposure):
        values = features.detach().cpu().double().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite IV features")
        by_item = exposure.tocsc(copy=True)
        by_item.sum_duplicates()
        by_item.eliminate_zeros()
        if by_item.shape[0] != len(values) or values.shape[1] != self.iv_dim:
            raise ValueError("IV feature/context shape mismatch")

        def basis(rows):
            if len(rows) == 0:
                return np.zeros((self.iv_dim, self.iv_dim)), 0
            _, s, vt = np.linalg.svd(rows, full_matrices=False)
            rank = int((s > (s[0] * 1e-10 if len(s) else 0)).sum())
            padded = np.zeros((self.iv_dim, self.iv_dim))
            padded[:, :rank] = vt[:rank].T
            return padded, rank

        active = np.flatnonzero(np.diff(exposure.tocsr().indptr))
        shared, shared_rank = basis(values[active])
        bases, ranks, owners, means = [shared], [shared_rank], [], []
        for item in range(by_item.shape[1]):
            users = by_item.indices[by_item.indptr[item]:by_item.indptr[item + 1]]
            rows = values[users]
            current, rank = basis(rows)
            means.append(rows.mean(0) if len(rows) else np.zeros(self.iv_dim))
            if rank == shared_rank:
                owners.append(0)
            else:
                owners.append(len(bases))
                bases.append(current)
                ranks.append(rank)
        device = self.iv_proj.weight.device
        self.bases = torch.as_tensor(np.asarray(bases), dtype=torch.float64, device=device)
        self.ranks = torch.tensor(ranks, dtype=torch.long, device=device)
        self.owners = torch.tensor(owners, dtype=torch.long, device=device)
        self.means = torch.as_tensor(np.asarray(means), dtype=features.dtype, device=device)

    def fitted(self, transformed):
        if len(self.owners) != len(transformed):
            raise ValueError("Set item-specific training context before IV reconstruction")
        fitted = torch.zeros_like(transformed)
        # Project with a compact basis of Z_j: Z_j pinv(Z_j) t. Never form an
        # item-by-user matrix or use a ridge term that changes the estimator.
        for rank in self.ranks.unique().tolist():
            if rank == 0:
                continue
            basis_ids = torch.nonzero(self.ranks == rank).flatten()
            z = self.iv_proj.weight.double() @ self.bases[basis_ids, :, :rank]
            inverse = torch.linalg.pinv(z, rtol=1e-10)
            if len(basis_ids) == 1:
                selected = self.owners == basis_ids[0]
                fitted[selected] = (transformed[selected].double() @ inverse[0].T @ z[0].T).to(transformed.dtype)
                continue
            for start in range(0, len(basis_ids), 256):
                ids = basis_ids[start:start + 256]
                matches = self.owners[:, None] == ids[None, :]
                item_ids, local_ids = torch.nonzero(matches, as_tuple=True)
                if len(item_ids):
                    t = transformed[item_ids].double().unsqueeze(-1)
                    fitted[item_ids] = (z[start:start + 256][local_ids] @
                                       (inverse[start:start + 256][local_ids] @ t)).squeeze(-1).to(transformed.dtype)
        return fitted

    def forward(self, item_embeddings, iv_features=None):
        transformed = self.mlp_0(item_embeddings)
        fitted = self.fitted(transformed)
        gate_input = torch.cat([transformed, self.means], dim=-1)
        return self.mlp_1(gate_input) * fitted + self.mlp_2(gate_input) * (transformed - fitted)


