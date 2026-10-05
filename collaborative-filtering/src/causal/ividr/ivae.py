# -*- coding: utf-8 -*-
"""
Mô hình Identifiable Variational Auto-Encoder (iVAE) trong IViDR
Mô hình hóa các biến ẩn gây nhiễu C có thể định danh toán học dựa trên biến đại diện W
Project: TopTop (agri-behavioral-recsys)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentifiableVAE(nn.Module):
    """
    Identifiable VAE (iVAE):
      - Tiên nghiệm điều kiện p_theta(C | W): Gaussian N(mu_w(W), diag(sigma_w^2(W)))
      - Hậu nghiệm biến phân q_phi(C | X, A, W): Gaussian N(mu_xaw, diag(sigma_xaw^2))
      - Giải mã vector phơi bày A: p_theta(A | C) ~ Bernoulli(sigmoid(MLP_dec(C)))
    """
    def __init__(self, input_dim, proxy_dim, latent_dim=16, num_items=300, elbo_reduction="sum"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_items = num_items
        if elbo_reduction not in ("sum", "legacy_mean"):
            raise ValueError("Unknown ELBO reduction")
        self.elbo_reduction = elbo_reduction

        # Mạng tiên nghiệm Prior Net: W -> mu_w, logvar_w
        self.prior_net = nn.Sequential(
            nn.Linear(proxy_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim * 2)
        )

        # Mạng hậu nghiệm Encoder: [X, W] -> mu_post, logvar_post
        enc_in_dim = input_dim + proxy_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim * 2)
        )

        # Mạng giải mã phơi bày Decoder: C -> logits của A (kích thước num_items)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
        )
        self.decoder_output = nn.Linear(64, num_items)

    def encode(self, x, w):
        """Xấp xỉ hậu nghiệm q(C | X, W)"""
        inputs = torch.cat([x, w], dim=-1)
        params = self.encoder(inputs)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        # Giới hạn logvar trong [-4.0, 4.0] để tránh sụp đổ phương sai và nổ gradient
        logvar = torch.clamp(logvar, min=-4.0, max=4.0)
        return mu, logvar

    def get_prior(self, w):
        """Tiên nghiệm điều kiện p(C | W)"""
        params = self.prior_net(w)
        mu_p, logvar_p = torch.chunk(params, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, min=-4.0, max=4.0)
        return mu_p, logvar_p

    def reparameterize(self, mu, logvar):
        """Kỹ thuật Reparameterization Trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, w, a_target=None, exposure_item_ids=None):
        """
        x: Biểu diễn tương tác người dùng (N, input_dim)
        w: Biến đại diện proxy (N, proxy_dim)
        a_target: Exposure labels for all items or sampled item IDs.
        exposure_item_ids: Optional sampled item IDs (N, K) for large catalogs.
        """
        mu_q, logvar_q = self.encode(x, w)
        c_sample = self.reparameterize(mu_q, logvar_q) if self.training else mu_q

        mu_p, logvar_p = self.get_prior(w)

        # KL Divergence giữa 2 phân phối Gaussian: KL(q || p)
        # KL = 0.5 * sum [ logvar_p - logvar_q + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p) - 1 ]
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p).clamp(min=1e-3)
        # Chuẩn hóa KL theo latent_dim để cùng thang đo với BCE loss (~0.6)
        kl = 0.5 * torch.mean(logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0, dim=-1)
        if self.elbo_reduction == "sum":
            kl = kl * self.latent_dim

        # Tái tạo phơi bày (chỉ tính toán khi có a_target để tránh lãng phí tài nguyên CPU/Memory)
        if a_target is not None:
            hidden = self.decoder_hidden(c_sample)
            if exposure_item_ids is None:
                a_logits = self.decoder_output(hidden)
            else:
                weights = self.decoder_output.weight[exposure_item_ids]
                biases = self.decoder_output.bias[exposure_item_ids]
                a_logits = (hidden.unsqueeze(1) * weights).sum(dim=-1) + biases
            if a_logits.shape != a_target.shape:
                raise ValueError("Exposure targets must match decoder logits")
            recon_loss = F.binary_cross_entropy_with_logits(a_logits, a_target, reduction="none").mean(dim=-1)
            if self.elbo_reduction == "sum":
                # Uniform item sampling estimates the full Bernoulli log likelihood
                # with N/K times the sample sum, not an unscaled sample mean.
                recon_loss = recon_loss * self.num_items
            elbo = recon_loss + kl
        else:
            elbo = kl

        return c_sample, mu_q, elbo.mean()
