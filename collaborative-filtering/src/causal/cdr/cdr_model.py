# -*- coding: utf-8 -*-
"""
Kiến trúc mô hình tổng thể DR-BIAS + CDR
Bao gồm Recommendation Model f_Theta và Imputation Model g_Psi
Biến thể địa phương tham khảo CDR CIKM 2023 (Song et al.)
Project: TopTop (agri-behavioral-recsys)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.matrix_factorization import MatrixFactorization
from src.causal.cdr.imputation_model import ImputationModel
from src.causal.cdr.dr_bias_loss import (
    dr_bias_closed_form_imputation_loss,
    cdr_recommendation_loss
)


class CDRFramework(nn.Module):
    """
    Khung làm việc Conservative Doubly Robust (CDR):
      - f_Theta: Mô hình gợi ý chính Matrix Factorization
      - g_Psi: Mô hình gán nhãn sai số Imputation Model với Monte Carlo Dropout
    """
    def __init__(self, num_users, num_items, embedding_dim=4, un_thres=10.0, dropout_rate=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.un_thres = un_thres

        # 1. Recommendation Model f_Theta
        self.rec_model = MatrixFactorization(num_users, num_items, embedding_dim=embedding_dim, use_bias=True)

        # 2. Imputation Model g_Psi
        self.imp_model = ImputationModel(num_users, num_items, embedding_dim=embedding_dim, dropout_rate=dropout_rate)

    def forward(self, user_idx, item_idx):
        """Dự đoán logits của mô hình gợi ý chính"""
        return self.rec_model(user_idx, item_idx)

    def compute_cdr_losses(self, u_obs, i_obs, r_obs, inv_p_obs,
                           u_unobs, i_unobs, mc_samples=10,
                           counterfactual_scale=1.0):
        """
        Tính toán 2 hàm mất mát chuẩn:
          1. L_rec: Huấn luyện Recommendation Model theo CDR (loại bỏ poisonous imputation)
          2. L_imp: Huấn luyện Imputation Model theo DR-BIAS closed form
        """
        # --- BƯỚC 1: TÍNH CDR LOSS CHO RECOMMENDATION MODEL ---
        # Dự đoán trên tập quan sát
        pred_obs = torch.sigmoid(self.rec_model(u_obs, i_obs))
        imp_samples_obs = self.imp_model.sample_predict(u_obs, i_obs, num_samples=mc_samples)
        imp_loss_obs, imp_unc_obs = self.imp_model.sample_bce(pred_obs, imp_samples_obs)

        # Dự đoán trên tập unobserved/counterfactuals
        pred_unobs = torch.sigmoid(self.rec_model(u_unobs, i_unobs))
        imp_samples_unobs = self.imp_model.sample_predict(u_unobs, i_unobs, num_samples=mc_samples)
        imp_loss_unobs, imp_unc_unobs = self.imp_model.sample_bce(pred_unobs, imp_samples_unobs)

        loss_rec, filtered_ratio = cdr_recommendation_loss(
            pred_obs=pred_obs,
            y_obs=r_obs,
            inv_prop_obs=inv_p_obs,
            imp_loss_obs=imp_loss_obs,
            imp_unc_obs=imp_unc_obs,
            pred_unobs=pred_unobs,
            imp_loss_unobs=imp_loss_unobs,
            imp_unc_unobs=imp_unc_unobs,
            un_thres=self.un_thres,
            counterfactual_scale=counterfactual_scale,
        )

        # --- BƯỚC 2: TÍNH DR-BIAS LOSS CHO IMPUTATION MODEL ---
        # Imputation model g_psi dự đoán nhãn giả mạo sao cho e_hat mô phỏng sai số của rec model
        # Theo chuẩn CIKM '23 (Song et al.) & DR-BIAS: e_hat_loss = BCE(pred_imp, pred_rec_fixed)
        with torch.no_grad():
            pred_rec_fixed = torch.sigmoid(self.rec_model(u_obs, i_obs))
        pred_imp = torch.sigmoid(self.imp_model(u_obs, i_obs))

        e_loss = F.binary_cross_entropy(pred_rec_fixed, r_obs, reduction="none")
        e_hat_loss = F.binary_cross_entropy(pred_imp, pred_rec_fixed, reduction="none")
        loss_imp = dr_bias_closed_form_imputation_loss(e_loss, e_hat_loss, inv_p_obs)

        return loss_rec, loss_imp, filtered_ratio


    def get_evaluation_factors(self):
        """Trích xuất factors của Recommendation Model phục vụ Unbiased Re-ranking."""
        return self.rec_model.get_evaluation_factors()
