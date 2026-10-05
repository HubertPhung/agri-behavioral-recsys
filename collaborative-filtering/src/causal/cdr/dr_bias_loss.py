# -*- coding: utf-8 -*-
"""
Hàm mất mát DR-BIAS và Bộ lọc Thận trọng CDR (Conservative Doubly Robust)
Biến thể địa phương tham khảo CDR (Song et al.) và DR-BIAS (Dai et al.)
Project: TopTop (agri-behavioral-recsys)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dr_bias_closed_form_imputation_loss(e_loss, e_hat_loss, inv_prop):
    """
    Hàm mất mát huấn luyện mô hình Imputation Model theo DR-BIAS (Dai et al., 2022; Song et al., CIKM 2023):
      L_imp = sum_{O} (e_ui - e_hat_ui)^2 * (inv_p)^3 * (1 - 1/inv_p)^2
    giúp tối ưu hóa cân bằng giữa bias và variance của bộ gán nhãn sai số.
    """
    weight = (inv_prop ** 3) * ((1.0 - 1.0 / torch.clamp(inv_prop, min=1.0)) ** 2)
    imp_loss = (((e_loss - e_hat_loss) ** 2) * weight).mean()
    return imp_loss


def cdr_recommendation_loss(pred_obs, y_obs, inv_prop_obs, imp_loss_obs, imp_unc_obs,
                            pred_unobs, imp_loss_unobs, imp_unc_unobs,
                            un_thres=10.0, counterfactual_scale=1.0):
    """
    Hàm mất mát Conservative Doubly Robust (CDR) cập nhật mô hình gợi ý chính:
      L_CDR = (IPS_part + Direct_part) / |x_unobs|
    trong đó:
      - IPS_part = sum_{O} [ inv_p * e_ui - (inv_p - 1) * e_hat_ui * I(sigma / mu < un_thres) ]
      - Direct_part = sum_{D} e_hat_ui * I(sigma / mu < un_thres)
      - Bộ lọc CDR loại bỏ poisonous imputation theo hệ số biến thiên (Coefficient of Variation)
        dùng trong biến thể địa phương: sigma / mu < eta
    """
    # 1. Thành phần trên tập quan sát (IPS có hiệu chỉnh imputation đã lọc)
    xent_loss = (F.binary_cross_entropy(pred_obs, y_obs, reduction="none") * inv_prop_obs).sum()
    
    # Lọc thận trọng theo tỷ số biến thiên sigma / mu < un_thres
    std_obs = torch.sqrt(imp_unc_obs.clamp(min=1e-8))
    filter_obs = ((std_obs / imp_loss_obs.detach().clamp(min=1e-4)) < un_thres).float()
    
    # Số hạng Doubly Robust chuẩn: trừ đi (inv_prop - 1) * e_hat
    prop_weight_obs = torch.clamp(inv_prop_obs - 1.0, min=0.0)
    imputation_loss_obs = (imp_loss_obs * filter_obs * prop_weight_obs).sum()
    ips_loss = xent_loss - imputation_loss_obs

    # 2. Thành phần trên tập đối chứng / chưa quan sát (Direct Imputation đã lọc)
    std_unobs = torch.sqrt(imp_unc_unobs.clamp(min=1e-8))
    filter_unobs = ((std_unobs / imp_loss_unobs.detach().clamp(min=1e-4)) < un_thres).float()
    direct_loss = (imp_loss_unobs * filter_unobs).sum() * counterfactual_scale

    num_samples = max(pred_unobs.shape[0], 1)
    loss = (ips_loss + direct_loss) / num_samples

    total_filtered = (1.0 - filter_unobs).sum().item() + (1.0 - filter_obs).sum().item()
    total_items = filter_unobs.numel() + filter_obs.numel()
    filtered_ratio = total_filtered / max(total_items, 1)

    return loss, filtered_ratio

