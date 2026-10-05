# -*- coding: utf-8 -*-
"""
Tiện ích hệ thống: Cấu hình thiết bị, tái lập Seed, định vị thư mục chuẩn
Project: TopTop (agri-behavioral-recsys)
"""

import os
import random
import numpy as np
import torch

# Đường dẫn gốc chuẩn của dự án:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Tìm DATA_DIR: ưu tiên tìm trong PROJECT_ROOT/data, nếu không có thì tìm ở thư mục cha ../data
candidate_data = os.path.join(PROJECT_ROOT, "data")
if os.path.exists(candidate_data):
    DATA_DIR = candidate_data
elif os.path.exists(os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data"))):
    DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data"))
else:
    DATA_DIR = candidate_data

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=42):
    """Thiết lập seed cho random, numpy và pytorch nhằm bảo đảm khả năng tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str=None):
    """Lấy device (CUDA nếu có hoặc CPU)."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def resolve_dataset_dir(dataset, data_dir=None):
    """Resolve once; an explicit directory is never replaced by another dataset."""
    if data_dir is not None:
        root = os.path.abspath(os.fspath(data_dir))
    else:
        root = os.path.join(RAW_DATA_DIR, dataset)
        if dataset == "kuairand_pure" and os.path.isdir(os.path.join(root, "data")):
            root = os.path.join(root, "data")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root
