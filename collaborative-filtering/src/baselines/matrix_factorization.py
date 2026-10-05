# -*- coding: utf-8 -*-
"""
Matrix Factorization module (Forwarded to src.models.matrix_factorization)
Duy trì backward compatibility cho các module causal debiasing (CDR, IViDR) và baselines.
Single Source of Truth: src/models/matrix_factorization.py
Project: TopTop (agri-behavioral-recsys)
"""

from src.models.matrix_factorization import MatrixFactorization

__all__ = ["MatrixFactorization"]


