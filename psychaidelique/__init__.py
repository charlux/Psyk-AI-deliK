"""
Psyk-AI-deliK — RLEF Package
"""
from .metrics import (
    compute_attention_entropy,
    compute_clmi_matrix,
    compute_pev,
    compute_dts,
    compute_crs,
)
from .reward_model import RLEFRewardModel, RLEFRewardOutput, RLEFWeights, PROFILE_WEIGHTS
from .trainer import RLEFTrainingConfig, train

__all__ = [
    "RLEFRewardModel",
    "RLEFRewardOutput",
    "RLEFWeights",
    "RLEFTrainingConfig",
    "PROFILE_WEIGHTS",
    "train",
    "compute_attention_entropy",
    "compute_clmi_matrix",
    "compute_pev",
    "compute_dts",
    "compute_crs",
]
