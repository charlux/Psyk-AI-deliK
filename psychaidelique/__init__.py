"""
Psyk-AI-deliK — RLEF Package

Keep package import lightweight: optional ML/training dependencies are loaded
only when their functionality is explicitly requested.
"""
from .metrics import (
    compute_attention_entropy,
    compute_clmi_matrix,
    compute_pev,
    compute_dts,
    compute_crs,
)
from .reward_model import RLEFRewardModel, RLEFRewardOutput, RLEFWeights, PROFILE_WEIGHTS

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


def __getattr__(name: str):
    """Load training symbols lazily so logic-only tests do not require PyTorch."""
    if name in {"RLEFTrainingConfig", "train"}:
        from .trainer import RLEFTrainingConfig, train
        return {"RLEFTrainingConfig": RLEFTrainingConfig, "train": train}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
