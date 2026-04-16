"""
Psyk-AI-deliK — RLEF Reward Model (v2)
=======================================
Replaces the TTR-based placeholder with four empirically grounded signals.

Reward = α·PEV + β·AE_norm + γ·CLMI_norm + δ·DTS  −  λ·(1−CRS)
         ↑ divergence   ↑ entropy     ↑ cross-layer    ↑ creativity   ↓ incoherence penalty
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .metrics import compute_pev, compute_attention_entropy, compute_clmi_matrix, compute_dts, compute_crs

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class RLEFWeights:
    """Reward function coefficients. Override per psychedelic profile."""
    alpha: float = 0.35   # PEV weight   (semantic escape)
    beta:  float = 0.25   # AE weight    (attention entropy)
    gamma: float = 0.20   # CLMI weight  (cross-layer coupling)
    delta: float = 0.10   # DTS weight   (divergent thinking)
    lam:   float = 0.10   # CRS penalty  (coherence retention)


# Profile-specific weight presets — calibrated from Girn & Bzdok (2026) Table 2
PROFILE_WEIGHTS: dict[str, RLEFWeights] = {
    "sober":     RLEFWeights(alpha=0.0,  beta=0.0,  gamma=0.0,  delta=0.0,  lam=1.0),
    "psilocybin":RLEFWeights(alpha=0.35, beta=0.25, gamma=0.20, delta=0.10, lam=0.10),
    "lsd":       RLEFWeights(alpha=0.40, beta=0.30, gamma=0.15, delta=0.10, lam=0.05),
    "dmt":       RLEFWeights(alpha=0.45, beta=0.30, gamma=0.15, delta=0.05, lam=0.05),
    "mescaline": RLEFWeights(alpha=0.30, beta=0.20, gamma=0.25, delta=0.15, lam=0.10),
    "ayahuasca": RLEFWeights(alpha=0.30, beta=0.20, gamma=0.20, delta=0.20, lam=0.10),
}


@dataclass
class RLEFRewardOutput:
    reward_score:     float
    pev:              dict  = field(default_factory=dict)
    attention_entropy:dict  = field(default_factory=dict)
    clmi:             dict  = field(default_factory=dict)
    dts:              dict  = field(default_factory=dict)
    crs:              dict  = field(default_factory=dict)
    profile:          str   = "psilocybin"
    weights:          RLEFWeights = field(default_factory=RLEFWeights)

    def summary(self) -> str:
        lines = [
            f"╔══ RLEF Reward [{self.profile.upper()}] ══╗",
            f"  Reward Score  : {self.reward_score:.4f}",
            f"  PEV           : {self.pev.get('PEV', 'n/a')}",
            f"  Attention AE  : {self.attention_entropy.get('mean_AE', 'n/a')}",
            f"  CLMI          : {self.clmi.get('mean_CLMI', 'n/a')}",
            f"  DTS           : {self.dts.get('DTS', 'n/a')}",
            f"  CRS           : {self.crs.get('CRS', 'n/a')}",
            f"╚{'═'*30}╝",
        ]
        return "\n".join(lines)


class RLEFRewardModel:
    """
    Full RLEF reward signal.

    Two modes:
      - inference_only=True  → only PEV + DTS (no model internals needed, Ollama-compatible)
      - inference_only=False → all 5 metrics (requires HF model with output_attentions=True)
    """

    def __init__(
        self,
        hf_model: Optional["PreTrainedModel"] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        inference_only: bool = True,
        baseline_perplexity: Optional[float] = None,
    ):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.inference_only = inference_only or (hf_model is None)
        self.baseline_perplexity = baseline_perplexity

    # ------------------------------------------------------------------
    def calculate_reward(
        self,
        original_prompt: str,
        psychedelic_text: str,
        baseline_text: str,
        profile: str = "psilocybin",
        sentences_for_dts: Optional[list[str]] = None,
        attentions: Optional[tuple] = None,
        hidden_states: Optional[tuple] = None,
    ) -> RLEFRewardOutput:
        """
        Args:
            original_prompt:     The user's input prompt.
            psychedelic_text:    Output under RLEF modulation.
            baseline_text:       Output of same prompt at T=1.0 (sober).
            profile:             One of PROFILE_WEIGHTS keys.
            sentences_for_dts:   Pre-split list of ideas/sentences for DTS.
                                 If None, auto-splits psychedelic_text.
            attentions:          HF model attentions tuple (optional).
            hidden_states:       HF model hidden states tuple (optional).
        """
        W = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["psilocybin"])

        # 1. PEV — always available
        pev_result = compute_pev(psychedelic_text, baseline_text)

        # 2. AE — needs model internals
        ae_result = {}
        if not self.inference_only and attentions is not None:
            ae_result = compute_attention_entropy(attentions)

        # 3. CLMI — needs hidden states
        clmi_result = {}
        if not self.inference_only and hidden_states is not None:
            clmi_result = compute_clmi_matrix(hidden_states)

        # 4. DTS — always available
        if sentences_for_dts is None:
            sentences_for_dts = [s.strip() for s in psychedelic_text.split(".") if len(s.strip()) > 10]
        dts_result = compute_dts(sentences_for_dts)

        # 5. CRS — needs HF model for perplexity
        crs_result = {}
        if not self.inference_only and self.hf_model is not None:
            crs_result = compute_crs(
                psychedelic_text,
                self.hf_model,
                self.tokenizer,
                baseline_perplexity=self.baseline_perplexity,
            )

        # --- Compose reward ---
        pev_score  = pev_result.get("PEV", 0.0)
        ae_score   = min(1.0, ae_result.get("mean_AE", 0.0) / 3.0)  # normalise [0,3]→[0,1]
        clmi_score = min(1.0, clmi_result.get("mean_CLMI", 0.0) / 2.0)  # normalise
        dts_score  = dts_result.get("DTS", 0.0)
        crs_score  = crs_result.get("CRS", 1.0)  # default 1.0 if not computed

        reward = (
            W.alpha * pev_score
            + W.beta  * ae_score
            + W.gamma * clmi_score
            + W.delta * dts_score
            - W.lam   * (1.0 - crs_score)  # penalty for incoherence
        )

        return RLEFRewardOutput(
            reward_score=round(float(reward), 4),
            pev=pev_result,
            attention_entropy=ae_result,
            clmi=clmi_result,
            dts=dts_result,
            crs=crs_result,
            profile=profile,
            weights=W,
        )


if __name__ == "__main__":
    rm = RLEFRewardModel(inference_only=True)

    psychedelic = (
        "La propriété est un fantasme collectif tissé par des architectures de contrainte "
        "invisibles — le cadastre comme hallucination partagée, la frontière comme "
        "délire bureaucratique cristallisé dans le béton."
    )
    baseline = (
        "La propriété est un droit légal qui permet à une personne de posséder un bien "
        "et d'en disposer comme elle le souhaite dans le cadre de la loi."
    )

    result = rm.calculate_reward(
        original_prompt="Explique la propriété",
        psychedelic_text=psychedelic,
        baseline_text=baseline,
        profile="dmt",
    )
    print(result.summary())
