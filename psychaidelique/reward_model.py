"""
Psyk-AI-deliK — RLEF Reward Model (v2.1 - Lazy Loading)
=======================================================
Optimisé pour Manjaro OneTwo (Validation) & Apple M4 (Inférence).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

# On n'importe PAS torch ici globalement pour éviter de casser le bastion Manjaro
from .metrics import compute_pev, compute_attention_entropy, compute_clmi_matrix, compute_dts, compute_crs

if TYPE_CHECKING:
    import torch
    import numpy as np
    from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class RLEFWeights:
    """Coefficients de la fonction de récompense."""
    alpha: float = 0.35   # Poids PEV (évasion sémantique)
    beta:  float = 0.25   # Poids AE (entropie d'attention)
    gamma: float = 0.20   # Poids CLMI (couplage inter-couches)
    delta: float = 0.10   # Poids DTS (pensée divergente)
    lam:   float = 0.10   # Pénalité CRS (rétention de cohérence)

# Calibrations basées sur Girn & Bzdok (2026)
PROFILE_WEIGHTS: dict[str, RLEFWeights] = {
    "sober":     RLEFWeights(alpha=0.0,  beta=0.0,  gamma=0.0,  delta=0.0,  lam=1.0),
    "psilocybin":RLEFWeights(alpha=0.35, beta=0.25, gamma=0.20, delta=0.10, lam=0.10),
    "lsd":       RLEFWeights(alpha=0.40, beta=0.30, gamma=0.15, delta=0.10, lam=0.05),
    "dmt":       RLEFWeights(alpha=0.45, beta=0.30, gamma=0.15, delta=0.05, lam=0.05),
    "ayahuasca": RLEFWeights(alpha=0.30, beta=0.20, gamma=0.20, delta=0.20, lam=0.10),
}

@dataclass
class RLEFRewardOutput:
    reward_score:      float
    pev:               dict  = field(default_factory=dict)
    attention_entropy: dict  = field(default_factory=dict)
    clmi:              dict  = field(default_factory=dict)
    dts:               dict  = field(default_factory=dict)
    crs:               dict  = field(default_factory=dict)
    profile:           str   = "psilocybin"
    weights:           RLEFWeights = field(default_factory=RLEFWeights)

    def summary(self) -> str:
        return (
            f"╔══ RLEF Reward [{self.profile.upper()}] ══╗\n"
            f"  Reward Score  : {self.reward_score:.4f}\n"
            f"  PEV           : {self.pev.get('PEV', 'n/a')}\n"
            f"  Attention AE  : {self.attention_entropy.get('mean_AE', 'n/a')}\n"
            f"  DTS           : {self.dts.get('DTS', 'n/a')}\n"
            f"╚{'═'*30}╝"
        )

class RLEFRewardModel:
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
        
        # Imports locaux pour éviter de bloquer les machines sans GPU/Torch
        import numpy as np 

        W = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["psilocybin"])

        # 1. PEV (Toujours dispo - String matching / Embeddings légers)
        pev_result = compute_pev(psychedelic_text, baseline_text)

        # 2. & 3. AE / CLMI (Nécessitent Torch et les internes du modèle)
        ae_result, clmi_result = {}, {}
        if not self.inference_only:
            import torch # Chargement uniquement si nécessaire
            if attentions is not None:
                ae_result = compute_attention_entropy(attentions)
            if hidden_states is not None:
                clmi_result = compute_clmi_matrix(hidden_states)

        # 4. DTS (Analyse sémantique divergente)
        if sentences_for_dts is None:
            sentences_for_dts = [s.strip() for s in psychedelic_text.split(".") if len(s.strip()) > 10]
        dts_result = compute_dts(sentences_for_dts)

        # 5. CRS (Cohérence - Nécessite le modèle HF)
        crs_result = {}
        if not self.inference_only and self.hf_model is not None:
            crs_result = compute_crs(
                psychedelic_text,
                self.hf_model,
                self.tokenizer,
                baseline_perplexity=self.baseline_perplexity,
            )

        # --- Calcul des scores finaux ---
        pev_score  = pev_result.get("PEV", 0.0)
        ae_score   = min(1.0, ae_result.get("mean_AE", 0.0) / 3.0)
        clmi_score = min(1.0, clmi_result.get("mean_CLMI", 0.0) / 2.0)
        dts_score  = dts_result.get("DTS", 0.0)
        crs_score  = crs_result.get("CRS", 1.0)

        reward = (
            W.alpha * pev_score
            + W.beta  * ae_score
            + W.gamma * clmi_score
            + W.delta * dts_score
            - W.lam   * (1.0 - crs_score)
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
    # Test rapide en mode inference_only (compatible OneTwo)
    rm = RLEFRewardModel(inference_only=True)
    res = rm.calculate_reward("Test", "L'hallucination est une vérité temporaire.", "L'hallucination est une erreur de perception.", profile="dmt")
    print(res.summary())
