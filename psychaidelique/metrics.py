"""
Psyk-AI-deliK — RLEF Metrics
=============================
Four empirically grounded metrics replacing the TTR-based placeholder.

References:
  - Girn, Bzdok et al. (2026) — Neural fingerprint of psychedelics. Nature Medicine.
  - Carhart-Harris & Friston (2019) — REBUS. Pharmacological Reviews.
  - Reimers & Gurevych (2019) — Sentence-BERT. EMNLP.
"""

import math
import torch
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# 1. ATTENTION ENTROPY (AE)
#    Mean Shannon entropy of attention distributions across all heads/layers.
#    Requires a HuggingFace model with output_attentions=True.
#    Biological analogue: ↑ between-network connectivity under psychedelics.
# ---------------------------------------------------------------------------

def compute_attention_entropy(
    attentions: tuple[torch.Tensor, ...],
    layer_group: Optional[str] = "all",
) -> dict:
    """
    Args:
        attentions: tuple of tensors (n_layers,) each [batch, heads, seq, seq]
        layer_group: "early" (0–33%), "mid" (33–66%), "late" (66–100%), or "all"
    Returns:
        dict with per-layer and mean entropy
    """
    n_layers = len(attentions)
    slices = {
        "early": range(0, n_layers // 3),
        "mid":   range(n_layers // 3, 2 * n_layers // 3),
        "late":  range(2 * n_layers // 3, n_layers),
        "all":   range(n_layers),
    }
    target_layers = slices.get(layer_group, range(n_layers))

    layer_entropies = []
    for i in target_layers:
        attn = attentions[i]  # [batch, heads, seq, seq]
        # Clamp for numerical stability before log
        attn = attn.clamp(min=1e-9)
        entropy = -(attn * attn.log()).sum(dim=-1)  # [batch, heads, seq]
        layer_entropies.append(entropy.mean().item())

    return {
        "per_layer": layer_entropies,
        "mean_AE": float(np.mean(layer_entropies)),
        "std_AE":  float(np.std(layer_entropies)),
    }


# ---------------------------------------------------------------------------
# 2. CROSS-LAYER MUTUAL INFORMATION (CLMI)
#    Estimates MI between hidden states of distant layers via binned histogram.
#    Biological analogue: hierarchy flattening (REBUS model).
# ---------------------------------------------------------------------------

def compute_clmi(
    hidden_states: tuple[torch.Tensor, ...],
    layer_a: int,
    layer_b: int,
    n_bins: int = 32,
) -> float:
    """
    Args:
        hidden_states: tuple (n_layers+1,) each [batch, seq, hidden]
        layer_a, layer_b: indices of the two layers to compare
        n_bins: histogram resolution
    Returns:
        Estimated mutual information in nats
    """
    def _marginal_entropy(x: np.ndarray, bins: int) -> float:
        counts, _ = np.histogram(x, bins=bins, density=False)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    h_a = hidden_states[layer_a][0].mean(dim=0).detach().cpu().numpy()  # [hidden]
    h_b = hidden_states[layer_b][0].mean(dim=0).detach().cpu().numpy()

    # Project to 1D via first principal direction (cheap proxy)
    pca_a = h_a - h_a.mean()
    pca_b = h_b - h_b.mean()

    H_a  = _marginal_entropy(pca_a, n_bins)
    H_b  = _marginal_entropy(pca_b, n_bins)
    H_ab = _marginal_entropy(np.stack([pca_a, pca_b], axis=1).ravel(), n_bins * 2)

    mi = max(0.0, H_a + H_b - H_ab)
    return float(mi)


def compute_clmi_matrix(
    hidden_states: tuple[torch.Tensor, ...],
    layer_pairs: Optional[list[tuple[int, int]]] = None,
) -> dict:
    """Compute CLMI for a set of layer pairs (defaults to early↔late)."""
    n = len(hidden_states)
    if layer_pairs is None:
        # Early ↔ late pairs as in Girn et al. cross-network connectivity
        layer_pairs = [
            (0, n - 1),
            (0, n // 2),
            (n // 4, 3 * n // 4),
        ]
    results = {}
    for a, b in layer_pairs:
        results[f"L{a}↔L{b}"] = compute_clmi(hidden_states, a, b)
    results["mean_CLMI"] = float(np.mean(list(results.values())))
    return results


# ---------------------------------------------------------------------------
# 3. PATERNALISM ESCAPE VELOCITY (PEV)
#    Cosine distance between embeddings of psychedelic output vs sober baseline.
#    Uses sentence-transformers (all-MiniLM-L6-v2, ~80 MB).
#    Biological analogue: semantic divergence from default-mode suppression.
# ---------------------------------------------------------------------------

_sentence_model = None

def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def compute_pev(
    psychedelic_text: str,
    baseline_text: str,
) -> dict:
    """
    Args:
        psychedelic_text: output under RLEF modulation
        baseline_text: output of the same prompt under sober (T=1.0) inference
    Returns:
        dict with cosine_distance (0=identical, 2=opposite) and PEV score [0,1]
    """
    model = _get_sentence_model()
    emb = model.encode([psychedelic_text, baseline_text], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0)).item()
    cos_dist = 1.0 - cos_sim  # [0, 2], but practically [0, ~0.8] for natural text
    # Normalise to [0, 1] assuming max meaningful distance ≈ 0.8
    pev = min(1.0, cos_dist / 0.8)
    return {
        "cosine_distance": round(cos_dist, 4),
        "PEV": round(pev, 4),
        "raw_cosine_similarity": round(cos_sim, 4),
    }


# ---------------------------------------------------------------------------
# 4. DIVERGENT THINKING SCORE (DTS)
#    Algorithmic Alternative Uses Test: semantic spread of generated uses/ideas.
#    Biological analogue: psychedelic ↑ in divergent thinking (Prochazkova 2018).
# ---------------------------------------------------------------------------

def compute_dts(sentences: list[str]) -> dict:
    """
    Args:
        sentences: list of distinct ideas/uses generated by the model
    Returns:
        dict with mean pairwise distance and DTS score
    """
    if len(sentences) < 2:
        return {"mean_pairwise_distance": 0.0, "DTS": 0.0, "n_ideas": len(sentences)}

    model = _get_sentence_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)  # [n, dim]

    n = len(embeddings)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
            ).item()
            distances.append(1.0 - cos)

    mean_dist = float(np.mean(distances))
    # Normalise: mean distance of 0.5 → DTS ≈ 1.0 (high divergence)
    dts = min(1.0, mean_dist / 0.5)

    return {
        "mean_pairwise_distance": round(mean_dist, 4),
        "DTS": round(dts, 4),
        "n_ideas": n,
    }


# ---------------------------------------------------------------------------
# 5. COHERENCE RETENTION SCORE (CRS)  ← the counterpart to PEV
#    Perplexity-based: low perplexity = coherent text.
#    Normalised so that CRS=1.0 means baseline coherence, CRS→0 means chaos.
# ---------------------------------------------------------------------------

def compute_crs(
    text: str,
    model,       # HuggingFace CausalLM
    tokenizer,
    baseline_perplexity: Optional[float] = None,
    max_length: int = 512,
) -> dict:
    """
    Args:
        text: generated text to evaluate
        model/tokenizer: HuggingFace model (in eval mode, on correct device)
        baseline_perplexity: perplexity of sober baseline (for relative CRS)
    Returns:
        dict with perplexity and CRS
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = math.exp(loss.item())

    if baseline_perplexity is not None and baseline_perplexity > 0:
        crs = min(1.0, baseline_perplexity / perplexity)
    else:
        # Standalone: CRS from absolute perplexity (< 50 → good, > 500 → chaos)
        crs = max(0.0, 1.0 - math.log(max(1, perplexity)) / math.log(500))

    return {
        "perplexity": round(perplexity, 2),
        "CRS": round(crs, 4),
        "baseline_perplexity": baseline_perplexity,
    }
