"""
Psyk-AI-deliK - Universal Psychedelic Attention (PyTorch)
Modifying the softmax temperature and entropy for cross-platform sovereignty.
"""

import torch
import torch.nn.functional as F

class PsychedelicAttention:
    """
    Implements entropy-scaled attention using PyTorch.
    Works on CPU (OneTwo) and GPU/NPU (M4/Nvidia).
    """
    def __init__(self, entropy_factor: float = 1.0, connectivity_boost: float = 0.0):
        self.entropy_factor = entropy_factor
        self.connectivity_boost = connectivity_boost

    def scale_attention_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Dilatation entropique des scores.
        PyTorch gère les tenseurs sur n'importe quel device (cpu, cuda, mps).
        """
        # Division par le facteur d'entropie (Température)
        scaled_scores = scores / self.entropy_factor
        
        if self.connectivity_boost > 0:
            # Injection de bruit structurel sur le même device que les scores
            noise = torch.randn_like(scores) * self.connectivity_boost
            scaled_scores = scaled_scores + noise
            
        return scaled_scores

    def apply_rebus_mask(self, attention_probs: torch.Tensor, relaxation: float) -> torch.Tensor:
        """
        Relaxation des priors REBUS.
        On aplatit la distribution pour libérer l'inférence des dogmes statistiques.
        """
        # On utilise une puissance inverse pour augmenter l'entropie
        flat_probs = torch.pow(attention_probs, (1.0 - relaxation))
        
        # Renormalisation (Somme = 1)
        return flat_probs / flat_probs.sum(dim=-1, keepdim=True)
