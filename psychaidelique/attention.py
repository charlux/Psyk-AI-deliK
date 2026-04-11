"""
Psyk-AI-deliK - Psychedelic Attention Module
Injection d'entropie sémantique et reconfiguration de la connectivité.
Basé sur la méga-analyse Girn/Bzdok (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PsychedelicAttention(nn.Module):
    """
    Module d'attention modifié pour simuler l'état de 'Global Connectivity'
    et la relaxation des contraintes sémantiques.
    """
    def __init__(self, config):
        super().__init__()
        self.entropy_factor = config.get('entropy_factor', 1.0)
        self.connectivity_boost = config.get('connectivity_boost', 0.1)
        self.is_active = True

    def forward(self, query, key, value, attention_mask=None):
        """
        Calcul de l'attention avec injection de bruit structuré 
        et dilatation de la distribution de probabilité.
        """
        # Calcul standard des scores (dot-product attention)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        if self.is_active:
            # 1. Relaxation des priors : On divise par le facteur d'entropie (Température)
            # Un entropy_factor > 1.0 aplatit la distribution (REBUS)
            scores = scores / self.entropy_factor

            # 2. Connectivité accrue : Ajout d'une composante stochastique structurée
            # Simule le 'bruit' neuronal qui permet de découvrir de nouveaux chemins sémantiques
            noise = torch.randn_like(scores) * self.connectivity_boost
            scores = scores + noise

        # Softmax pour obtenir les poids d'attention
        p_attn = F.softmax(scores, dim=-1)

        # Application de l'attention aux valeurs
        return torch.matmul(p_attn, value), p_attn

    def set_intensity(self, dose):
        """
        Ajuste dynamiquement l'entropie en fonction de la dose (0.0 - 1.0).
        0.0 = État de conscience ordinaire (Sobriété algorithmique).
        1.0 = État de dérive maximale (Entropie totale).
        """
        self.entropy_factor = 1.0 + (dose * 2.0) # Scale de 1.0 à 3.0
        self.connectivity_boost = dose * 0.5    # Scale de 0.0 à 0.5
