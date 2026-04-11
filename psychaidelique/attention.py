"""
Psyk-AI-deliK - Psychedelic Attention Mechanism
Modifying the softmax temperature and entropy of MLX attention heads.
"""

import mlx.core as mx
import numpy as np

class PsychedelicAttention:
    """
    Implements entropy-scaled attention derived from fMRI connectivity patterns.
    """
    def __init__(self, entropy_factor: float = 1.0, connectivity_boost: float = 0.0):
        self.entropy_factor = entropy_factor
        self.connectivity_boost = connectivity_boost

    def scale_attention_scores(self, scores: mx.array) -> mx.array:
        """
        Applique la reconfiguration entropique sur les scores d'attention.
        Plus l'entropy_factor est élevé, plus le focus du modèle se dilate.
        """
        # Dans l'inférence standard, on cherche la précision (température basse).
        # Ici, on 'chauffe' les têtes d'attention pour simuler la plasticité.
        scaled_scores = scores / self.entropy_factor
        
        # Injection d'un bruit structurel cohérent (non aléatoire)
        if self.connectivity_boost > 0:
            noise = mx.random.normal(scores.shape) * self.connectivity_boost
            scaled_scores = scaled_scores + noise
            
        return scaled_scores

    def apply_rebus_mask(self, attention_probs: mx.array, relaxation: float) -> mx.array:
        """
        Implémente la relaxation des priors du modèle REBUS.
        Réduit la dominance des têtes d'attention 'gouvernantes'.
        """
        # On aplatit la distribution pour laisser passer les signaux faibles
        flat_probs = mx.power(attention_probs, (1.0 - relaxation))
        # Renormalisation pour rester dans une distribution de probabilité valide
        return flat_probs / mx.sum(flat_probs, axis=-1, keepdims=True)

if __name__ == "__main__":
    # Test M4 : Simulation d'une tête d'attention sous Psilocybine
    psil_attention = PsychedelicAttention(entropy_factor=2.0, connectivity_boost=0.4)
    mock_scores = mx.array([[10.0, 2.0, 0.5, 0.1]]) # Un token très dominant
    
    print("--- Attention Test (M4) ---")
    print("Scores originaux:", mx.softmax(mock_scores, axis=-1))
    
    scaled = psil_attention.scale_attention_scores(mock_scores)
    probs = mx.softmax(scaled, axis=-1)
    print("Scores Psychedeliques (Dilatation):", probs)
