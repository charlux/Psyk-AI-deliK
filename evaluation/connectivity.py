"""
Psyk-AI-deliK - Connectivity Analysis
Measures Attention Entropy and Cross-Layer Mutual Information.
"""
import torch

class ConnectivityAnalyst:
    def measure_entropy(self, attention_weights):
        """
        Calcule l'entropie de Shannon sur les poids d'attention.
        Simule l'augmentation de la connectivité fonctionnelle (fMRI).
        """
        # On évite les zéros pour le log
        probs = torch.clamp(attention_weights, min=1e-9)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean().item()

    def cross_layer_sync(self, layer_a_states, layer_b_states):
        """Mesure la corrélation entre couches distantes (Ponts)."""
        cos = torch.nn.CosineSimilarity(dim=-1)
        return cos(layer_a_states, layer_b_states).mean().item()
