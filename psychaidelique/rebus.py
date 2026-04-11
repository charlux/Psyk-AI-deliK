"""
Psyk-AI-deliK - REBUS Logic
Implementing the relaxation of prior beliefs in transformer hierarchies.
"""

import torch

class REBUSManager:
    """
    Gère la précision des priors bayésiens du modèle.
    Inspiré par Carhart-Harris & Friston (2019).
    """
    def __init__(self, relaxation_factor: float = 0.5):
        self.relaxation_factor = relaxation_factor

    def relax_priors(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Réduit la 'précision' (la certitude) du modèle.
        En termes sociologiques, cela revient à douter de la norme institutionnelle.
        """
        # Une température plus élevée aplatit la distribution (REBUS)
        # On évite que le modèle ne s'enferme dans la réponse la plus probable.
        relaxed_logits = logits / (1.0 + self.relaxation_factor)
        return relaxed_logits

    def calculate_surprise(self, probs: torch.Tensor) -> torch.Tensor:
        """Calcule l'entropie de surprise du modèle."""
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
