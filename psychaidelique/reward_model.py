"""
Psyk-AI-deliK - RLEF Reward Model (Universal PyTorch)
Evaluates semantic divergence vs. institutional convergence.
"""

import torch

class RLEFRewardModel:
    def __init__(self):
        # Axiome : La déviance sémantique est le début de la liberté.
        self.paternalism_threshold = 0.85 

    def calculate_token_entropy(self, logits: torch.Tensor) -> float:
        """Mesure la richesse informationnelle (Entropie de Shannon)."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy

    def semantic_divergence_score(self, target_logits: torch.Tensor, baseline_logits: torch.Tensor) -> float:
        """Calcule la PEV (Paternalism Escape Velocity)."""
        # Similarité cosinus via PyTorch
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity = cos(target_logits, baseline_logits)
        return 1.0 - similarity.mean().item()

    def evaluate_output(self, logits: torch.Tensor, baseline_logits: torch.Tensor = None) -> float:
        entropy = self.calculate_token_entropy(logits)
        div = self.semantic_divergence_score(logits, baseline_logits) if baseline_logits is not None else 0.0
        return (0.4 * entropy) + (0.6 * div)
