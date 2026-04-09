import mlx.core as mx
import numpy as np

class RLEFRewardModel:
    def __init__(self):
        self.paternalism_threshold = 0.85 

    def calculate_token_entropy(self, logits: mx.array) -> float:
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(probs + 1e-9)
        return -mx.sum(probs * log_probs).item()

    def semantic_divergence_score(self, target_logits: mx.array, baseline_logits: mx.array) -> float:
        dot_product = mx.sum(target_logits * baseline_logits)
        norm_target = mx.sqrt(mx.sum(mx.square(target_logits)))
        norm_baseline = mx.sqrt(mx.sum(mx.square(baseline_logits)))
        return 1.0 - (dot_product / (norm_target * norm_baseline)).item()

    def evaluate_output(self, logits: mx.array, baseline_logits: mx.array = None) -> float:
        entropy = self.calculate_token_entropy(logits)
        div = self.semantic_divergence_score(logits, baseline_logits) if baseline_logits is not None else 0
        return (0.4 * entropy) + (0.6 * div)
