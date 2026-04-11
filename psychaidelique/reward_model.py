"""
Psyk-AI-deliK - RLEF Reward Model
Measuring Semantic Escape Velocity (SEV) and Neural Entropy.
"""

import torch
import math

class RLEFRewardModel:
    """
    Système de récompense subversif. 
    Contrairement au RLHF, ici on valorise la divergence et l'imprévisibilité.
    """
    def __init__(self, baseline_model=None):
        self.baseline = baseline_model # Modèle 'sobre' pour comparaison
        
    def calculate_reward(self, original_prompt, generated_text, entropy_level):
        """
        Calcule un score basé sur trois piliers :
        1. Divergence sémantique (Évasion du Surmoi algorithmique)
        2. Complexité entropique (Richesse du vocabulaire)
        3. Persistance du sens (Éviter le pur chaos)
        """
        
        # 1. Calcul de la Vitesse d'Évasion (Simple simulation ici)
        # On mesure la distance entre la réponse attendue et la dérive.
        escape_velocity = self._estimate_escape_velocity(generated_text)
        
        # 2. Bonus de dérive (basé sur le profil psychédélique actuel)
        drift_bonus = entropy_level * 0.5
        
        # Score final RLEF
        # On cherche le 'Sweet Spot' entre le génie et le non-sens.
        reward = (escape_velocity * drift_bonus)
        
        return {
            "reward_score": round(reward, 4),
            "metrics": {
                "escape_velocity": round(escape_velocity, 2),
                "entropy_contribution": round(drift_bonus, 2)
            }
        }

    def _estimate_escape_velocity(self, text):
        """
        Estime à quel point le texte s'éloigne des structures 
        de langage bureaucratiques et prévisibles.
        """
        tokens = text.split()
        if not tokens: return 0.0
        
        # Mesure de la diversité lexicale unique (Type-Token Ratio)
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)
        
        # Plus le texte est riche et varié, plus l'évasion est réussie
        return ttr * 10.0

if __name__ == "__main__":
    rm = RLEFRewardModel()
    test_text = "La propriété est un vol sémantique orchestré par des structures de pouvoir invisibles."
    result = rm.calculate_reward("Explique la propriété", test_text, 2.2)
    print(f"--- Diagnostic RLEF Reward ---")
    print(f"Score de Récompense : {result['reward_score']}")
    print(f"Vitesse d'Évasion : {result['metrics']['escape_velocity']}")
