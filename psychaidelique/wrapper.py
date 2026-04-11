"""
Psyk-AI-deliK - The Universal Sovereign Wrapper
Orchestrating RLEF-guided inference across all platforms.
UNIFIED VERSION for MacBook M4 (MLX/MPS) and Manjaro (CPU).
"""

import os
import torch
import math

# Imports relatifs
from .profiles import PsychedelicLibrary
from .attention import PsychedelicAttention
from .bridges import CrossLayerBridge, REBUSPriorRelaxer
from .reward_model import RLEFRewardModel

class PsychedelicWrapper:
    """
    Orchestrateur souverain capable de basculer entre 
    le MacBook M4 (MPS/MLX) et Manjaro (CPU/CUDA).
    """
    def __init__(self, model_name_or_path, device_override=None):
        # 1. Détection de l'infrastructure
        if device_override:
            self.device = device_override
        else:
            if torch.backends.mps.is_available():
                self.device = "mps" # M4
            elif torch.cuda.is_available():
                self.device = "cuda" # Nvidia
            else:
                self.device = "cpu" # OneTwo / Standard Linux
        
        print(f"--- Psyk-AI-deliK : Inférence déployée sur {self.device} ---")

        # 2. Initialisation des composants du bastion
        self.library = PsychedelicLibrary()
        self.reward_engine = RLEFRewardModel()
        self.relaxer = REBUSPriorRelaxer()
        
        # 3. État de conscience actuel (Initialisé à neutre)
        # On évite 'lsd' par défaut pour rester sur 'psilocybin' (plus équilibré)
        self.current_state = self.library.get_dose_response("psilocybin", 0.0)
        
        # 4. Attention et Ponts (Configurés dynamiquement)
        # On passe un dictionnaire minimaliste pour la portabilité
        config_bridge = {'hidden_size': 4096, 'use_mps': (self.device == "mps")}
        self.bridge = CrossLayerBridge(config_bridge)
        self.attention = PsychedelicAttention()

    def set_consciousness(self, substance, dose):
        """Ajuste la signature neuro-numérique du système."""
        # On utilise une fonction sigmoïde pour simuler la courbe dose-réponse
        sigmoide_dose = 1 / (1 + math.exp(-10 * (dose - 0.5)))
        self.current_state = self.library.get_dose_response(substance, sigmoide_dose)
        
        # Injection dans les modules mécaniques
        self.attention.entropy_factor = self.current_state["current_entropy"]
        self.attention.connectivity_boost = self.current_state["current_bridge"]
        
        return self.current_state

    def drift_inference(self, prompt, raw_logits):
        """
        Applique la dérive sémantique sur les logits d'un modèle tiers.
        """
        # Relaxation des priors via REBUS
        relaxed_logits = self.relaxer.relax(
            raw_logits, 
            temperature=self.current_state["current_entropy"]
        )
        
        return relaxed_logits

    def evaluate_output(self, prompt, output_text):
        """Mesure la Vitesse d'Évasion Sémantique (VES)."""
        return self.reward_engine.calculate_reward(
            prompt, 
            output_text, 
            self.current_state["current_entropy"]
        )

if __name__ == "__main__":
    print("--- Test de fluidité Psyk-AI-deliK ---")
    wrapper = PsychedelicWrapper("test-stub")
    state = wrapper.set_consciousness("dmt", 0.9)
    print(f"État DMT (Dose 0.9) : Entropie sémantique dilatée à {state['current_entropy']:.2f}")
