"""
Psyk-AI-deliK - The Sovereign Wrapper
Orchestrating RLEF-guided inference by intercepting standard Transformer flows.
"""

import mlx.core as mx
from mlx_lm import load, generate
from .profiles import PsychedelicLibrary
from .attention import PsychedelicAttention
from .bridges import BridgeManager
from .reward_model import RLEFRewardModel

class PsychedelicWrapper:
    """
    The main entry point. It wraps a standard MLX model and applies 
    structural reconfiguration during inference.
    """
    def __init__(self, config):
        self.config = config
        # Chargement du modèle et du tokenizer via mlx-lm
        self.model, self.tokenizer = load(config.model_name)
        
        # Initialisation des composants subversifs
        self.library = PsychedelicLibrary()
        self.reward_engine = RLEFRewardModel()
        
        # Récupération de la signature selon le dosage
        self.params = self.library.get_dose_response(config.profile, config.dose)
        
        # Initialisation du moteur d'attention et des ponts
        self.psyk_attention = PsychedelicAttention(
            entropy_factor=self.params["current_entropy"],
            connectivity_boost=self.params["current_bridge"]
        )
        
        self.num_layers = len(self.model.layers)
        self.bridge_manager = BridgeManager(self.num_layers, self.model.model_dims)

    def generate(self, prompt: str, psychedelic: bool = True, max_tokens: int = 500):
        """
        Génère du texte en alternant entre mode standard et mode psychédélique (RLEF).
        """
        if not psychedelic:
            # Inférence standard (Baseline institutionnelle)
            return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

        # Inférence souveraine (RLEF Active)
        # Note : Dans une implémentation réelle, nous intercepterions ici les 
        # hooks d'attention de self.model pour injecter self.psyk_attention.
        
        print(f"--- Launching RLEF Inference [{self.config.profile} @ {self.config.dose}] ---")
        
        # Pour cette version, nous utilisons les paramètres cinétiques pour 
        # influencer la génération via le contrôle de température et de top_p
        # calqués sur l'entropie de Girn/Bzdok.
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            temp=self.params["current_entropy"], # Injection de la température entropique
        )
        
        return response

    def visualize_fingerprint(self, prompt: str):
        """
        Simule la visualisation de l'empreinte neurale (Entropie vs Profondeur).
        """
        print(f"Visualizing neural reconfiguration for: {self.config.profile}")
        # Logique de plotting à implémenter dans notebooks/
        pass
