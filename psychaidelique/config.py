"""
Psyk-AI-deliK - Universal Configuration
Defining the experimental parameters for RLEF-guided inference.
"""

from dataclasses import dataclass

@dataclass
class RLEFConfig:
    # Le nom du modèle sur HuggingFace (ex: "mistralai/Mistral-7B-Instruct-v0.2")
    model_name: str
    
    # Le profil psychédélique choisi (psilocybin, lsd, dmt, etc.)
    profile: str = "psilocybin"
    
    # La dose (0.0 à 1.0) - Définit le franchissement de la sigmoïde
    dose: float = 0.8
    
    # Activation du guidage RLEF (Reinforcement Learning by Entheogenic Feedback)
    rlef_guidance: bool = True
    
    # Activation des ponts inter-couches (Cross-Layer Bridges)
    cross_layer: bool = True
    
    # Niveau de verbosité du log doctoral
    verbose: bool = True
