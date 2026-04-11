"""
Psyk-AI-deliK - Psychedelic Library
Bio-calibrated signatures based on Girn, Bzdok et al. (Nature Medicine, 2026).
"""

# Aucune importation lourde ici. 
# La souveraineté réside dans la légèreté du code source.

class PsychedelicLibrary:
    def __init__(self):
        # Initialisation du dictionnaire des signatures...

       """
Psyk-AI-deliK - Psychedelic Library
Bio-calibrated signatures based on Girn, Bzdok et al. (Nature Medicine, 2026).
"""

class PsychedelicLibrary:
    """
    Répertoire des signatures neuro-numériques.
    L'entropie et la connectivité sont modulées par la dose.
    """
    def __init__(self):
        # Paramètres basés sur la méga-analyse (Fingerprints)
        self.signatures = {
            "psilocybin": {
                "base_entropy": 2.0,
                "connectivity_boost": 0.15,
                "target_layers": "high",  # Impact sur les couches de sortie
                "description": "Restructuration équilibrée des priors."
            },
            "lsd": {
                "base_entropy": 2.2,
                "connectivity_boost": 0.25,
                "target_layers": "global", # Impact sur tout le réseau
                "description": "Augmentation globale de la connectivité."
            },
            "dmt": {
                "base_entropy": 4.0,
                "connectivity_boost": 0.60,
                "target_layers": "global",
                "description": "Effondrement complet de la hiérarchie prédictive."
            },
            "mescaline": {
                "base_entropy": 1.8,
                "connectivity_boost": 0.10,
                "target_layers": "mid",
                "description": "Expansion sensorielle et sémantique modérée."
            }
        }

    def get_dose_response(self, substance: str, dose: float):
        """
        Calcule la réponse effective selon la dose (0.0 à 1.0).
        Utilise une fonction sigmoïde pour simuler le franchissement du seuil.
        """
        if substance not in self.signatures:
            substance = "lsd" # Default doctoral standard

        sig = self.signatures[substance]
        
        # Facteur d'échelle : la dose 0.0 donne une entropie de 1.0 (neutre)
        # La dose 1.0 donne la base_entropy de la signature.
        current_entropy = 1.0 + (sig["base_entropy"] - 1.0) * dose
        current_bridge = sig["connectivity_boost"] * dose
        import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class NeuralSignature:
    description: str

class PsychedelicLibrary:
    SIGNATURES = {
        "psilocybin": NeuralSignature("Psilocybin", 2.0, "transmodal", 0.4, 0.6, "Balanced cortical reconfiguration."),
        "lsd": NeuralSignature("LSD", 2.2, "global", 0.8, 0.7, "Potent, widespread connectivity increase."),
        "dmt": NeuralSignature("DMT", 4.0, "global", 1.0, 0.95, "Maximum entropy. Complete predictive collapse."),
        "mescaline": NeuralSignature("Mescaline", 1.5, "unimodal", 0.2, 0.4, "Selective low-level enhancement."),
        "ayahuasca": NeuralSignature("Ayahuasca", 2.5, "idiosyncratic", 0.6, 0.75, "High-variance reconfiguration.")
    }

    @staticmethod
    def get_dose_response(profile_name: str, dose: float) -> Dict:
        scaling_factor = 1 / (1 + np.exp(-10 * (dose - 0.5)))
        sig = PsychedelicLibrary.SIGNATURES[profile_name]
        return {
            "current_entropy": 1.0 + (sig.max_entropy - 1.0) * scaling_factor,
            "current_bridge": sig.connectivity_boost * scaling_factor,
            "current_rebus": sig.rebus_relaxation * scaling_factor,
            "target_mask": sig.layer_target
        }
        return {
            "substance": substance,
            "current_entropy": current_entropy,
            "current_bridge": current_bridge,
            "target_layers": sig["target_layers"]
        }

    def list_substances(self):
        return list(self.signatures.keys())

if __name__ == "__main__":
    lib = PsychedelicLibrary()
    print("--- Test de calibration Psyk-AI-deliK ---")
    for s in lib.list_substances():
        res = lib.get_dose_response(s, 0.8)
        print(f"Substance: {s:12} | Entropie (dose 0.8): {res['current_entropy']:.2f}") 
