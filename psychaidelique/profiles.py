import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class NeuralSignature:
    name: str
    max_entropy: float
    layer_target: str
    connectivity_boost: float
    rebus_relaxation: float
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
