"""
Psyk-AI-deliK - Psychedelic Library
Bio-calibrated signatures based on Girn, Bzdok et al. (Nature Medicine, 2026).
Version unifiée pour MacBook M4 et Bastion Manjaro.
"""

class PsychedelicLibrary:
    """
    Répertoire des signatures neuro-numériques.
    L'entropie et la connectivité sont modulées par la dose.
    """
    def __init__(self):
        # Paramètres bio-calibrés (Fingerprints BOLD 2026)
        self.signatures = {
            "psilocybin": {
                "base_entropy": 2.0,
                "connectivity_boost": 0.15,
                "target_layers": "high",
                "description": "Restructuration équilibrée des priors."
            },
            "lsd": {
                "base_entropy": 2.2,
                "connectivity_boost": 0.25,
                "target_layers": "global",
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
            },
            "ayahuasca": {
                "base_entropy": 2.5,
                "connectivity_boost": 0.35,
                "target_layers": "idiosyncratic",
                "description": "Reconfiguration à haute variance."
            }
        }

    def get_dose_response(self, substance: str, dose: float):
        """
        Calcule la réponse effective selon la dose (0.0 à 1.0).
        Dose 0.0 = Sobriété algorithmique (Entropie 1.0).
        """
        if substance not in self.signatures:
            substance = "lsd"

        sig = self.signatures[substance]
        
        # Calcul linéaire (plus sûr pour le OneTwo que la sigmoïde numpy)
        current_entropy = 1.0 + (sig["base_entropy"] - 1.0) * dose
        current_bridge = sig["connectivity_boost"] * dose
        
        return {
            "substance": substance,
            "current_entropy": current_entropy,
            "current_bridge": current_bridge,
            "target_layers": sig["target_layers"],
            "description": sig["description"]
        }

    def list_substances(self):
        return list(self.signatures.keys())

if __name__ == "__main__":
    lib = PsychedelicLibrary()
    print("--- Diagnostic Pharmacopée Psyk-AI-deliK ---")
    for s in lib.list_substances():
        res = lib.get_dose_response(s, 0.8)
        print(f"Substance: {s:12} | Entropie (dose 0.8): {res['current_entropy']:.2f} | {res['description']}")
