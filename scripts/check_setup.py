"""
Psyk-AI-deliK - Environment Integrity Check
Verifying hardware, torch backend, and directory structure.
"""

import os
import sys
import torch

def check_integrity():
    print("--- Psyk-AI-deliK: Integrity Check (Sovereign Mode) ---")
    
    # 1. Vérification du Hardware
    print(f"Python Version: {sys.version.split()[0]}")
    if torch.backends.mps.is_available():
        device = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device = "Nvidia (CUDA)"
    else:
        device = "Universal CPU (Standard Manjaro/OneTwo)"
    print(f"Detection Hardware: {device}")

    # 2. Vérification de l'arborescence
    required_dirs = ['psychaidelique', 'evaluation', 'scripts']
    for d in required_dirs:
        status = "OK" if os.path.isdir(d) else "MANQUANT"
        print(f"Répertoire {d:15}: {status}")

    # 3. Vérification des composants critiques
    critical_files = [
        'psychaidelique/profiles.py',
        'psychaidelique/attention.py',
        'psychaidelique/wrapper.py',
        'psychaidelique/reward_model.py'
    ]
    for f in critical_files:
        status = "PRÊT" if os.path.isfile(f) else "ERREUR"
        print(f"Composant {f:25}: {status}")

    print("\n--- Diagnostic Doctoral ---")
    if device == "Universal CPU (Standard Manjaro/OneTwo)":
        print("Note: L'inférence sera lente mais la souveraineté est préservée.")
    else:
        print("Note: Accélération matérielle détectée. Inférence à haute vélocité.")

if __name__ == "__main__":
    check_integrity()
