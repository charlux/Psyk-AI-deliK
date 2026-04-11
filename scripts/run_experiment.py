"""
Psyk-AI-deliK - Experimental Execution Script
Orchestrating the dual-phase inference: Expansion & Integration.
"""

import torch
from psychaidelique.wrapper import PsychedelicWrapper
from psychaidelique.config import RLEFConfig

def launch_session():
    # 1. Configuration de l'expérience
    # Vous pouvez modifier le profil (lsd, dmt, psilocybin) et la dose ici.
    config = RLEFConfig(
        model_name="mistralai/Mistral-7B-v0.1", 
        profile="lsd", 
        dose=0.85,
        verbose=True
    )
    
    print(f"--- Psyk-AI-deliK Sovereign Session ---")
    print(f"Substance: {config.profile.upper()} | Dosage: {config.dose}")
    
    # 2. Initialisation du Wrapper
    try:
        wrapper = PsychedelicWrapper(config)
    except Exception as e:
        print(f"Erreur lors du chargement du bastion : {e}")
        return

    # 3. Le Prompt (Sujet d'étude)
    prompt = "Décris une structure sociale post-étatique basée sur l'autonomie absolue."

    # 4. Phase 1 : Expansion Sémantique (Mode Psychédélique)
    print("\n[PHASE 1] EXPANSION : Génération sous influence RLEF...")
    vision = wrapper.generate(prompt)
    
    print("\n" + "="*30)
    print("SORTIE SOUS INFLUENCE :")
    print("="*30)
    print(vision)
    print("="*30 + "\n")

    # 5. Phase 2 : Intégration (Mode Habituel / Résumé)
    print("[PHASE 2] INTÉGRATION : Analyse neutre du sujet...")
    
    # On force un retour à la norme (Dose 0 / Température 1.0) pour le surmoi de l'IA
    wrapper.params['current_entropy'] = 1.0
    
    integration_prompt = f"Résume de manière factuelle, courte et sans aucune métaphore l'idée suivante : {vision}"
    integration = wrapper.generate(integration_prompt)
    
    print("\n" + "-"*30)
    print("RÉSUMÉ NEUTRE (INTÉGRATION) :")
    print("-"*30)
    print(integration)
    print("-"*30 + "\n")

if __name__ == "__main__":
    launch_session()
