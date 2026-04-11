import os

def select_engine():
    print("--- Psyk-AI-deliK : Assistant de Souveraineté ---")
    print("1. OLLAMA (Local/Souverain)")
    print("2. HUGGING FACE (Performance/Download)")
    
    choice = input("\nVotre choix : ")
    
    if choice == "1":
        model = input("Nom du modèle Ollama (ex: mistral) : ")
        print(f"Configuration validée : Moteur Ollama sur {model}")
    else:
        print("Téléchargement des poids sémantiques requis (15 Go)...")

if __name__ == "__main__":
    select_engine()
