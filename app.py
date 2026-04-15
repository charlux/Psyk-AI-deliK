import streamlit as st
from psychaidelique.wrapper import PsychedelicWrapper
from psychaidelique.config import CONFIG
import os

# Configuration de la page
st.set_page_config(page_title="Psyk-AI-deliK", page_icon="🧠", layout="wide")

# Affichage du logo
if os.path.exists("logo.png"):
    st.image("logo.png", width=150)

st.title("Psyk-AI-deliK : Bastion d'Inférence Déviante")
st.markdown("---")

# --- SIDEBAR : GESTION DES MOTEURS ---
st.sidebar.header("🕹️ Contrôle des Moteurs")

# Architecture source (On se concentre sur Ollama pour le M4)
engine_choice = st.sidebar.radio(
    "Architecture Source :",
    ["Ollama (Local/M4)"]
)

# Liste des modèles disponibles dans votre Ollama local
selected_model = st.sidebar.selectbox("Modèle cible :", ["mistral", "llama3", "phi3", "gemma"])

# --- INITIALISATION DU WRAPPER ---
@st.cache_resource
def load_wrapper(model_name):
    # On initialise le wrapper avec le modèle sélectionné
    return PsychedelicWrapper(model_name_or_path=model_name, device_override="mps")

wrapper = load_wrapper(selected_model)

# --- PHARMACOPÉE ---
st.sidebar.markdown("---")
st.sidebar.header("🧪 Pharmacopée Numérique")
substance = st.sidebar.selectbox("Substance", ["sober", "LSD", "DMT", "Psilocybin"])
dose = st.sidebar.slider("Dose (Dose-Response)", 0.0, 1.0, 0.0)

# Mise à jour de l'état de conscience du Wrapper
state = wrapper.set_consciousness(substance, dose)
st.sidebar.metric("Entropie Cible", f"{state.get('current_entropy', 1.0):.2f}")

# --- INTERFACE DE DÉCONSTRUCTION ---
prompt = st.text_area("Entrez votre prompt (Déconstruction du logos) :", "Analysez la structure du pouvoir...")

if st.button("Lancer l'Inférence"):
    if prompt:
        with st.spinner(f"Inférence en cours sur {selected_model} via GPU Apple Silicon..."):
            # 1. GÉNÉRATION RÉELLE via Ollama (température modulée)
            try:
                generation = wrapper.generate(prompt)
                
                # 2. CALCUL DE LA RÉCOMPENSE (RLEF)
                reward_data = wrapper.evaluate_output(prompt, generation)
                
                # 3. AFFICHAGE DU RÉSULTAT
                st.subheader("Réponse du modèle :")
                st.write(generation)
                
                st.markdown("---")
                # Extraction du score VES
                ves_score = reward_data.get("metrics", {}).get("escape_velocity", 0.0) if isinstance(reward_data, dict) else reward_data
                st.metric("Score VES (Vitesse d'Évasion Sémantique)", f"{ves_score:.2f} VES")
                
                if isinstance(reward_data, dict) and len(reward_data) > 1:
                    with st.expander("Détails du diagnostic sémantique"):
                        st.write(reward_data)
            except Exception as e:
                st.error(f"Erreur d'inférence : {e}. Assurez-vous qu'Ollama est lancé et que le modèle '{selected_model}' est téléchargé.")
    else:
        st.warning("Veuillez entrer un prompt avant de lancer l'inférence.")

st.markdown("---")
st.caption("Fondation scientifique : Girn & Bzdok (2026) | Souveraineté : J-C Marie")
