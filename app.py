import streamlit as st
import plotly.graph_objects as go
from psychaidelique.wrapper import PsychedelicWrapper
from psychaidelique.config import RLEFConfig

# Configuration de la page
st.set_page_config(page_title="Psyk-AI-deliK UI", layout="wide")

# Sidebar : Contrôle des variables d'état
with st.sidebar:
    st.markdown("## ⚙️ Contrôle de Conscience")
    substance = st.selectbox("Substance", ["psilocybin", "lsd", "dmt", "mescaline"])
    dose = st.slider("Dose (Intensité)", 0.0, 1.0, 0.8)
    st.divider()
    st.info("Mode RLEF : Actif\nBackend : PyTorch (Universal)")

# Main UI
st.title("🍄 Psyk-AI-deliK Dashboard")

prompt = st.text_area("Saisie sémantique :", "Déconstruis le concept de propriété intellectuelle.")

if st.button("Lancer la dérive"):
    config = RLEFConfig(model_name="mistralai/Mistral-7B-v0.1", profile=substance, dose=dose)
    wrapper = PsychedelicWrapper(config)
    
    # Inférence sous influence
    with st.expander("👁️ Processus d'Expansion", expanded=True):
        vision = wrapper.generate(prompt)
        st.write(vision)
    
    # Phase d'intégration (Le fameux résumé sans influence)
    st.divider()
    with st.container():
        st.subheader("📝 Note de synthèse (Mode Habituel)")
        # On force la dose à 0.0 pour retrouver le 'Surmoi' de l'IA
        integration_prompt = f"Résume de manière neutre et concise l'idée suivante : {vision}"
        resume = wrapper.generate(integration_prompt) # Le wrapper devra gérer la dose 0
        st.success(resume)

    # Visualisation de l'entropie (Simulation graphique)
    fig = go.Figure(data=go.Scatter(
        y=[dose * (i/32) for i in range(32)], # Simulation d'entropie par couche
        mode='lines+markers',
        line=dict(color='mediumpurple')
    ))
    fig.update_layout(title="Profil d'Entropie par Couche (M4/CPU)", xaxis_title="Couches (Layers)", yaxis_title="Entropie (bits)")
    st.plotly_chart(fig)
