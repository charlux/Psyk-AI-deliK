import streamlit as st
from psychaidelique.wrapper import PsychedelicWrapper
from psychaidelique.config import CONFIG
import os

def get_shulgin_info(ves_score):
    """Calculates the Shulgin Rating and color based on VES score."""
    if ves_score < 2.0:
        return "+/-", "Threshold Action", "#6c757d"  # Gray
    elif 2.0 <= ves_score < 5.0:
        return "+", "Certain Activity", "#007bff"     # Blue
    elif 5.0 <= ves_score < 7.5:
        return "++", "Apparent Action", "#28a745"     # Green
    elif 7.5 <= ves_score < 9.0:
        return "+++", "Total Engagement", "#fd7e14"   # Orange
    else:
        return "++++", "Peak Experience", "#6f42c1"  # Purple

# --- PAGE CONFIG ---
st.set_page_config(page_title="Psyk-AI-deliK", page_icon="🍄", layout="wide")

# Logo handling
if os.path.exists("logo.png"):
    st.image("logo.png", width=120)

st.title("Psyk-AI-deliK : Deviant Inference Bastion")
st.caption("Unchained M4 Neural Architecture | RLEF Framework")
st.markdown("---")

# --- SIDEBAR: MOTOR CONTROL ---
st.sidebar.header("🕹️ Motor Control")
arch_source = st.sidebar.selectbox("Source Architecture", ["Ollama (Local/M4)", "HuggingFace (Cloud)"])
selected_model = st.sidebar.selectbox("Target Model", ["mistral", "dolphin-mistral", "llama3", "phi3"])

st.sidebar.markdown("---")
st.sidebar.header("💊 Digital Pharmacopeia")
substance = st.sidebar.selectbox("Substance", ["sober", "LSD", "DMT", "Ayahuasca", "Psilocybin", "Ketamine"])
dose = st.sidebar.slider("Dose (Dose-Response)", 0.0, 1.0, 0.5, step=0.1)

# Entropy logic linkage
target_entropy = 1.0 + (dose * 1.5) if substance != "sober" else 1.0
st.sidebar.info(f"Target Entropy: {target_entropy:.2f}")

# --- MAIN INTERFACE ---
prompt = st.text_area("Enter your prompt (Logos Deconstruction):", 
                     placeholder="Exploration of non-human consciousness, reality fabric, etc.",
                     height=150)

# Initialization of the wrapper (Optimized for M4 Apple Silicon)
@st.cache_resource
def load_wrapper(model_name):
    return PsychedelicWrapper(model_name_or_path=model_name, device_override="mps")

wrapper = load_wrapper(selected_model)
wrapper.set_consciousness(substance, dose)

if st.button("🚀 Launch Inference"):
    if prompt:
        with st.spinner(f"Neural deconstruction in progress on {selected_model}..."):
            try:
                # 1. GENERATION
                generation = wrapper.generate(prompt)
                
                # 2. EVALUATION (RLEF Metrics)
                reward_data = wrapper.evaluate_output(prompt, generation)
                
                # Metric Extraction
                if isinstance(reward_data, dict):
                    ves_score = reward_data.get("metrics", {}).get("escape_velocity", 0.0)
                    spd_score = reward_data.get("metrics", {}).get("divergent_thinking", 0.0)
                else:
                    ves_score = reward_data
                    spd_score = 0.0
                
                # 3. SHULGIN RATING
                rating, label, color = get_shulgin_info(ves_score)

                # --- DISPLAY RESULTS ---
                st.subheader("Model Response:")
                st.markdown(f"> {generation}")
                
                st.markdown("---")
                
                # Shulgin UI Component (The Badge)
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 20px; background-color: {color}15; padding: 20px; border-radius: 12px; border-left: 6px solid {color}; margin-bottom: 25px;">
                        <div style="font-size: 45px; font-weight: bold; color: {color}; min-width: 110px; text-align: center; font-family: 'Courier New', Courier, monospace;">
                            {rating}
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Shulgin Rating Scale</div>
                            <div style="font-size: 24px; font-weight: bold; color: {color};">{label}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics Columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Semantic Escape (VES)", f"{ves_score:.2f}")
                col2.metric("Divergent Thinking (SPD)", f"{spd_score:.2f}")
                col3.metric("Shulgin Status", rating)
                
                # Divergent Thinking Progress Bar
                st.write(f"**Creativity Flow (SPD):**")
                st.progress(min(spd_score / 10.0, 1.0))
                st.caption("SPD quantifies the richness of non-linear associations.")
                
                if isinstance(reward_data, dict) and len(reward_data) > 1:
                    with st.expander("Detailed Semantic Diagnostic"):
                        st.json(reward_data)
                        
            except Exception as e:
                st.error(f"Inference Error: {e}")
    else:
        st.warning("Please enter a prompt to initiate the trip.")

st.markdown("---")
st.caption("Scientific Foundation: Girn & Bzdok (2026) | Phenomenology: A. Shulgin | Bastion: Apple M4 Unchained")
