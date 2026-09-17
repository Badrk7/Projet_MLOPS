import os
import sys
import requests
import streamlit as st

# Résolution du problème d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

# Chargement de la config
try:
    cfg = Config.from_yaml("configs/config.yaml")
    default_api_url = os.environ.get("API_URL", cfg.app.api_url)
except Exception:
    default_api_url = "http://localhost:8000"

st.set_page_config(page_title="Détection d'Émotions", page_icon="🎭")
st.title("🎭 IA : Détection d'Émotions")
st.caption("Interface connectée au modèle DistilBERT entraîné localement")

api_url = st.sidebar.text_input("URL de l'API (FastAPI)", value=default_api_url)
st.sidebar.markdown("---")
st.sidebar.info("Entraîné sur le dataset `dair-ai/emotion` avec 6 classes : Tristesse, Joie, Amour, Colère, Peur, Surprise.")

texte_utilisateur = st.text_area(
    "💬 Entrez une phrase en anglais :", 
    "I am so incredibly happy today, everything is perfect!"
)

if st.button("Analyser l'émotion", type="primary"):
    if not texte_utilisateur:
        st.warning("Veuillez entrer du texte.")
    else:
        with st.spinner("Analyse par le modèle..."):
            try:
                payload = {"text": texte_utilisateur}
                response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                
            except requests.RequestException as exc:
                st.error(f"❌ Impossible de contacter l'API : {exc}")
                st.info("Avez-vous pensé à lancer l'API dans un autre terminal avec 'make run_api' ?")
            else:
                label = result.get("label", "Inconnu")
                score = result.get("score", 0.0)
                
                st.success(f"**Émotion détectée : {label}**")
                st.progress(score, text=f"Confiance du modèle : {score:.1%}")