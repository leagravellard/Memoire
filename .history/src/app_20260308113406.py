import streamlit as st
import pandas as pd
import joblib
import requests # Pour l'API météo

# Configuration de la page
st.set_page_config(page_title="Pollution Predictor", layout="wide")

# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load('rf_meteo_model.pkl')

data = load_model()
model = data["model"]
features = data["features"]

st.title("Qualité de l'Air : Anticiper par la Météo")

# Menu de navigation
tab1, tab2, tab3 = st.tabs(["📊 Analyse", "🖱️ Prédiction Manuelle", "🏙️ Prédiction par Ville"])

with tab1:
    st.header("Pourquoi la météo ?")
    st.write("Explication de la problématique et affichage de tes graphiques...")

with tab2:
    st.header("Simulateur de conditions")
    # On mettra ici des sliders pour chaque variable (temp, vent, etc.)

with tab3:
    st.header("État actuel dans le monde")
    city = st.text_input("Entrez une ville", "Paris")
    # On mettra ici l'appel API et la prédiction automatique