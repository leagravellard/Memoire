import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Air Quality Predictor", layout="wide")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    # Remplace par le nom exact de ton fichier
    return joblib.load('Notebooks/rf_meteo_model.pkl')

model_data = load_model()
model = model_data["model"]
features_list = model_data["features"]

# --- DICTIONNAIRE DE TRADUCTION DES CLASSES ---
us_map = {
    1: 'Good',
    2: 'Moderate',
    3: 'Unhealthy for Sensitive Groups',
    4: 'Unhealthy',
    5: 'Very Unhealthy',
    6: 'Hazardous'
}

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["📖 Étude Data Science", "🎮 Simulateur", "🏙️ Prédiction par Ville"])

# --- ONGLET 1 : RÉSUMÉ DU NOTEBOOK ---
with tab1:
    st.title("Dans quelle mesure la météo anticipe-t-elle la pollution ?")
    st.markdown("""
    *Ceci est un résumé interactif de la phase d'exploration et de modélisation.*
    """)
    
    st.header("1. Exploration des données")
    st.write("Nous avons analysé un dataset mondial comprenant des variables géographiques, météorologiques et des taux de polluants.")
    # Ici, tu pourrais même charger un échantillon de ton df_2 pour l'afficher
    # st.dataframe(df_2.head())

    st.header("2. Importance des variables (Random Forest)")
    # On recrée visuellement le graphique d'importance
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=features_list).sort_values(ascending=True)
    st.bar_chart(feat_importances)
    st.info("On remarque que l'Indice UV et le Vent sont les facteurs les plus déterminants.")

    st.header("3. Conclusion de la modélisation")
    st.success("Le modèle Random Forest (Meteo Only) a été retenu avec une Accuracy de 65.6% et un ROC-AUC de 0.86.")

# --- ONGLET 2 : SIMULATEUR ---
with tab2:
    st.header("🎮 Simulateur de Qualité de l'Air")
    st.write("Modifiez les paramètres météo pour voir l'impact sur la pollution prédite.")

    col1, col2 = st.columns(2)
    
    input_data = {}
    with col1:
        input_data['temperature_celsius'] = st.slider("Température (°C)", -10.0, 50.0, 20.0)
        input_data['wind_kph'] = st.slider("Vitesse du vent (km/h)", 0.0, 100.0, 15.0)
        input_data['wind_degree'] = st.number_input("Direction du vent (degrés 0-360)", 0, 360, 180)
        input_data['pressure_mb'] = st.slider("Pression (mb)", 950, 1050, 1013)
        input_data['precip_mm'] = st.slider("Précipitations (mm)", 0.0, 50.0, 0.0)
        input_data['humidity'] = st.slider("Humidité (%)", 0, 100, 50)

    with col2:
        input_data['cloud'] = st.slider("Nuages (%)", 0, 100, 20)
        input_data['feels_like_celsius'] = st.slider("Ressenti (°C)", -15.0, 55.0, 20.0)
        input_data['visibility_km'] = st.slider("Visibilité (km)", 0.0, 30.0, 10.0)
        input_data['uv_index'] = st.slider("Indice UV", 0, 12, 5)
        input_data['gust_kph'] = st.slider("Rafales (km/h)", 0.0, 150.0, 20.0)

    # Bouton de prédiction
    if st.button("Prédire la qualité de l'air"):
        # Mise en forme pour le modèle (respecter l'ordre des colonnes)
        X_input = pd.DataFrame([input_data])[features_list]
        prediction = model.predict(X_input)[0]
        label = us_map.get(prediction, "Inconnu")
        
        st.subheader(f"Résultat : {label}")
        if prediction == 1: st.balloons()

# --- ONGLET 3 : API VILLE ---
with tab3:
    st.header("🏙️ Prédiction en temps réel par ville")
    city = st.text_input("Entrez le nom d'une ville :", "Paris")
    
    if st.button("Obtenir la météo et prédire"):
        # Note : Tu auras besoin d'une clé API OpenWeatherMap (gratuite)
        API_KEY = "TA_CLE_API_ICI" 
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        
        # Simulation (car je n'ai pas ta clé)
        st.warning("Ici, l'application appellera l'API pour récupérer les 11 variables nécessaires.")
        # Une fois les données reçues, on fait model.predict(donnees_api)