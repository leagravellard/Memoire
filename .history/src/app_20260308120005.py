import streamlit as st
import pandas as pd
import joblib
import requests
import streamlit.components.v1 as components
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Air Quality Predictor - Météo & Pollution",
    page_icon="🌍",
    layout="wide"
)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    # Chemin vers le modèle sauvegardé dans le dossier Notebooks
    model_path = os.path.join('Notebooks', 'rf_meteo_model.pkl')
    return joblib.load(model_path)

try:
    data_model = load_model()
    model = data_model["model"]
    features_list = data_model["features"]
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# Dictionnaire de correspondance pour l'indice US EPA
us_map = {
    1: 'Good (Bon)',
    2: 'Moderate (Modéré)',
    3: 'Unhealthy for Sensitive Groups (Médiocre)',
    4: 'Unhealthy (Mauvais)',
    5: 'Very Unhealthy (Très mauvais)',
    6: 'Hazardous (Dangereux)'
}

# --- NAVIGATION PAR ONGLETS ---
tab_full, tab_summary, tab_sim, tab_city = st.tabs([
    "📓 Notebook Intégral", 
    "📝 Résumé & Réponse", 
    "🎮 Simulateur", 
    "🏙️ Prédiction par Ville"
])

# --- 1. PAGE : NOTEBOOK INTÉGRAL ---
with tab_full:
    st.title("📓 Travail d'Exploration et de Modélisation")
    st.write("Ce volet présente l'intégralité de la démarche scientifique réalisée sous Jupyter Notebook.")
    
    html_file_path = os.path.join('Notebooks', '01_exploration.html')
    
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Affichage du notebook avec scroll interne
        components.html(html_content, height=850, scrolling=True)
    else:
        st.warning("⚠️ Le fichier HTML du notebook est introuvable.")
        st.info("Lancez la commande suivante dans votre terminal pour le générer :")
        st.code("jupyter nbconvert --to html Notebooks/01_exploration.ipynb")

# --- 2. PAGE : RÉSUMÉ & RÉPONSE À LA PROBLÉMATIQUE ---
with tab_summary:
    st.title("📝 Synthèse de l'Étude")
    
    st.markdown("""
    ### 🎯 Problématique
    **Dans quelle mesure les variables météorologiques permettent-elles d’anticiper la catégorie globale de qualité de l’air, indépendamment des mesures directes de polluants ?**
    
    ### 📊 Conclusion de la Modélisation
    Après comparaison de plusieurs algorithmes (Random Forest, XGBoost, Régression Logistique), le modèle **Random Forest** a été sélectionné pour sa robustesse.
    
    - **Modèle retenu :** Random Forest (Scénario Météo uniquement)
    - **Accuracy :** ~65.6%
    - **ROC-AUC :** 0.86
    
    ### 💡 Ce qu'il faut retenir
    Les résultats montrent que la météo est un **indicateur avancé performant**. 
    L'influence du vent (dispersion) et de l'indice UV (réactions chimiques créant l'ozone) sont les facteurs les plus critiques pour prédire la qualité de l'air sans capteurs de pollution coûteux.
    """)
    
    st.info("Cette application utilise le modèle final entraîné pour réaliser des prédictions en temps réel.")

# --- 3. PAGE : SIMULATEUR ---
with tab_sim:
    st.title("🎮 Simulateur de Conditions Météo")
    st.write("Modifiez les paramètres pour voir comment le modèle prédit la qualité de l'air.")
    
    col1, col2 = st.columns(2)
    inputs = {}
    
    with col1:
        inputs['temperature_celsius'] = st.slider("Température (°C)", -10.0, 50.0, 20.0)
        inputs['wind_kph'] = st.slider("Vitesse du vent (km/h)", 0.0, 150.0, 20.0)
        inputs['wind_degree'] = st.number_input("Direction du vent (0-360°)", 0, 360, 180)
        inputs['pressure_mb'] = st.slider("Pression (mb)", 950, 1050, 1013)
        inputs['precip_mm'] = st.slider("Précipitations (mm)", 0.0, 50.0, 0.0)
        inputs['humidity'] = st.slider("Humidité (%)", 0, 100, 50)

    with col2:
        inputs['cloud'] = st.slider("Couverture nuageuse (%)", 0, 100, 20)
        inputs['feels_like_celsius'] = st.slider("Ressenti (°C)", -15.0, 55.0, 20.0)
        inputs['visibility_km'] = st.slider("Visibilité (km)", 0.0, 30.0, 10.0)
        inputs['uv_index'] = st.slider("Indice UV", 0, 12, 5)
        inputs['gust_kph'] = st.slider("Rafales de vent (km/h)", 0.0, 150.0, 25.0)

    if st.button("Lancer la prédiction simulation"):
        df_sim = pd.DataFrame([inputs])[features_list]
        prediction = model.predict(df_sim)[0]
        
        st.markdown("---")
        st.subheader(f"Résultat prédit : **{us_map[prediction]}**")
        
        # Petit feedback visuel selon le résultat
        if prediction == 1:
            st.success("L'air est sain selon ces conditions !")
        elif prediction >= 4:
            st.error("Attention : Risque de pollution élevée !")

with tab_city:
    st.title("🏙️ Prédiction en Temps Réel par Ville")
    st.write("Cette page utilise l'API OpenWeatherMap pour interroger la météo actuelle d'une ville et injecter les données dans le modèle prédictif.")
    
    city = st.text_input("Entrez le nom d'une ville (ex: Lyon, Berlin, Montreal) :", "")
    
    if st.button("Obtenir la météo et prédire la pollution"):
        if city:
            # REMPLACEZ 'VOTRE_CLE_ICI' par la clé copiée sur le site
            API_KEY = "e194bd0b10a152fb7eb4fd12a2893a67" 
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    w = response.json()
                    
                    # Transformation des données API pour correspondre aux 11 features du modèle
                    weather_data = {
                        'temperature_celsius': w['main']['temp'],
                        'wind_kph': w['wind']['speed'] * 3.6,        # Conversion m/s en km/h
                        'wind_degree': w['wind']['deg'],
                        'pressure_mb': w['main']['pressure'],
                        'precip_mm': w.get('rain', {}).get('1h', 0), # Gère le cas où il ne pleut pas
                        'humidity': w['main']['humidity'],
                        'cloud': w['clouds']['all'],
                        'feels_like_celsius': w['main']['feels_like'],
                        'visibility_km': w.get('visibility', 10000) / 1000,
                        'uv_index': 5.0,                             # Valeur moyenne par défaut
                        'gust_kph': w.get('wind', {}).get('gust', w['wind']['speed']) * 3.6
                    }
                    
                    # Conversion en DataFrame (respect de l'ordre des colonnes utilisé lors de l'entraînement)
                    df_city = pd.DataFrame([weather_data])[features_list]
                    
                    # Prédiction
                    pred_city = model.predict(df_city)[0]
                    
                    st.success(f"✅ Données météo récupérées pour {city}")
                    
                    # Affichage des métriques météo
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Température", f"{weather_data['temperature_celsius']} °C")
                    m2.metric("Vitesse Vent", f"{round(weather_data['wind_kph'], 1)} km/h")
                    m3.metric("Humidité", f"{weather_data['humidity']} %")
                    
                    # Affichage du résultat final
                    st.markdown("---")
                    st.subheader(f"Qualité de l'air estimée : **{us_map[pred_city]}**")
                else:
                    st.error(f"Ville '{city}' introuvable ou service indisponible (Code: {response.status_code}).")
            except Exception as e:
                st.error(f"Erreur technique : {e}")
        else:
            st.warning("Veuillez saisir une ville.")