import streamlit as st
import pandas as pd
import joblib
import requests
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ---
# Cela va chercher le fichier .env et charger la clé API
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

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

# --- TITRE GÉNÉRAL (PROBLÉMATIQUE) ---
st.markdown("""
    <h2 style='text-align: center; color: #1E88E5; margin-top: -50px;'>
    Dans quelle mesure les variables météorologiques permettent-elles d’anticiper la catégorie globale de qualité de l’air, indépendamment des mesures directes de polluants ?
    </h2>
    """, unsafe_allow_html=True)

# 1. Ajoute l'onglet dans la liste
tab_full, tab_summary, tab_sim, tab_city, tab_history = st.tabs([
    "Notebook", 
    "Résumé & Réponse", 
    "Simulateur", 
    "Prédiction par Ville",
    "Historique"
])

# --- 1. PAGE : NOTEBOOK INTÉGRAL ---
with tab_full:
    st.title("Travail d'Exploration et de Modélisation")
    st.write("Ce volet présente l'intégralité de la démarche réalisée pour répondre à la problématique.")
    
    html_file_path = os.path.join('Notebooks', '01_exploration.html')
    
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=850, scrolling=True)
    else:
        st.warning("⚠️ Le fichier HTML du notebook est introuvable.")
        st.info("Lancez 'jupyter nbconvert --to html Notebooks/01_exploration.ipynb' pour le générer.")

# --- 2. PAGE : RÉSUMÉ & RÉPONSE ---
with tab_summary:
    st.title("📝 Synthèse de l'Étude")
    
    # --- PROBLÉMATIQUE ---
    st.markdown("""
    ### 🎯 Problématique
    **Dans quelle mesure les variables météorologiques permettent-elles d'anticiper la catégorie globale de qualité de l'air, indépendamment des mesures directes de polluants ?**
    
    ---
    """)
    
    # --- MÉTHODOLOGIE ---
    st.markdown("""
    ### 🔬 Méthodologie
    
    #### 📥 Source et Préparation des Données
    - **Dataset :** GlobalWeatherRepository contenant des mesures météorologiques et de pollution en temps réel
    - **Variables collectées :** 
      - 🌡️ Paramètres météo (température, humidité, pression, vent, UV, visibilité, etc.)
      - 🏭 Mesures de polluants (CO, O₃, NO₂, SO₂, PM2.5, PM10)
      - 🌍 Indices de qualité de l'air (US EPA Index, GB DEFRA Index)
    
    #### 🛠️ Étapes de Traitement
    1. **Nettoyage :** Suppression des colonnes redondantes, conversion des types de données
    2. **Enrichissement :** Extraction de variables temporelles (année, mois, jour, heure)
    3. **Normalisation :** Standardisation des variables numériques
    4. **Division :** Partitionnement train/test pour évaluation impartiale
    
    #### 🤖 Stratégie de Modélisation
    Nous avons comparé 3 algorithmes sur 3 scénarios différents :
    
    **Algorithmes testés :**
    - **Random Forest** : Modèle d'ensemble idéal pour les relations non-linéaires
    - **XGBoost** : Boosting de gradient pour haute performance
    - **Régression Logistique** : Baseline statistique simple pour comparaison
    
    **Scénarios d'entraînement :**
    - Modèle A : Polluants uniquement
    - Modèle B : **Météo uniquement** ✅ (Réponse directe à la problématique)
    - Modèle C : Météo + Polluants (à titre informatif)
    
    ---
    """)
    
    # --- RÉSULTATS ---
    st.markdown("""
    ### 📊 Résultats Obtenus
    
    #### 🏆 Modèle Retenu : Random Forest (Météo Seule)
    
    **Performances sur l'ensemble de test :**
    """)
    
    # Métriques du meilleur modèle
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", "65.6%", delta="Favorable")
    with col2:
        st.metric("📈 ROC-AUC", "0.86", delta="Excellent")
    with col3:
        st.metric("🎯 Recall Class 1", "87%", delta="Très haut")
    with col4:
        st.metric("⚖️ F1-Score", "Équilibré", delta="Robuste")
    
    st.markdown("""
    **Interprétation :**
    - ✅ Une **accuracy de 65.6%** signifie que le modèle prédit correctement la catégorie de pollution dans 2 cas sur 3
    - ✅ Un **ROC-AUC de 0.86** (excellent) démontre une capacité forte à distinguer les différentes catégories de qualité de l'air
    - ✅ Un **recall de 87%** pour la classe "Good" garantit une détection efficace des zones à air sain (évite les faux négatifs)
    - ✅ Validation croisée stable : Les scores se maintiennent sur différentes partitions des données
    
    #### 📉 Vérification de l'Overfitting
    - Score train : AUC = 1.0 (comportement classique des Random Forest)
    - Score test : AUC = 0.92 → 0.86 (micro) 
    - **Conclusion :** L'overfitting est modéré ; le modèle généralise bien sur données inédites
    
    ---
    """)
    
    # --- FACTEURS DÉTERMINANTS ---
    st.markdown("""
    ### 🔑 Les Leviers Météorologiques de la Pollution
    
    **Variables les plus influentes :**
    1. **🌪️ Vitesse et direction du vent** - Facteur dominant
       - Vent fort disperse les polluants → Meilleure qualité de l'air
       - Vent faible concentre la pollution → Dégradation
    
    2. **☀️ Indice UV** - Influence majeure
       - UV élevé → Réactions photochimiques intensifiées → Ozone secondaire ↑
       - Lien complexe entre radiation solaire et pollution
    
    3. **🌡️ Température**
       - Inversions thermiques froides piègent la pollution en altitude
       - Cumul avec faible vent = conditions critiques
    
    4. **💧 Humidité et visibilité**
       - Indicateurs des conditions stagnantes
       - Corrélées avec accumulation des polluants
    
    5. **☁️ Couverture nuageuse et pression**
       - Influence indirecte via dynamique atmosphérique
    
    ---
    """)
    
    # --- CONCLUSION ---
    st.markdown("""
    ### ✅ CONCLUSION
    
    **Réponse à la problématique :**
    
    **OUI**, les variables météorologiques permettent d'anticiper de manière significative la catégorie globale de qualité de l'air, indépendamment des mesures directes de polluants.
    
    **Principaux enseignements :**
    
    ✨ **Preuve d'efficacité :**
    - Un modèle entraîné uniquement sur des données météo atteint 65.6% de précision prédictive
    - L'AUC de 0.86 valide une excellente capacité de discrimination entre catégories
    - Cela démontre que la météo seule encode suffisamment d'information sur la qualité de l'air
    
    🌍 **Implications pratiques :**
    - Prédiction de pollution sans capteurs directs coûteux
    - Anticipation précoce des pics de pollution (vent faible + UV fort)
    - Adaptation des alertes sanitaires basées sur la météo
    - Aide à la gestion urbaine et à la santé publique
    
    🔬 **Limites acceptables :**
    - ~35% de cas mal classifiés : À attendu dans un domaine complexe
    - Performance concentrée sur détection des zones "saines" (recall 87%)
    - Les phénomènes extrêmes restent plus difficiles à prédire
    
    💡 **Recommandations futures :**
    - Enrichir avec données historiques plus longues
    - Intégrer paramètres régionaux (géographie, industrie locale)
    - Combiner avec mesures directes pour améliorer la performance
    """)
    
    st.divider()
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #1E88E5;'>
    <p><strong>📌 À retenir :</strong> Les météorologues et décideurs publics peuvent désormais s'appuyer sur les prévisions météorologiques pour anticiper les épisodes de pollution, ouvrant la voie à une gestion proactive de la qualité de l'air.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 3. PAGE : SIMULATEUR ---
with tab_sim:
    st.title("🎮 Simulateur de Conditions Météo")
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
        st.subheader(f"Résultat prédit : **{us_map[prediction]}**")

# --- 4. PAGE : PRÉDICTION PAR VILLE ---
with tab_city:
    st.title("🏙️ Prédiction en Temps Réel par Ville")
    
    if not API_KEY:
        st.error("Clé API manquante. Vérifiez votre fichier .env")
    else:
        city = st.text_input("Nom de la ville :", "")
        
        if st.button("Obtenir la météo et prédire"):
            if city:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        w = response.json()
                        weather_data = {
                            'temperature_celsius': w['main']['temp'],
                            'wind_kph': w['wind']['speed'] * 3.6,
                            'wind_degree': w['wind']['deg'],
                            'pressure_mb': w['main']['pressure'],
                            'precip_mm': w.get('rain', {}).get('1h', 0),
                            'humidity': w['main']['humidity'],
                            'cloud': w['clouds']['all'],
                            'feels_like_celsius': w['main']['feels_like'],
                            'visibility_km': w.get('visibility', 10000) / 1000,
                            'uv_index': 5.0, # Valeur par défaut
                            'gust_kph': w['wind'].get('gust', w['wind']['speed']) * 3.6
                        }
                        
                        df_city = pd.DataFrame([weather_data])[features_list]
                        pred_city = model.predict(df_city)[0]
                        
                        st.success(f"Données récupérées pour {city}")
                        c1, c2 = st.columns(2)
                        c1.metric("Température", f"{weather_data['temperature_celsius']} °C")
                        c1.metric("Vent", f"{round(weather_data['wind_kph'], 1)} km/h")
                        c2.subheader(f"Qualité de l'air estimée : **{us_map[pred_city]}**")
                    else:
                        st.error(f"Erreur API ({response.status_code}) : Vérifiez le nom de la ville ou attendez l'activation de la clé.")
                except Exception as e:
                    st.error(f"Erreur technique : {e}")

# 5. Code pour l'onglet Historique
with tab_history:
    st.title("📈 Évolution de la Pollution")
    st.write("Visualisation des données collectées quotidiennement par Airflow.")
    
    try:
        df_hist = pd.read_csv("pollution_history.csv")
        
        # Filtre par ville
        selected_city = st.selectbox("Sélectionnez une ville à analyser :", df_hist['city'].unique())
        df_filtered = df_hist[df_hist['city'] == selected_city]
        
        # Graphique Plotly
        fig = px.line(df_filtered, x='date', y='predicted_epa', 
                      title=f"Évolution de l'indice EPA à {selected_city}",
                      labels={'predicted_epa': 'Indice Pollution (1=Bon, 6=Dangereux)', 'date': 'Date'})
        
        # Ajout de zones de couleurs (optionnel pour le look)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_filtered)
        
    except FileNotFoundError:
        st.info("L'historique est vide. Le pipeline Airflow doit être lancé pour générer les premières données.")