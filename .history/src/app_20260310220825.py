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
    st.title("Synthèse de l'Étude")
    
    # --- PROBLÉMATIQUE ---
    st.markdown("""
    ### Problématique
    **Dans quelle mesure les variables météorologiques permettent-elles d'anticiper la catégorie globale de qualité de l'air, indépendamment des mesures directes de polluants ?**
    
    ---
    """)

    # --- SECTION : CONTEXTE ET ENJEUX ---
    st.markdown("""
    ### Contexte et Enjeux
    La surveillance de la qualité de l'air repose traditionnellement sur des stations de mesure au sol équipées de capteurs chimiques onéreux. Cependant, ces infrastructures sont souvent limitées géographiquement.

    **Pourquoi ce sujet est-il crucial ?**
    1.  **Santé Publique :** La pollution de l'air est un facteur majeur de maladies respiratoires. Anticiper les pics permet de protéger les populations vulnérables.
    2.  **Accessibilité des données :** Les données météorologiques (vent, température, UV) sont disponibles partout, en temps réel et à bas coût, contrairement aux mesures de micro-particules.
    3.  **Aide à la décision :** Comprendre le lien entre météo et pollution permet aux autorités de réguler le trafic ou les activités industrielles avant même que les seuils critiques ne soient atteints.
    
    **L'hypothèse de l'étude :** La météo n'est pas qu'un simple accompagnement, c'est le **moteur physique** de la pollution (transport, stagnation ou réaction chimique).
    
    ---
    """)
    
    # --- MÉTHODOLOGIE ---
    st.markdown("""
    ### Méthodologie
    
    #### Source et Préparation des Données
    - **Origine :** Le dataset utilisé provient de la plateforme **Kaggle** (*Global Weather Repository*).
    - **Variables :** Il regroupe des paramètres météo complets et des mesures de polluants (CO, O₃, NO₂, SO₂, PM2.5, PM10) pour des milliers de localisations mondiales.
    
    #### tratégie de Modélisation (Pourquoi ce choix ?)
    Pour répondre à la problématique, nous avons testé trois scénarios. Il est crucial de comprendre pourquoi seul le modèle **"Météo uniquement"** a une valeur prédictive réelle :
    
    1.  **Scénario A : Polluants uniquement**
        * *Observation :* Ce modèle obtient un score parfait (**ROC-AUC = 1.0**).
        * *Explication :* C'est un résultat **trivial**. L'indice de qualité de l'air (cible) est mathématiquement calculé à partir des concentrations de ces mêmes polluants. Le modèle ne fait que réapprendre une formule de calcul déjà existante.
    
    2.  **Scénario B : Météo uniquement (Choix Final)** * *Objectif :* C'est le **cœur du projet**. Ici, le modèle doit "deviner" la pollution sans jamais voir les capteurs chimiques. 
        * *Utilité :* C'est le seul scénario qui permet une **anticipation réelle** dans des zones non équipées de capteurs de pollution.
    
    3.  **Scénario C : Mix Météo + Polluants** * *Observation :* Très performant, mais **inutile en pratique**. 
        * *Explication :* Si nous devons déjà posséder des capteurs de polluants pour faire fonctionner le modèle, l'utilisation de la météo perd son intérêt principal (qui est de s'affranchir de ces capteurs).
    
    ---
    """)
    
    # --- RÉSULTATS ---
    st.markdown("""
    ### Résultats Obtenus
    
    #### Modèle Gagnant : Random Forest (Scénario Météo Seule)
    *Pourquoi cet algorithme ?* Le Random Forest a démontré la meilleure capacité à capturer les interactions non-linéaires (ex: l'effet combiné du fort ensoleillement et du vent faible).
    """)

    
    
    # Métriques du meilleur modèle
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", "65.6%", delta="Stable")
    with col2:
        st.metric("📈 ROC-AUC", "0.86", delta="Fort")
    with col3:
        st.metric("🎯 Recall Class 1", "87%", delta="Optimal")
    with col4:
        st.metric("⚖️ F1-Score", "0.79", delta="Robuste")
    
    st.markdown("""
    **Interprétation technique :**
    - L'**Accuracy (65.6%)** indique une bonne fiabilité globale sur un problème complexe à 6 classes.
    - Le **ROC-AUC (0.86)** confirme que le modèle possède un fort pouvoir de discrimination entre les différents niveaux de pollution.
    - Le **Recall élevé (87%)** sur l'air sain est crucial : le modèle identifie très bien les moments où il n'y a aucun risque respiratoire.
    
    ---
    """)
    
    # --- FACTEURS DÉTERMINANTS ---
    st.markdown("""
    ### Les Leviers Météorologiques de la Pollution
    
    Selon l'importance des variables du modèle (**Gini Importance**), voici les facteurs qui dictent la qualité de l'air :
    1.  **Indice UV :** Le principal catalyseur. Le rayonnement solaire transforme les gaz en ozone de basse altitude.
    2.  **Direction et Vitesse du vent :** Le facteur de transport et de dispersion.
    3.  **Température et Humidité :** Ils influent sur la stabilité de la masse d'air et le piégeage des particules au sol.
    
    ---
    """)
    
    # --- CONCLUSION ---
    st.markdown("""
    ### ✅ CONCLUSION
    
    **Réponse à la problématique :**
    **OUI**, les variables météorologiques permettent d'anticiper la qualité de l'air. 
    
    En utilisant des données provenant de **Kaggle**, nous avons prouvé qu'un modèle basé sur la physique de l'atmosphère (météo) est une alternative viable et économique aux mesures chimiques directes pour la mise en place de systèmes d'alerte précoce.
    """)
    
    st.divider()
    
    # --- BLOC FINAL CORRIGÉ (Texte en noir) ---
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #1E88E5;'>
    <p style='color: #000000;'><strong>📌 Note :</strong> Le succès du modèle "Météo Seule" (AUC 0.86) valide que les conditions climatiques encodent suffisamment d'informations pour prédire l'état sanitaire de l'air sans aucun capteur chimique.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 3. PAGE : SIMULATEUR ---
with tab_sim:
    st.title("Simulateur de Conditions Météo")

    # --- INTRODUCTION ---
    st.markdown("""
    Bienvenue dans le simulateur interactif. Cette interface vous permet de tester l'influence de chaque paramètre météorologique sur la qualité de l'air. 
    
    **Comment ça marche ?**
    1.  **Ajustez les curseurs** ci-dessous pour créer un scénario météo spécifique (ex: une canicule sans vent ou une journée pluvieuse).
    2.  **Analysez les interactions** : Observez comment le changement d'une seule variable (comme l'indice UV ou l'humidité) fait basculer la prédiction du modèle.
    3.  **Lancez le calcul** : Cliquez sur le bouton en bas de page pour obtenir le diagnostic immédiat de notre intelligence artificielle.
    """)
    st.info("💡 Conseil : Essayez de réduire le vent tout en augmentant l'indice UV pour observer l'apparition d'un pic de pollution simulé.")
    
    st.divider()

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
    st.title("Prédiction en Temps Réel par Ville")

    # --- INTRODUCTION & FONCTIONNEMENT ---
    st.markdown("""
    Cette page permet d'appliquer notre modèle prédictif à des situations réelles actuelles. 
    
    **Comment ça fonctionne ?**
    1. **Extraction :** En saisissant une ville, nous interrogeons l'API **OpenWeatherMap** pour récupérer les conditions atmosphériques en direct (température, vent, humidité, pression, etc.).
    2. **Traitement :** Ces données brutes sont formatées pour correspondre exactement aux entrées attendues par notre modèle *Random Forest*.
    3. **Prédiction :** L'intelligence artificielle analyse ces paramètres météo réels pour estimer la catégorie de pollution (US EPA Index) correspondante à cet instant précis.
    """)
    st.info("💡 Note : Cette approche permet d'estimer la qualité de l'air même dans des villes ne possédant pas de capteurs chimiques onéreux.")
    
    st.divider()
    
    if not API_KEY:
        st.error("Clé API manquante. Vérifiez votre fichier .env")
    else:
        city_input = st.text_input("Nom de la ville (ex: Paris, Tokyo, New York) :", "")
        
        if st.button("Obtenir la météo et prédire"):
            if city_input:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city_input}&appid={API_KEY}&units=metric"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        w = response.json()
                        
                        # Récupération de TOUTES les données nécessaires pour le modèle
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
                            'uv_index': 5.0, # Valeur estimée (non fournie par l'API gratuite)
                            'gust_kph': w.get('wind', {}).get('gust', w['wind']['speed']) * 3.6
                        }
                        
                        # 1. Prédiction (utilise les 11 colonnes)
                        df_city = pd.DataFrame([weather_data])[features_list]
                        pred_city = model.predict(df_city)[0]
                        
                        st.success(f"✅ Données météo complètes récupérées pour {city_input}")
                        
                        # 2. Affichage des résultats en colonnes pour la lisibilité
                        st.markdown("### Paramètres observés")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Température", f"{weather_data['temperature_celsius']} °C")
                        m1.metric("Ressenti", f"{weather_data['feels_like_celsius']} °C")
                        
                        m2.metric("Vent", f"{round(weather_data['wind_kph'], 1)} km/h")
                        m2.metric("Rafales", f"{round(weather_data['gust_kph'], 1)} km/h")
                        
                        m3.metric("Humidité", f"{weather_data['humidity']}%")
                        m3.metric("Pression", f"{weather_data['pressure_mb']} hPa")
                        
                        m4.metric("Nuages", f"{weather_data['cloud']}%")
                        m4.metric("Visibilité", f"{weather_data['visibility_km']} km")

                        # 3. Résultat Final
                        st.markdown("---")
                        st.subheader(f"Qualité de l'air estimée par le modèle :")
                        st.info(f"**{us_map[pred_city]}**")
                        
                    else:
                        st.error(f"Erreur API ({response.status_code}) : Ville non trouvée ou clé invalide.")
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