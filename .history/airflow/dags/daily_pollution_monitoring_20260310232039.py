from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import requests
import joblib
import os
from dotenv import load_dotenv

# --- CONFIGURATION DES CHEMINS ---
# Utilise des chemins absolus pour éviter les erreurs "File Not Found"
BASE_DIR = "C:/Users/LéaGravellard/Desktop/Mémoire_new" # <--- VÉRIFIE CE CHEMIN
MODEL_PATH = os.path.join(BASE_DIR, "Notebooks", "rf_meteo_model.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "pollution_history.csv")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Chargement de la clé API depuis le .env
load_dotenv(ENV_PATH)
API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = ["Paris", "Tokyo", "New York", "London", "Beijing", "Dakar", "Mumbai", "Sao Paulo", "Sydney", "Berlin"]

def fetch_and_predict():
    # 1. Chargement du modèle
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    features_list = model_data["features"]
    
    results = []
    for city in CITIES:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            w = response.json()
            
            # 2. Préparation des données (Doit être IDENTIQUE à app.py)
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
            
            # 3. Prédiction
            df_input = pd.DataFrame([weather_data])[features_list]
            prediction = model.predict(df_input)[0]
            
            # 4. Ajout des infos de suivi
            weather_data['city'] = city
            weather_data['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            weather_data['predicted_epa'] = prediction
            results.append(weather_data)
    
    # 5. Sauvegarde dans le CSV
    if results:
        new_data = pd.DataFrame(results)
        if not os.path.isfile(HISTORY_FILE):
            new_data.to_csv(HISTORY_FILE, index=False)
        else:
            new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# --- DÉFINITION DU DAG ---
default_args = {
    'owner': 'lea',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_pollution_ingestion',
    default_args=default_args,
    description='Collecte météo et prédiction pollution quotidienne',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False
) as dag:

    task_ingest = PythonOperator(
        task_id='fetch_weather_and_predict',
        python_callable=fetch_and_predict
    )