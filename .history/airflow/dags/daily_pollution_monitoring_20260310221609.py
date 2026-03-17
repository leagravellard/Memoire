from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import requests
import joblib
import os

# Liste des 10 villes choisies
CITIES = ["Paris", "Tokyo", "New York", "London", "Beijing", "Dakar", "Mumbai", "Sao Paulo", "Sydney", "Berlin"]
MODEL_PATH = "/chemin/vers/votre/rf_meteo_model.pkl"
HISTORY_FILE = "/chemin/vers/votre/pollution_history.csv"

def fetch_and_predict():
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    features_list = model_data["features"]
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    
    results = []
    for city in CITIES:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url).json()
        
        if "main" in resp:
            # Transformation identique à app.py
            weather_data = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'city': city,
                'temperature_celsius': resp['main']['temp'],
                'wind_kph': resp['wind']['speed'] * 3.6,
                'humidity': resp['main']['humidity'],
                'uv_index': 5.0 # Valeur par défaut
                # ... ajouter les autres variables nécessaires ...
            }
            
            # Prédiction
            df_input = pd.DataFrame([weather_data])[features_list]
            weather_data['predicted_epa'] = model.predict(df_input)[0]
            results.append(weather_data)
    
    # Stockage dans le CSV historique
    new_data = pd.DataFrame(results)
    if not os.path.isfile(HISTORY_FILE):
        new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# Définition du DAG
with DAG(
    'daily_pollution_ingestion',
    default_args={'retries': 1},
    schedule_interval='@daily', # Exécution tous les jours
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    task_ingest = PythonOperator(
        task_id='fetch_weather_and_predict',
        python_callable=fetch_and_predict
    )