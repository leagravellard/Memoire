# Mémoire — Prédiction de la qualité de l'air par les variables météorologiques

**Léa Gravellard — Mémoire de recherche en Data Science**

---

## Problématique

> *Modélisation prédictive de la qualité de l'air : dans quelle mesure le Machine Learning permet-il de prédire la qualité de l'air à partir des seules données météorologiques, indépendamment des mesures directes de polluants ?*


L'objectif est de démontrer que des données météorologiques seules (température, vent, humidité, etc.) peuvent prédire la qualité de l'air, sans recours aux capteurs chimiques coûteux. Cela ouvre la voie à des systèmes d'alerte précoce accessibles partout dans le monde.

---

## Données

- **Source :** [Global Weather Repository — Kaggle](https://www.kaggle.com/)
- **Fichier :** `Dataset/GlobalWeatherRepository.csv`
- **Taille :** ~126 477 lignes, une par ville/horodatage
- **Couverture :** Mondiale (capitales et grandes villes)

### Variables utilisées

| Type | Variables |
|---|---|
| **Entrées météo (features)** | température, vitesse du vent, direction du vent, pression, précipitations, humidité, couverture nuageuse, ressenti thermique, visibilité, indice UV, rafales |
| **Variable cible** | `air_quality_us-epa-index` — 6 classes (1 = Bon → 6 = Dangereux), selon l'EPA américaine |

---

## Méthodologie

### Scénario de modélisation

| Scénario | Features | Objectif |
| ** Variables météorologiques seulement** | 11 variables météorologiques | 
|

### Algorithmes testés

- **Random Forest** (modèle final retenu)
- **XGBoost**
- **Régression Logistique**

### Validation

- Validation croisée stratifiée à 5 folds
- Analyse du surapprentissage (modèle contraint vs. non contraint)
- Courbes ROC-AUC (train et test)
- Importance des variables (Gini Impurity)

---

## Résultats

| Métrique | Valeur |
|---|---|
| Accuracy (test) | **65,6 %** |
| ROC-AUC macro | **0,86** |
| ROC-AUC micro | **0,929** |
| F1-Score | 0,79 |
| Recall classe 1 (Bon) | 87 % |
| Cross-val accuracy (5-fold) | 0,6565 ± 0,0011 |

**Variables météo les plus importantes :**
1. Indice UV (~12,5 %) — réactions photochimiques atmosphériques
2. Direction et vitesse du vent — dispersion des polluants
3. Température et humidité — stabilité des masses d'air

### Le notebook en HTML

Le notebook est intégré dans l'application Streamlit sous forme de fichier HTML (`Notebooks/01_exploration.html`), ce qui permet de l'afficher directement dans le navigateur sans dépendance à Jupyter. Après chaque mise à jour du notebook `.ipynb`, régénérer le HTML avec :

```bash
jupyter nbconvert --to html Notebooks/01_exploration.ipynb
```

---

## Application Streamlit

Le projet est livré avec une application web interactive (`src/app.py`) comportant 4 onglets :

1. **Notebook complet** — l'analyse end-to-end
2. **Synthèse rédigée** — les conclusions de l'étude
3. **Simulateur météo** — sliders pour tester manuellement chaque paramètre
4. **Prédiction en temps réel** — saisir n'importe quelle ville du monde pour obtenir une prédiction basée sur les conditions actuelles (via l'API OpenWeatherMap)

### Dépendances principales

| Package | Version | Usage |
|---|---|---|
| `streamlit` | 1.55.0 | Application web interactive |
| `scikit-learn` | 1.8.0 | Random Forest, Régression Logistique, métriques |
| `xgboost` | 3.2.0 | XGBoost classifier |
| `pandas` | 2.3.3 | Manipulation des données |
| `numpy` | 2.4.2 | Calcul numérique |
| `matplotlib` | 3.10.8 | Visualisations statiques |
| `seaborn` | 0.13.2 | Heatmaps et distributions |
| `plotly` | 6.5.2 | Graphiques interactifs |
| `shap` | 0.51.0 | Interprétabilité des modèles |
| `mlflow` | 3.10.0 | Suivi des expériences |
| `requests` | 2.32.3 | Appels API OpenWeatherMap |

> Le fichier `requirements.txt` contient l'environnement complet (`pip freeze`).

### Lancer l'application

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit run src/app.py
```

---

## Structure du projet

```
Memoire/
├── Dataset/
│   └── GlobalWeatherRepository.csv       # Données brutes (~126k lignes)
├── Notebooks/
│   ├── 01_exploration.ipynb              # Notebook principal (analyse complète)
│   ├── rf_meteo_model.pkl                # Modèle Random Forest sauvegardé
│   └── Visualisations/                   # Graphiques exportés (PNG)
├── src/
│   └── app.py                            # Application Streamlit
├── venv/                                 # Environnement virtuel Python
├── README.md
└── Mémoire.pdf                           # Mémoire (PDF)

```

