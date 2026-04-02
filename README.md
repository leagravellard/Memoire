# Mémoire — Prédiction de la qualité de l'air par les variables météorologiques

**Léa Gravellard — Mémoire de recherche en Data Science**

---

## Problématique

> *Dans quelle mesure les variables météorologiques permettent-elles d'anticiper la catégorie globale de qualité de l'air, indépendamment des mesures directes de polluants ?*

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
| **Polluants (comparaison uniquement)** | CO, O3, NO2, SO2, PM2.5, PM10 |
| **Variable cible** | `air_quality_us-epa-index` — 6 classes (1 = Bon → 6 = Dangereux), selon l'EPA américaine |

---

## Méthodologie

### Trois scénarios de modélisation

| Scénario | Features | Objectif |
|---|---|---|
| A — Polluants seuls | CO, O3, NO2, SO2, PM2.5, PM10 | Baseline / contrôle |
| **B — Météo seule** | 11 variables météorologiques | **Cœur de l'étude** |
| C — Météo + Polluants | 17 variables combinées | Borne supérieure théorique |

### Algorithmes testés

- **Random Forest** (modèle final retenu)
- **XGBoost**
- **Régression Logistique**

### Validation

- Validation croisée stratifiée à 5 plis
- Analyse du surapprentissage (modèle contraint vs. non contraint)
- Courbes ROC-AUC (train et test)
- Importance des variables (Gini Impurity)

---

## Résultats (Scénario B — Météo seule)

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

---

## Application Streamlit

Le projet est livré avec une application web interactive (`src/app.py`) comportant 4 onglets :

1. **Notebook complet** — l'analyse end-to-end
2. **Synthèse rédigée** — les conclusions de l'étude
3. **Simulateur météo** — sliders pour tester manuellement chaque paramètre
4. **Prédiction en temps réel** — saisir n'importe quelle ville du monde pour obtenir une prédiction basée sur les conditions actuelles (via l'API OpenWeatherMap)

### Lancer l'application

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit streamlit run src/app.py
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
└── README.md
```

---

## Conclusion

Les variables météorologiques seules permettent de prédire la catégorie de qualité de l'air avec un **AUC de 0,86** sur un problème à 6 classes. Ce résultat valide l'hypothèse centrale : la météo peut servir de proxy aux mesures chimiques, rendant possible des systèmes d'alerte accessibles dans les zones sans infrastructure de surveillance de la qualité de l'air.
