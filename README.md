# Projet Hadoop : Network Intrusion Detection (CIC-IDS-2017)

Ce projet implémente un pipeline de traitement Big Data et d'apprentissage automatique via Apache Spark (PySpark) pour la détection d'intrusions réseau.

## Objectifs
- Unifier et nettoyer de multiples fichiers CSV de trafic réseau (~800 Mo).
- Entraîner un modèle de Machine Learning (Random Forest) pour la classification d'attaques.

## Prérequis
- Java 17
- Python 3
- Apache Spark / PySpark

## Exécution

### 1. Configuration
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyspark pandas scikit-learn
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
```

### 2. Traitement des données
Fusionne les CSV bruts depuis `datasets/` vers `unified_dataset.parquet`.
```bash
python3 process_data.py
```

### 3. Entraînement du modèle
Entraîne un modèle Random Forest sur les données unifiées.
```bash
python3 train_model.py
```

## Performances
- Accuracy : 95.96%
- F1-Score : 95.20%
*(Entraîné sur un échantillon de 10%)*
