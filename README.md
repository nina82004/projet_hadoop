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

Les données brutes sont stockées sur **HDFS** dans `/user/projet/cicids2017` (8 CSV, ~800 Mo).

### 2. Traitement des données
Fusionne les CSV bruts depuis HDFS vers un Parquet unifié sur HDFS.
```bash
spark-submit --driver-memory 4g process_data.py \
  --input hdfs:///user/projet/cicids2017 \
  --output hdfs:///user/projet/unified_dataset.parquet
```

### 3. Entraînement du modèle
Entraîne un modèle Random Forest sur les données unifiées lues depuis HDFS.
```bash
spark-submit --driver-memory 4g train_model.py \
  --input hdfs:///user/projet/unified_dataset.parquet
```

## Performances
- Accuracy : 95.96%
- F1-Score : 95.20%
*(Entraîné sur un échantillon de 10%)*
