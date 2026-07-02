# Projet Hadoop : Network Intrusion Detection (CIC-IDS-2017)

Ce projet implémente un pipeline de traitement Big Data et d'apprentissage automatique via Apache Spark (PySpark) sur Hadoop/HDFS, pour la détection d'intrusions réseau.

## Objectifs
- Unifier de multiples fichiers CSV de trafic réseau (~800 Mo, 2,8 M de lignes).
- Nettoyer le dataset (colonnes constantes/redondantes, doublons, valeurs invalides).
- Analyser et visualiser les données (notebook Jupyter).
- Entraîner un modèle de Machine Learning (Random Forest) pour la classification d'attaques.

## Prérequis
- Hadoop (HDFS + YARN) installé et démarré (testé sous WSL Ubuntu)
- Java, Python 3
- Les 8 CSV CIC-IDS-2017 chargés sur HDFS dans `/user/projet/cicids2017`
- Librairies Python : voir `requirements.txt` (ou installation ci-dessous)

## Fichiers du projet
| Fichier | Rôle |
|---|---|
| `process_data.py` | Fusion + nettoyage → `unified_dataset.parquet` et `cleaned_dataset.parquet` (HDFS) |
| `train_model.py` | Entraînement Random Forest sur le dataset nettoyé → Accuracy / F1 |
| `analyse_resultats.ipynb` | Notebook d'analyse et de visualisation |
| `GUIDE_DEMO.txt` | Aide-mémoire des commandes de démo |

---

## Exécution (pas à pas)

### 0. Installation de l'environnement (une seule fois)
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyspark pandas numpy scikit-learn matplotlib seaborn jupyterlab
```

Charger les données sur HDFS (si ce n'est pas déjà fait) :
```bash
hdfs dfs -mkdir -p /user/projet/cicids2017
hdfs dfs -put *.csv /user/projet/cicids2017
```

### 1. Démarrer Hadoop
```bash
start-dfs.sh      # démarre HDFS (NameNode + DataNode)
start-yarn.sh     # démarre YARN (ResourceManager + NodeManager)
jps               # doit afficher 5 services
```

### 2. Vérifier les données sur HDFS
```bash
hdfs dfs -ls -h /user/projet/cicids2017
```

### 3. Traitement + nettoyage des données (~10 min)
Fusionne les CSV, nettoie (colonnes constantes/redondantes, doublons, valeurs invalides)
et écrit le Parquet brut + le Parquet nettoyé sur HDFS.
```bash
spark-submit --driver-memory 4g process_data.py \
  --input hdfs:///user/projet/cicids2017 \
  --output hdfs:///user/projet/unified_dataset.parquet \
  --cleaned-output hdfs:///user/projet/cleaned_dataset.parquet
```
Vérifier la sortie :
```bash
hdfs dfs -ls -h /user/projet/cleaned_dataset.parquet
```

### 4. Entraînement du modèle
Entraîne un Random Forest sur le dataset **nettoyé** lu depuis HDFS.
```bash
spark-submit --driver-memory 4g train_model.py \
  --input hdfs:///user/projet/cleaned_dataset.parquet
```

### 5. Notebook d'analyse
```bash
jupyter lab --no-browser
```
Copier l'URL `http://localhost:8888/lab?token=...` dans Chrome, ouvrir
`analyse_resultats.ipynb`, puis **Run → Run All Cells**.

---

## Interfaces web à consulter (Chrome)
| URL | Contenu |
|---|---|
| http://localhost:9870 | HDFS — les fichiers stockés |
| http://localhost:8088 | YARN — le job Spark en cours |
| http://localhost:8888 | JupyterLab — le notebook |

## Notes
- `process_data.py` écrit en mode `overwrite` : relancer la commande recalcule et
  remplace les résultats précédents (aucune accumulation).
- Les données (CSV / Parquet) sont sur HDFS, pas dans le dépôt Git (voir `.gitignore`).

## Performances (indicatif)
- Accuracy : ~95 %
- F1-Score : ~95 %
*(Random Forest, échantillon d'entraînement.)*
