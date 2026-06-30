import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def main():
    print("Initialisation de SparkSession pour l'entraînement ML...")
    spark = SparkSession.builder \
        .appName("Network Intrusion ML - RandomForest") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    input_path = "unified_dataset.parquet"
    
    print(f"Chargement des données depuis {input_path}...")
    df = spark.read.parquet(input_path)
    
    # 1. Nettoyage des données
    print("Nettoyage des données (suppression des valeurs manquantes/infinies)...")
    # Pour simplifier, on supprime les lignes ayant des valeurs manquantes dans cette ébauche
    df_clean = df.dropna()
    
    # Beaucoup de datasets réseau ont des valeurs infinies dans Flow Bytes/s et Flow Packets/s
    # On va les filtrer s'il y en a. En PySpark, Infinity peut être traité.
    # Dans ce script simple, on suppose que `dropna` fait le gros du travail.
    
    # Pour un premier test rapide, prenons un échantillon de 10% (car 2.8M de lignes peut être long localement)
    print("Échantillonnage à 10% pour un entraînement rapide (à modifier pour le modèle final)...")
    df_sample = df_clean.sample(withReplacement=False, fraction=0.1, seed=42)
    
    # 2. Préparation des Features
    # Toutes les colonnes sauf "Label" sont des features.
    feature_cols = [c for c in df_sample.columns if c != "Label"]
    
    # Certains types peuvent être "string", on doit s'assurer que les features sont numériques.
    # Pour simplifier, on va caster toutes les features en Double.
    for c in feature_cols:
        df_sample = df_sample.withColumn(c, col(c).cast("double"))
        
    df_sample = df_sample.fillna(0.0) # Sécurité pour les casts échoués
    
    # Nettoyage robuste des valeurs infinies et NaN qui font planter StandardScaler
    print("Remplacement des valeurs infinies et NaN par 0.0...")
    for c in feature_cols:
        df_sample = df_sample.withColumn(c, when(isnan(col(c)) | col(c).isNull() | (col(c) == float('inf')) | (col(c) == float('-inf')), 0.0).otherwise(col(c)))
        
    print("Configuration du Pipeline ML (VectorAssembler -> StringIndexer -> RandomForest)...")
    
    # Assembler les features en un vecteur
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="rawFeatures", handleInvalid="skip")
    
    # Mettre à l'échelle (Scaler) - Recommandé bien que RF y soit moins sensible
    scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=False)
    
    # Encoder le Label textuel en index numérique (0.0, 1.0, 2.0...)
    indexer = StringIndexer(inputCol="Label", outputCol="labelIndex", handleInvalid="skip")
    
    # Modèle Random Forest
    # Limité à 20 arbres pour l'ébauche pour que ça soit rapide
    rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="features", numTrees=20, maxDepth=5, seed=42)
    
    pipeline = Pipeline(stages=[assembler, scaler, indexer, rf])
    
    # 3. Séparation Train/Test
    print("Séparation des données en Train (70%) et Test (30%)...")
    train_data, test_data = df_sample.randomSplit([0.7, 0.3], seed=42)
    
    # 4. Entraînement
    print("Entraînement du modèle (cela peut prendre quelques minutes)...")
    model = pipeline.fit(train_data)
    
    # 5. Prédictions et Évaluation
    print("Génération des prédictions sur le jeu de test...")
    predictions = model.transform(test_data)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ÉVALUATION (Ébauche)")
    print("="*50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1_score:.4f}")
    
    print("\nAperçu des prédictions :")
    predictions.select("Label", "labelIndex", "prediction", "probability").show(10, truncate=False)
    
    spark.stop()

if __name__ == "__main__":
    main()
