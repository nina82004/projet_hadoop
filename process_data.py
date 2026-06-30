import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when

def main():
    print("Initialisation de SparkSession...")
    spark = SparkSession.builder \
        .appName("Network Intrusion Detection") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    dataset_dir = "datasets"
    output_path = "unified_dataset.parquet"
    
    print(f"Chargement des fichiers CSV depuis {dataset_dir}...")
    # Lire tous les fichiers CSV dans le dossier
    df = spark.read.csv(f"{dataset_dir}/*.csv", header=True, inferSchema=True)
    
    print("--- Statistiques Descriptives Basiques ---")
    total_count = df.count()
    print(f"Nombre total de lignes : {total_count}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    
    # Nettoyage basique des noms de colonnes (espaces au début/fin)
    print("Nettoyage des noms de colonnes...")
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())
        
    print("\nRépartition des labels (classes) :")
    df.groupBy("Label").count().orderBy("count", ascending=False).show(truncate=False)
    
    print("\nRecherche de valeurs manquantes (NaN ou Null) dans quelques colonnes clés...")
    # Checking for missing values in a few columns to avoid massive output
    cols_to_check = df.columns[:5] + ["Label"]
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in cols_to_check]).show()

    print("\nSauvegarde du dataset unifié au format Parquet...")
    df.write.mode("overwrite").parquet(output_path)
    print(f"Dataset unifié sauvegardé dans {output_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()
