import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, isnan, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

def main():
    # Chemins HDFS par défaut (les données sont stockées dans HDFS, pas en local)
    parser = argparse.ArgumentParser(description="Traitement + nettoyage des données CIC-IDS-2017 depuis HDFS")
    parser.add_argument("--input", default="hdfs:///user/projet/cicids2017",
                        help="Dossier HDFS contenant les CSV bruts")
    parser.add_argument("--output", default="hdfs:///user/projet/unified_dataset.parquet",
                        help="Chemin HDFS du Parquet unifié (brut)")
    parser.add_argument("--cleaned-output", default="hdfs:///user/projet/cleaned_dataset.parquet",
                        help="Chemin HDFS du Parquet nettoyé")
    parser.add_argument("--corr-threshold", type=float, default=0.9,
                        help="Seuil de corrélation au-delà duquel une colonne est jugée redondante")
    args = parser.parse_args()

    print("Initialisation de SparkSession...")
    spark = SparkSession.builder \
        .appName("Network Intrusion Detection") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print(f"Chargement des fichiers CSV depuis {args.input}...")
    # on lit tt les fichiers csv dans le dossier HDFS puis création un DataFrame unique
    df = spark.read.csv(f"{args.input}/*.csv", header=True, inferSchema=True)

    print("Nettoyage des noms de colonnes...")
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())

    print("--- Statistiques Descriptives Basiques ---")
    total_count = df.count()
    n_cols_init = len(df.columns)
    print(f"Nombre total de lignes : {total_count}")
    print(f"Nombre de colonnes : {n_cols_init}")

    print("\nRépartition des labels (classes) :")
    df.groupBy("Label").count().orderBy("count", ascending=False).show(truncate=False)

    print("\nRecherche de valeurs manquantes (NaN ou Null) dans quelques colonnes clés...")
    cols_to_check = df.columns[:5] + ["Label"]
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in cols_to_check]).show()

    print(f"\nSauvegarde du dataset unifié BRUT au format Parquet dans {args.output}...")
    df.write.mode("overwrite").parquet(args.output)

    # =========================================================
    #                    NETTOYAGE DU DATASET
    # =========================================================
    print("\n" + "=" * 55)
    print("NETTOYAGE DU DATASET")
    print("=" * 55)

    label_col = "Label"
    feature_cols = [c for c in df.columns if c != label_col]

    # 1. Conversion en numérique + neutralisation des valeurs infinies / NaN (-> null)
    print("Conversion en numérique et neutralisation des valeurs infinies/NaN...")
    clean = df
    for c in feature_cols:
        clean = clean.withColumn(c, col(c).cast("double"))
    for c in feature_cols:
        clean = clean.withColumn(
            c,
            when(isnan(col(c)) | (col(c) == float("inf")) | (col(c) == float("-inf")), None).otherwise(col(c)),
        )

    # 2. Suppression des lignes dupliquées
    print("Suppression des lignes dupliquées...")
    clean = clean.dropDuplicates()
    clean.cache()
    n_after_dedup = clean.count()
    print(f"  -> {total_count - n_after_dedup} lignes dupliquées supprimées")

    # 3. Détection des colonnes constantes (écart-type nul) sur un échantillon de 10%
    print("Détection des colonnes constantes (variance nulle)...")
    sample = clean.sample(fraction=0.1, seed=42).cache()
    stddev_row = sample.select([F.stddev(col(c)).alias(c) for c in feature_cols]).collect()[0]
    const_cols = [c for c in feature_cols if stddev_row[c] is None or stddev_row[c] == 0.0]
    print(f"  -> {len(const_cols)} colonnes constantes : {const_cols}")
    remaining = [c for c in feature_cols if c not in const_cols]

    # 4. Détection des colonnes redondantes (corrélation > seuil) sur le même échantillon
    print(f"Détection des colonnes redondantes (|corr| > {args.corr_threshold})...")
    assembler = VectorAssembler(inputCols=remaining, outputCol="__features", handleInvalid="skip")
    vec = assembler.transform(sample.select(remaining))
    corr_matrix = Correlation.corr(vec, "__features", "pearson").collect()[0][0].toArray()
    redundant = set()
    for i in range(len(remaining)):
        for j in range(i + 1, len(remaining)):
            if remaining[i] not in redundant and remaining[j] not in redundant:
                if abs(corr_matrix[i][j]) > args.corr_threshold:
                    redundant.add(remaining[j])  # on garde la 1re, on supprime la 2e
    print(f"  -> {len(redundant)} colonnes redondantes : {sorted(redundant)}")
    kept = [c for c in remaining if c not in redundant]

    # 5. Application : suppression des colonnes + lignes invalides sur le dataset complet
    print("Suppression des colonnes inutiles et des lignes invalides...")
    to_drop = const_cols + list(redundant)
    clean = clean.drop(*to_drop)
    clean = clean.filter(col(label_col).isNotNull())
    clean = clean.dropna(subset=kept)
    n_final = clean.count()
    print(f"  -> {n_after_dedup - n_final} lignes invalides supprimées")

    # Bilan
    print("\n" + "-" * 55)
    print("BILAN DU NETTOYAGE")
    print("-" * 55)
    print(f"Lignes   : {total_count:>12,} -> {n_final:>12,}")
    print(f"Colonnes : {n_cols_init:>12} -> {len(clean.columns):>12}  ({len(to_drop)} supprimées)")

    print(f"\nSauvegarde du dataset NETTOYÉ dans {args.cleaned_output}...")
    clean.write.mode("overwrite").parquet(args.cleaned_output)
    print("Dataset nettoyé sauvegardé.")

    spark.stop()

if __name__ == "__main__":
    main()
