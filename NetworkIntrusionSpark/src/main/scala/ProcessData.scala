import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ProcessData {
  def main(args: Array[String]): Unit = {
    println("Initialisation de SparkSession (Scala)...")
    val spark = SparkSession.builder()
      .appName("Network Intrusion Detection (Scala)")
      .config("spark.driver.memory", "4g")
      .master("local[*]")
      .getOrCreate()
      
    import spark.implicits._

    val datasetDir = "../datasets"
    val outputPath = "../unified_dataset_scala.parquet"

    println(s"Chargement des fichiers CSV depuis $datasetDir...")
    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(s"$datasetDir/*.csv")

    println("--- Statistiques Descriptives Basiques ---")
    val totalCount = df.count()
    println(s"Nombre total de lignes : $totalCount")
    println(s"Nombre de colonnes : ${df.columns.length}")

    println("Nettoyage des noms de colonnes...")
    for (colName <- df.columns) {
      df = df.withColumnRenamed(colName, colName.trim)
    }

    println("\nRépartition des labels (classes) :")
    df.groupBy("Label").count().orderBy($"count".desc).show(truncate = false)

    println("\nSauvegarde du dataset unifié au format Parquet...")
    df.write.mode("overwrite").parquet(outputPath)
    println(s"Dataset unifié sauvegardé dans $outputPath")

    spark.stop()
  }
}
