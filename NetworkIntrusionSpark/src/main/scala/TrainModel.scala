import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

object TrainModel {
  def main(args: Array[String]): Unit = {
    println("Initialisation de SparkSession pour l'entraînement ML (Scala)...")
    val spark = SparkSession.builder()
      .appName("Network Intrusion ML - RandomForest (Scala)")
      .config("spark.driver.memory", "4g")
      .master("local[*]")
      .getOrCreate()
      
    import spark.implicits._

    val inputPath = "../unified_dataset.parquet"

    println(s"Chargement des données depuis $inputPath...")
    val df = spark.read.parquet(inputPath)

    println("Nettoyage des données (suppression des valeurs manquantes/infinies)...")
    val dfClean = df.na.drop()

    println("Échantillonnage à 10% pour un entraînement rapide...")
    var dfSample = dfClean.sample(withReplacement = false, fraction = 0.1, seed = 42)

    val featureCols = dfSample.columns.filter(_ != "Label")

    for (c <- featureCols) {
      dfSample = dfSample.withColumn(c, col(c).cast("double"))
    }
    
    dfSample = dfSample.na.fill(0.0)

    println("Remplacement des valeurs infinies et NaN par 0.0...")
    for (c <- featureCols) {
      dfSample = dfSample.withColumn(
        c, 
        when(isnan(col(c)) || col(c).isNull || col(c) === Double.PositiveInfinity || col(c) === Double.NegativeInfinity, 0.0)
          .otherwise(col(c))
      )
    }

    println("Configuration du Pipeline ML (VectorAssembler -> StringIndexer -> RandomForest)...")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("rawFeatures")
      .setHandleInvalid("skip")

    val scaler = new StandardScaler()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false)

    val indexer = new StringIndexer()
      .setInputCol("Label")
      .setOutputCol("labelIndex")
      .setHandleInvalid("skip")

    val rf = new RandomForestClassifier()
      .setLabelCol("labelIndex")
      .setFeaturesCol("features")
      .setNumTrees(20)
      .setMaxDepth(5)
      .setSeed(42)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, indexer, rf))

    println("Séparation des données en Train (70%) et Test (30%)...")
    val Array(trainData, testData) = dfSample.randomSplit(Array(0.7, 0.3), seed = 42)

    println("Entraînement du modèle (cela peut prendre quelques minutes)...")
    val model = pipeline.fit(trainData)

    println("Génération des prédictions sur le jeu de test...")
    val predictions = model.transform(testData)

    val evaluatorAcc = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val accuracy = evaluatorAcc.evaluate(predictions)
    val f1Score = evaluatorF1.evaluate(predictions)

    println("\n" + "=" * 50)
    println("RÉSULTATS DE L'ÉVALUATION (Scala)")
    println("=" * 50)
    println(f"Accuracy : $accuracy%.4f")
    println(f"F1-Score : $f1Score%.4f")

    println("\nAperçu des prédictions :")
    predictions.select("Label", "labelIndex", "prediction", "probability").show(10, truncate = false)

    spark.stop()
  }
}
