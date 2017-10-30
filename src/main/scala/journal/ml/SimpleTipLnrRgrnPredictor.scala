package journal.ml

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object SimpleTipLnrRgrnPredictor {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
                .builder()
                .master("local[4]")
					      .appName("SimpleTipLnrRgrnPredictor")
                .getOrCreate()
    
    val tipDataDF = spark.read
                          .option("header","true")
                          .option("inferSchema","true")
                          .csv("resources/machinelearning/data/tips/tips.csv")
                          .select("total_bill","tip")
                          .withColumnRenamed("tip", "label")
    
    val assembler = new VectorAssembler()
                          .setInputCols(Array("total_bill"))
                          .setOutputCol("features")
    val tipFeatureVectorDF = assembler.transform(tipDataDF)                      

    val lnrRgrn = new LinearRegression()
                          .setMaxIter(200)
                          .setRegParam(0.3)
                          .setElasticNetParam(0.8)
    val tipPredictorModel = lnrRgrn.fit(tipFeatureVectorDF)

    println("Coefficients : " + tipPredictorModel.coefficients)
    println("Intercept : " + tipPredictorModel.intercept)
    println("r2 : " + tipPredictorModel.summary.r2)
    println("RMSE : " + tipPredictorModel.summary.rootMeanSquaredError)
    println("Simple Predictor model params : " + tipPredictorModel.parent.extractParamMap)
    println("Simple Predictor model param Explain : " + tipPredictorModel.parent.explainParams)
    //predict using orig data
    tipPredictorModel.transform(tipFeatureVectorDF)
                          .withColumnRenamed("prediction", "predicted_tip")
                          .withColumnRenamed("label", "act_tip")
                          .show(500)
    
  }
}