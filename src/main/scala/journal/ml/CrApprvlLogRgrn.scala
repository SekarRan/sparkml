package journal.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary

object CrApprvlLogRgrn {
  
  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.functions._ 
    val spark = SparkSession
                .builder()
                .master("local[4]")
					      .appName("CrApprvlLogRgrn")
                .getOrCreate()
    
    val mrtgeDataDF = spark.read
                          .option("header","true")
                          .option("inferSchema","true")
                          .csv("resources/machinelearning/data/mortgage-applications/mortgage-applications.csv")
                          .withColumnRenamed("approved", "label")
    mrtgeDataDF.show                    
    val assembler = new VectorAssembler()
                          .setInputCols(Array("fico"))
                          .setOutputCol("features")
                          
    val logRgrn = new LogisticRegression()
    
    val mrtgApplnPipeLine = new Pipeline().setStages(Array(assembler,logRgrn))
    
    val mrtgApplnModel = mrtgApplnPipeLine.fit(mrtgeDataDF)
    
    val model = mrtgApplnModel.stages.last.asInstanceOf[LogisticRegressionModel]
    
    println("Intercept : " + model.intercept)
    println("Coefficients : " + model.coefficients)
    
    val trainSummary = model.summary.asInstanceOf[BinaryLogisticRegressionSummary]
    println("AUC : " + trainSummary.areaUnderROC)
    trainSummary.roc.show
    
    val mrtgePredictionDataDF = mrtgApplnModel.transform(mrtgeDataDF)
    
    mrtgePredictionDataDF.show(100)
  }
}