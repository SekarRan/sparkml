package journal.ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.evaluation.RegressionEvaluator


object ComplexTipLnrRgrnWithPipeLinePredictor {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
                .builder()
                .master("local[4]")
					      .appName("ComplexTipLnrRgrnWithPipeLinePredictor")
                .getOrCreate()
    
    val tipDataDF = spark.read
                          .option("header","true")
                          .option("inferSchema","true")
                          .csv("resources/machinelearning/data/tips/tips.csv")
                          .withColumnRenamed("tip", "label")
    val sexIndexer = new StringIndexer()
                          .setInputCol("sex")
                          .setOutputCol("sexIndex")

    val dayIndexer = new StringIndexer()
                          .setInputCol("day")
                          .setOutputCol("dayIndex")
   
    val assembler = new VectorAssembler()
                          .setInputCols(Array("total_bill", "sexIndex", "dayIndex", "size"))
                          .setOutputCol("features")

    val lnrRgrn = new LinearRegression()
                          .setMaxIter(200)
                          .setRegParam(0.3)
                          .setElasticNetParam(0.8)
                          .setWeightCol("total_bill")
    /*
    val paramGrid = new ParamGridBuilder().addGrid(lnrRgrn.regParam, Array(0.1, 0.3, 0.5, 1.0))
                                .addGrid(lnrRgrn.fitIntercept)
                                .addGrid(lnrRgrn.elasticNetParam, Array(0.0, 0.5, 0.8, 1.0))
                                .addGrid(lnrRgrn.maxIter, Array(10, 50, 100, 150, 200))
                                //.addGrid(lnrRgrn.weightCol, Array ("total_bill", "sexIndex", "dayIndex", "size"))
                                .build 
    
    val tipCV = new CrossValidator()
                          .setEstimator(lnrRgrn)
                          .setEvaluator(new BinaryClassificationEvaluator)
                          .setEstimatorParamMaps(paramGrid)
                          .setNumFolds(6)*/
                          
    val tipCVpipeline = new Pipeline()
                          .setStages(Array(sexIndexer, dayIndexer, assembler, lnrRgrn))
                          
    val tipPredictorModel = tipCVpipeline.fit(tipDataDF)
    
    println("Coefficients : " + tipPredictorModel.stages.last.asInstanceOf[LinearRegressionModel].coefficients)
    println("Intercept : " + tipPredictorModel.stages.last.asInstanceOf[LinearRegressionModel].intercept)
    println("r2 : " + tipPredictorModel.stages.last.asInstanceOf[LinearRegressionModel].summary.r2)
    println("RMSE : " + tipPredictorModel.stages.last.asInstanceOf[LinearRegressionModel].summary.rootMeanSquaredError)
    /*val tipBestModel = tipPredictorModel.stages.last
                              .asInstanceOf[CrossValidatorModel]
                              .bestModel
                              .asInstanceOf[LinearRegressionModel]*/
    //predict using orig data
    val predictionDF = tipPredictorModel.transform(tipDataDF)

    predictionDF.show(500)
    val evaluator = new RegressionEvaluator().setMetricName("r2")
    val r2 = evaluator.evaluate(predictionDF)
    println("Tip Model r2 : " + r2)
    println("Tip Model param map : " + evaluator.extractParamMap)
  }
}