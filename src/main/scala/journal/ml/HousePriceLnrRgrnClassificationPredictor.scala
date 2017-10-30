package journal.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

object HousePriceLnrRgrnClassificationPredictor {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
                .builder()
                .master("local[4]")
					      .appName("HousePricePredictor")
                .getOrCreate()
    import spark.implicits._
    
    val housePricesDF = spark.read
                          .option("header","true")
                          .option("inferSchema","true")
                          .csv("resources/machinelearning/data/house-prices/house-sales-full.csv")
                          .select($"SalePrice", $"Bedrooms", $"Bathrooms", $"SqFtTotLiving", $"SqFtLot", $"YrBuilt")
                          .withColumnRenamed("SalePrice", "label")
    
    val assembler = new VectorAssembler()
                          .setInputCols(Array("Bedrooms", "Bathrooms", "SqFtTotLiving", "SqFtLot")) //, "YrBuilt"
                          .setOutputCol("features")
                          
    val housePricesFeatureVectorDF = assembler.transform(housePricesDF)
    
    housePricesFeatureVectorDF.show
    
    val lnrRgrn = new LinearRegression()
                          .setMaxIter(200)
                          .setRegParam(0.3)
                          .setElasticNetParam(0.8)
                          
    val housePredictorModel = lnrRgrn.fit(housePricesFeatureVectorDF)
    println("Coefficients : " + housePredictorModel.coefficients)
    println("Intercept : " + housePredictorModel.intercept)
    println("r2 : " + housePredictorModel.summary.r2)
    println("RMSE : " + housePredictorModel.summary.rootMeanSquaredError)
    println("ObjectiveHistory : " + housePredictorModel.summary.objectiveHistory)
    //test data
    val predictionDataDF = spark.sparkContext.parallelize(Seq((5,3,4400,10000,1981),(3,2,1800,5000,1995)
                                    ,(5,3,4400,10000,1995),(3,2,1800,5000,1981)))
                              .toDF("Bedrooms", "Bathrooms", "SqFtTotLiving", "SqFtLot","YrBuilt")
                              
    val predictionDataFeatureVectorDF = assembler.transform(predictionDataDF)
    val predictionDataPriceDF = housePredictorModel.transform(predictionDataFeatureVectorDF)
                                    .withColumnRenamed("prediction", "projected_price")
    predictionDataPriceDF.show
    //predict using orig data
    housePredictorModel.transform(housePricesFeatureVectorDF)
                          .withColumnRenamed("prediction", "projected_price")
                          .withColumnRenamed("label", "orig_price")
                          .show(500)
  }
}