package journal.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

object ComplexLoanDecisionTreeClassification {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.functions._ 
    
    val spark = SparkSession
                .builder()
                .master("local[4]")
					      .appName("ComplexLoanDecisionTreeClassification")
                .getOrCreate()
    import spark.implicits._
    val loanDataDF = spark.read
                          .option("header","true")
                          .option("inferSchema","true")
                          .csv("resources/machinelearning/data/prosper-loan/prosper-loan-data.csv.gz")
    loanDataDF.show
    val splits = loanDataDF.randomSplit(Array(0.7, 0.3))
    val loantrainingDataDF = splits(0).toDF.cache
    val loantestDataDF = splits(1).toDF
    
    val columns = Array("Term", "BorrowerRate", "ProsperRating (numeric)", "ProsperScore", "EmploymentStatusDuration", "IsBorrowerHomeowner",
           "CreditScore", "CurrentCreditLines", "OpenCreditLines",
           "TotalCreditLinespast7years", "OpenRevolvingAccounts", "OpenRevolvingMonthlyPayment",
           "InquiriesLast6Months", "TotalInquiries", "CurrentDelinquencies", "AmountDelinquent",
           "DelinquenciesLast7Years", "PublicRecordsLast10Years", "PublicRecordsLast12Months",
           "RevolvingCreditBalance", "BankcardUtilization", "AvailableBankcardCredit", "TotalTrades",
           "TradesNeverDelinquent (percentage)", "TradesOpenedLast6Months", "DebtToIncomeRatio",
           "IncomeVerifiable", "StatedMonthlyIncome", "TotalProsperLoans", "TotalProsperPaymentsBilled",
           "OnTimeProsperPayments", "ProsperPaymentsLessThanOneMonthLate", "ProsperPaymentsOneMonthPlusLate",
           "ProsperPrincipalBorrowed", "ProsperPrincipalOutstanding", "LoanOriginalAmount",
           "MonthlyLoanPayment", "Recommendations", "InvestmentFromFriendsCount", "InvestmentFromFriendsAmount",
           "Investors", "YearsWithCredit")
    val categoricalColumns = Array("EmploymentStatusIndex", "ListingCategoryIndex")
    val booleanColumns = Array("IsBorrowerHomeowner","CurrentlyInGroup", "IncomeVerifiable")
    
    val empStatusIndexer = new StringIndexer()
                          .setInputCol("EmploymentStatus")
                          .setOutputCol("EmploymentStatusIndex")
    val listCatIndexer = new StringIndexer()
                          .setInputCol("ListingCategory")
                          .setOutputCol("ListingCategoryIndex")
    val featuresAssembler = new VectorAssembler()
                                .setInputCols(columns ++ categoricalColumns)
                                .setOutputCol("features")
    val decisionTreeClassifier = new DecisionTreeClassifier()
                                      .setLabelCol("LoanStatus")
    val loanDecisionPipeLine = new Pipeline()
                                    .setStages(Array(empStatusIndexer
                                                    ,listCatIndexer
                                                    ,featuresAssembler
                                                    ,decisionTreeClassifier))
                                  
    val loadDecisionPLModel = loanDecisionPipeLine.fit(loantrainingDataDF)
    
    val loanDecisionDF = loadDecisionPLModel.transform(loantestDataDF)
    
    loanDecisionDF.select($"LoanStatus",$"prediction",$"features").show(300)
    
    val accuracy = new MulticlassClassificationEvaluator()
                        .setLabelCol("LoanStatus")
                        .setMetricName("accuracy")
                        .evaluate(loanDecisionDF)
    println("The accuracy of the loandecision model is:" + accuracy) 
    println("Test Error is:" + (1 - accuracy)) 
    
    val treeModel = loadDecisionPLModel.stages.last.asInstanceOf[DecisionTreeClassificationModel]
    println(treeModel)
  }
}