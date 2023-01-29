import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession 

# spark config
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "96g") \
    .config("spark.driver.memory", "96g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()


# get spark context
sc = spark.sparkContext

# read in the dataset into pyspark DataFrame
data = spark.read.csv('/user/sbyrapu/input/ml-20m/ratings.csv', header='True', inferSchema='True')


#Splitting the dataset into train(0.8) and test(0.2)
x_train, x_test = data.randomSplit([0.8, 0.2], 17)
print(x_train.show(n=50))
print("----data_schema------",x_train.printSchema())

print("Writing training data into file")
x_train.write.mode("overwrite").csv("/user/sbyrapu/input/train_df_final")

print("Writing testing data into file")
x_test.write.mode("overwrite").csv("/user/sbyrapu/input/test_df_final")

# cache data
x_train.cache()
x_test.cache()
#crossvalidation over a subset (80%) of the data
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# initialize the ALS model
als_model = ALS(userCol='userId', itemCol='movieId',regParam=0.01, ratingCol='rating')
# create the parameter grid
#params = ParamGridBuilder().addGrid(als_model.regParam, [.01, .05, .1, .20] ).addGrid(als_model.rank, [10, 20, 500, 100] ).build()
params = ParamGridBuilder().addGrid(als_model.regParam, [.05] ).addGrid(als_model.rank, [20] ).build()
evaluator = RegressionEvaluator()
evaluator.setLabelCol("rating").setPredictionCol("prediction")
#instantiating crossvalidator estimator
cv= CrossValidator(estimator=als_model, estimatorParamMaps=params, evaluator=evaluator, parallelism=4,numFolds=3)


# Build the recommendation model using Alternating Least Squares
print("\n--------------------------------Training recommendation model...-----------------------------------------")
best_model = cv.fit(x_train)
model = best_model



print("\n----------------------Saving the best recommendation model...---------------------------")
#Save and load model
#model.save('/user/sbyrapu/input/alsV11')


result=model.transform(x_test)
predictions=result.select("rating","prediction")
predictions=predictions.na.drop()
print("----------------------------predictions-------------------------------------------------------")
print(predictions.show(n=50))



mae=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
MSE=evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
RMSE=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print('------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is--------------------',MSE,RMSE,mae)
#------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is-------------------- 0.6177755717763813 0.7859870048393811 0.6080260057493034


result.write.mode("overwrite").csv("/user/sbyrapu/input/pred_df_final")

result.write.mode("overwrite").csv("/user/sbyrapu/input/preds_df_Item_Item_CF")