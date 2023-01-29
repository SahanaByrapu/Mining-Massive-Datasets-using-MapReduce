
import sys
import os
import math
import time
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession 

# spark config
spark = SparkSession \
    .builder \
    .appName("realized volatility recommendation") \
    .config("spark.driver.maxResultSize", "400g") \
    .config("spark.driver.memory", "400g") \
    .config("spark.executor.memory", "100g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()
# get spark context
sc = spark.sparkContext
"""train = spark.read.option("header",True) \
     .csv("/content/gdrive/MyDrive/train_pre_657.csv")
test = spark.read.option("header",True) \
     .csv("/content/gdrive/MyDrive/test_pre_657.csv")"""

###Pre-Processed data
train = spark.read.option("header",True) \
     .csv("/user/sbyrapu/input/preprocessed/train_pre_657.csv")
test = spark.read.option("header",True) \
     .csv("/user/sbyrapu/input/preprocessed/test_pre_657.csv")

train.count()

test.count()

train.show()

from pyspark.sql.types import FloatType
input_columns = train.columns
for column in input_columns:
   train=train.withColumn(column, train[column].cast(FloatType()))
   #test=test.withColumn(column, test[column].cast(FloatType()))

exclude_cols = 'time_id target row_id'.split()
features_lst = np.setdiff1d(train.columns, [exclude_cols]).tolist()
train_data,val_data=train.randomSplit([0.7, 0.3], seed=1)
label_key='target'

train_data.show()

from pyspark.ml.feature import VectorAssembler
print("converting features into single vector list")
assembler = VectorAssembler(inputCols=features_lst,outputCol='features')

train_tr = assembler.setHandleInvalid("skip").transform(train_data).select('features','target')
val_tr=assembler.setHandleInvalid("skip").transform(val_data).select('features','target')
#test = assembler.transform(test).select('features','target')

train_tr.show()

train_tr=train_tr.withColumnRenamed('target','label')
val_tr=val_tr.withColumnRenamed('target','label')

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.regression import LinearRegression,GBTRegressor, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


lr_reg=LinearRegression(labelCol='label')
paramGrid = (ParamGridBuilder() \
                .addGrid(lr_reg.maxIter, [10,15,20])
                .build())

evaluator=RegressionEvaluator()
crossval = CrossValidator(estimator=lr_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit Model: Run cross-validation, and choose the best set of parameters.
fitModel = crossval.fit(train_tr)
fitModel = fitModel.bestModel

predictions = fitModel.transform(val_tr)
RMSE1=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
MAE1=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
R1=evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print("-----------------------------Linear Regression-----------------------------")
print("RMSE for Linear Regression",RMSE1)

#-----------------------------Linear Regression-----------------------------
#RMSE for Linear Regression 0.0012098714084713288


gbt_reg=GBTRegressor(labelCol='label')
paramGrid = (ParamGridBuilder() \
                .addGrid(gbt_reg.maxIter, [10,15,20])
                .addGrid(gbt_reg.maxDepth, [3,5,10])
                .build())

evaluator=RegressionEvaluator()
crossval = CrossValidator(estimator=gbt_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit Model: Run cross-validation, and choose the best set of parameters.
fitModel = crossval.fit(train_tr)
fitModel = fitModel.bestModel

predictions = fitModel.transform(val_tr)
RMSE2=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
MAE2=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
R2=evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print("-----------------------------GBT Regression-----------------------------")
print("RMSE for GBT Regression",RMSE2)

#-----------------------------GBT Regression-----------------------------
#RMSE for GBT Regression 0.0012343948086735185 """

dec_reg=DecisionTreeRegressor(labelCol='label')
paramGrid = (ParamGridBuilder() \
                .addGrid(dec_reg.maxDepth, [3,5,9])
                .build())

evaluator=RegressionEvaluator()
crossval = CrossValidator(estimator=dec_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit Model: Run cross-validation, and choose the best set of parameters.
fitModel = crossval.fit(train_tr)
fitModel = fitModel.bestModel
                
predictions = fitModel.transform(val_tr)
RMSE3=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
MAE3=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
R3=evaluator.evaluate(predictions, {evaluator.metricName: "r2"})



print("-----------------------------DecisionTreeRegressor-----------------------------")
print("RMSE for DecisionTreeRegressor",RMSE3)
#-----------------------------DecisionTreeRegressor-----------------------------
#RMSE for DecisionTreeRegressor 0.0013495226019444248




rf_reg=RandomForestRegressor(labelCol='label')
paramGrid = (ParamGridBuilder() \
                .addGrid(rf_reg.maxDepth, [3,5,7])
                .build())

evaluator=RegressionEvaluator()
crossval = CrossValidator(estimator=rf_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit Model: Run cross-validation, and choose the best set of parameters.
fitModel = crossval.fit(train_tr)
fitModel = fitModel.bestModel
                
predictions = fitModel.transform(val_tr)
RMSE4=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
MAE4=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
R4=evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print("-----------------------------RandomForestRegressor-----------------------------")
print("RMSE for RandomForestRegressor",RMSE4)

#-----------------------------RandomForestRegressor-----------------------------
#sRMSE for RandomForestRegressor 0.0012968211053099618

predictions.show(n=30)

#predictions.write.mode("overwrite").csv("/user/sbyrapu/input/preds_df_RF_reg")

print("RMSE scores!!!")
print(RMSE1,RMSE2,RMSE3,RMSE4)
print("MAE scores!!!")
print(MAE1,MAE2,MAE3,MAE4)
print("R2 scores!!!")
print(R1,R2,R3,R4)




