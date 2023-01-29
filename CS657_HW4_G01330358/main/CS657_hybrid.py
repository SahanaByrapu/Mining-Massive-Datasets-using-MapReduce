import sys
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import udf, regexp_replace, lower, split, trim, explode, lit, col, collect_list
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SparkSession 

spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "96g") \
    .config("spark.driver.memory", "96g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()




spark = SparkSession.builder.master("local").getOrCreate()
# read in the dataset into pyspark DataFrame
result_ALS = spark.read.csv('/user/sbyrapu/input/pred_df_final')
print('------------------------ALS_predictions---------------------')
print(result_ALS.show(n=20))
#result_df_Item_Item_CF= spark.read.csv('/user/sbyrapu/input/pred_df_final')
result_df_Item_Item_CF=result_ALS
print('------------------------Item_Item_CF_predictions---------------------')
print(result_df_Item_Item_CF.show(n=20))
print('------------------------combining both ALS and Item_Item_CF ---------------------')


hybrid_df=spark.createDataFrame([],result_ALS.schema)



def hybrid_calculation_function(rating_als,rating_item_item):
    if math.isnan(rating_als):
        return rating_item_item
    return (rating_item_item * 0.6) + (rating_als * 0.4)



df1 = result_ALS.alias('df1').selectExpr('_c0 as uid1','_c1 as mid1','_c2 as r1', '_c4 as p1').cache()
df2 = result_df_Item_Item_CF.alias('df2').selectExpr('_c0 as uid2','_c1 as mid2','_c2 as r2', '_c4 as p2').cache()
#Put into a python list the criteria for the join
cond = [(df1.uid1 == df2.uid2) & (df1.mid1 =='83') & (df2.mid2 =='83')]
hybrid_df = df1.join(df2, cond, 'inner')

#print(hybrid_df.count())

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
#[uid1: string, mid1: string, r1: string, p1: string, uid2: string, mid2: string, r2: string, p2: string]>
print("printSchema!!!!!!!!!",hybrid_df.printSchema)

my_udf = udf(lambda x,y: hybrid_calculation_function(x,y), FloatType())

hybrid_df =hybrid_df.withColumn('r1',hybrid_df.r1.cast(FloatType()))
hybrid_df =hybrid_df.withColumn('p1',hybrid_df.p1.cast(FloatType()))
hybrid_df =hybrid_df.withColumn('r2',hybrid_df.r2.cast(FloatType()))
hybrid_df =hybrid_df.withColumn('p2',hybrid_df.p2.cast(FloatType()))

hybrid_df =hybrid_df.withColumn('new_pred',my_udf(hybrid_df.p1,hybrid_df.p2))

hybrid_df.show()

predictions=hybrid_df.select('r1','new_pred')

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
evaluator.setLabelCol("r1").setPredictionCol("new_pred")
mae=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
MSE=evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
RMSE=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print('------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is--------------------',MSE,RMSE,mae)


result_df_SL= spark.read.csv('/user/sbyrapu/input/preds_df_SL')
print('-----------------------------SVL_predictions------------------------------')
print(result_df_SL.show(n=50))

df3 = result_df_SL.alias('df3').selectExpr('_c0 as r3', '_c1 as p3').cache()

new_pred1=hybrid_df.select('new_pred').collect(n=20)
SVL_pred=result_df_SL.select('predictions').collect(n-20)

svl=[0.4*val for val in SVL_pred]
sv2=[0.6*val for val in new_pred1]

sv3=[]
for i in range(len(svl)):
  sv3.append(svl[i]+sv2[i])

hybrid_df=spark.createDataFrame(zip(SVL_pred,new_pred1,sv3),['SVL','ALS+Item_CF','hybrid_pred'])

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
evaluator.setLabelCol("SVL").setPredictionCol("hybrid_pred")
mae=evaluator.evaluate(hybrid_df, {evaluator.metricName: "mae"})
MSE=evaluator.evaluate(hybrid_df, {evaluator.metricName: "mse"})
RMSE=evaluator.evaluate(hybrid_df, {evaluator.metricName: "rmse"})
print('------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is--------------------',MSE,RMSE,mae)
#------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is-------------------- 0.09496538182514357 0.3081645369362665 0.23982833702747666