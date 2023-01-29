
from pyspark.sql import SparkSession 
spark = SparkSession \
    .builder \
    .appName("Predicting Fake Job Postings") \
    .getOrCreate()
df = spark.read.option("header",True) \
     .csv("/user/sbyrapu/input/fake_job_postings.csv")

#Deleting the data that are having other than valid data
print("Deleting the invalid data from the columns")
df=df.filter((df.fraudulent=='0')| (df.fraudulent  == '1'))
df=df.filter( (df.telecommuting == '0') | (df.fraudulent  == '1') ) 
df=df.filter( (df.has_company_logo == '0')| (df.has_company_logo  == '1') ) 
df=df.filter( (df.has_questions == '0') | (df.has_questions  == '1')) 
records=df.count()

from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler

#Encoding the columns using StringIndexer
print("Encoding the columns")
inputs = ["fraudulent", "telecommuting","has_company_logo","has_questions"]
outputs = ["label", "telecommuting1","has_company_logo1","has_questions1"]
stringIndexer = StringIndexer(inputCols=inputs, outputCols=outputs)
model = stringIndexer.fit(df)
result = model.transform(df)
result.show()

result1=result

#Computing null values percentage
import pyspark.sql.functions as sqlf
null_counts = result1.select([(sqlf.count(sqlf.when(sqlf.col(c).isNull(), c))/records).alias(c) for c in result1.columns]).collect()[0].asDict()
col_to_drop = [k for k, v in null_counts.items() if v > 0.01] 
result1 = result1.drop(*col_to_drop)

print("attributes with missing value percentage",)
print(null_counts)

print("cleaning the data -  removing punctuations,alpha-numeric, spaces, lowercase")
df_new=result

from pyspark.sql.types import StringType,ArrayType,StructType,StructField,IntegerType
from pyspark.sql.functions import udf, regexp_replace, lower, split, trim, explode, lit, col, collect_list
from pyspark.ml.feature import StopWordsRemover 

df_clean = df_new.select('job_id','title','description','telecommuting1','has_company_logo1','has_questions1','label',  regexp_replace("title", "[\-\.,!\(\)\"\'\?;]", " ").alias('text')) # remove punc
df_clean = df_clean.select('job_id','title','description','telecommuting1','has_company_logo1','has_questions1','label', regexp_replace("text", "\W+", " ").alias('text')) # remove alpha-numeric characters
df_clean = df_clean.select('job_id','title','description','telecommuting1','has_company_logo1','has_questions1','label', regexp_replace("text", "\s+", " ").alias('text')) # remove spaces
df_clean = df_clean.select('job_id','title','description','telecommuting1','has_company_logo1','has_questions1','label', trim("text").alias('text')) # remove spaces from start and end
df_clean = df_clean.select('job_id','title','description','telecommuting1','has_company_logo1','has_questions1','label', lower("text").alias('text')) # lower spaces

df_clean2 = df_clean.select('job_id','description', 'telecommuting1','has_company_logo1','has_questions1','label', 'text', regexp_replace("description", "[\-\.,!\(\)\"\'\?;]", " ").alias('text1')) # remove punc
df_clean2 = df_clean2.select('job_id','description','telecommuting1','has_company_logo1','has_questions1','label','text', regexp_replace("text1", "\W+", " ").alias('text1')) # remove alpha-numeric characters
df_clean2 = df_clean2.select('job_id','description','telecommuting1','has_company_logo1','has_questions1','label','text', regexp_replace("text1", "\s+", " ").alias('text1')) # remove spaces
df_clean2 = df_clean2.select('job_id','description','telecommuting1','has_company_logo1','has_questions1','label','text', trim("text1").alias('text1')) # remove spaces from start and end
df_clean2 = df_clean2.select('job_id','telecommuting1','has_company_logo1','has_questions1','label','text', lower("text1").alias('text1')) # lower spaces

df_clean2.show()

print("Split text into words, remove stopwords, and convert text into vectors")

from pyspark.ml.feature import Tokenizer,CountVectorizer,Word2Vec

tokenizer = Tokenizer(outputCol="words")
tokenizer.setInputCol("text")
df_tokens=tokenizer.transform(df_clean2)

remover = StopWordsRemover().setInputCol('words').setOutputCol('words1')
df_stop= remover.transform(df_tokens).select('job_id','telecommuting1','has_company_logo1','has_questions1','label',"words1","text1")

#cv = CountVectorizer().setInputCol("words1").setOutputCol("vectors1")
#model = cv.fit(df_stop)
#df_stop=model.setInputCol("words1").transform(df_stop)

cv = Word2Vec().setInputCol("words1").setOutputCol("vectors1")
model = cv.fit(df_stop)
df_stop=model.setInputCol("words1").transform(df_stop)

tokenizer = Tokenizer(outputCol="words")
tokenizer.setInputCol("text1")
df_tokens2=tokenizer.transform(df_stop)

remover = StopWordsRemover().setInputCol('words').setOutputCol('words2')
df_stop2=remover.transform(df_tokens2).select('job_id','telecommuting1','has_company_logo1','has_questions1','label',"vectors1","words2")

#cv = CountVectorizer().setInputCol("words2").setOutputCol("vectors2")
#model = cv.fit(df_stop2)
#df_stop2=model.setInputCol("words2").transform(df_stop2)

cv = Word2Vec().setInputCol("words2").setOutputCol("vectors2")
model = cv.fit(df_stop2)
df_stop2=model.setInputCol("words2").transform(df_stop2)

df_final=df_stop2.select('job_id','vectors1','vectors2','telecommuting1','has_company_logo1','has_questions1','label') 
df_final.show()

from pyspark.sql.functions import col

print("undersampling the majority class")
sampled = df_final.sampleBy("label", fractions={0.0: 0.06, 1.0:1.0}, seed=0)
sampled.groupBy("label").count().orderBy("label").show()

#Prepare training and test data.

print("random split (70%,30%) of the data into training and test.")
splits = sampled.randomSplit([0.7, 0.3], None)
x_train = splits[0]
x_test = splits[1]

# Set up independ and dependent vars
input_columns = x_train.columns
input_columns = input_columns[1:-1] # keep only relevant columns: everything but the first and last cols
dependent_var = 'label'

numeric_inputs = []
string_inputs = []
for column in input_columns:
  if  str(x_train.schema[column].dataType) == 'StringType':
       indexer = StringIndexer(inputCol=column, outputCol=column+"_num") 
       indexed = indexer.fit(x_train).transform(x_train)
       indexed1 = indexer.fit(x_test).transform(x_test)
       new_col_name = column+"_num"
       string_inputs.append(new_col_name)
  else:
      numeric_inputs.append(column)

print("converting features into single vector list")
features_list = numeric_inputs + string_inputs
assembler = VectorAssembler(inputCols=features_list,outputCol='features')
train = assembler.transform(x_train).select('features','label')
test = assembler.transform(x_test).select('features','label')

print("Cross Validation 10 folds...")

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,LinearSVC,MultilayerPerceptronClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

def InstanceFitModel(Mtype,folds,train):

            if Mtype in("LogisticRegression"):
                classifier=LogisticRegression()
                paramGrid = (ParamGridBuilder() \
                             .addGrid(classifier.maxIter, [10,15,20])
                             .build())
                

            if Mtype in("RandomForestClassifier"):
                classifier=RandomForestClassifier()
                paramGrid = (ParamGridBuilder() \
                               .addGrid(classifier.maxDepth, [2, 5, 10])
                             .build())
                
            if Mtype in("LinearSVC"):
                classifier=LinearSVC()
                paramGrid = (ParamGridBuilder() \
                             .addGrid(classifier.maxIter, [10, 15]) \
                             .addGrid(classifier.regParam, [0.1, 0.01]) \
                             .build())
                
            
            if Mtype == "MultilayerPerceptronClassifier":
              
              c=train.select("features").collect()
              count=len(c[0][0])
              layers = [203,203+1,203,2]

              MPC_classifier = MultilayerPerceptronClassifier(labelCol='label',featuresCol='features',maxIter=100, layers=layers, blockSize=128,seed=1234)
        
              """ paramGrid=(ParamGridBuilder() \
                             .addGrid(MPC_classifier.maxIter, [100, 150]) \
                             .build())
              evaluator=MulticlassClassificationEvaluator()
              crossval = CrossValidator(estimator=MPC_classifier,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=evaluator,
                                      numFolds=folds) 
              fitModel = crossval.fit(train)"""
              fitModel = MPC_classifier.fit(train)
              return fitModel
                
            #Cross Validator containing parameters
            evaluator=BinaryClassificationEvaluator()
            crossval = CrossValidator(estimator=classifier,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=evaluator,
                                      numFolds=folds) 

            # Fit-Model: Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            
           
            return fitModel


classifiers =['LogisticRegression','LinearSVC','RandomForestClassifier','MultilayerPerceptronClassifier']
#classifiers =[]
folds=10
for classfier in classifiers:
   fitModel = InstanceFitModel(classfier,folds,train)

   if classfier in ('LogisticRegression' ,'LinearSVC'):
      bestmodel=fitModel.bestModel
      best_param=bestmodel.getMaxIter()
      print('\033[1m' + 'Intercept: '+ '\033[0m',bestmodel.intercept)
      print('\033[1m' + 'Top 20 Coefficients:'+ '\033[0m')
      coeff_array = bestmodel.coefficients.toArray()
      coeff_scores = []
      for x in coeff_array:
          coeff_scores.append(float(x))
      result = spark.createDataFrame(zip(input_columns,coeff_scores), schema=['feature','coeff'])
      print(result.orderBy(result["coeff"].desc()).show(truncate=False))
  
   if classfier in ('RandomForestClassifier'):
      bestmodel=fitModel.bestModel
      best_param=bestmodel.getMaxDepth()
      print('\033[1m' + classfier," Top 20 Feature Importances"+ '\033[0m')
      print("(Scores add up to 1)")
      print("Lowest score is the least important")
      print(" ")
      featureImportances = bestmodel.featureImportances.toArray()
      imp_scores = []
      for x in featureImportances:
          imp_scores.append(float(x))
      result = spark.createDataFrame(zip(input_columns,imp_scores), schema=['feature','score'])
      print(result.orderBy(result["score"].desc()).show(truncate=False))
    
   predictions = fitModel.transform(test)
   #print(predictions.show())
   results = predictions.select(['prediction', 'label'])

   if classfier in ("MultilayerPerceptronClassifier"):
     best_param=fitModel.getMaxIter()
     print("")
     print('\033[1m' + classfier + '\033[0m')
     print('\033[1m' + "Model Weights: "+ '\033[0m', fitModel.weights.size)
     print("")
     metrics = ['weightedPrecision', 'weightedRecall', 'accuracy']
     for metric in metrics:
         evaluator = MulticlassClassificationEvaluator(metricName=metric)
         print('Test ' + metric + ' = ' + str(evaluator.evaluate(predictions)))
         if (metric == 'accuracy'):
             accuracy = str(evaluator.evaluate(predictions)) 
         if(metric == 'weightedPrecision'):
             precision= evaluator.evaluate(predictions)
         if(metric == 'weightedRecall'):
             recall= evaluator.evaluate(predictions)
             
      
   if classfier in ('LogisticRegression' ,'LinearSVC','RandomForestClassifier'):
      predictionAndLabels=results.rdd
      metrics = MulticlassMetrics(predictionAndLabels)
      cm=metrics.confusionMatrix().toArray()
      accuracy=(cm[0][0]+cm[1][1])/cm.sum()
      precision=(cm[0][0])/(cm[0][0]+cm[1][0])
      recall=(cm[0][0])/(cm[0][0]+cm[0][1])
    
   F1_Score =2*(recall * precision)/(recall + precision)
   print("Classifier",classfier,"accuracy,F1-Score", accuracy,F1_Score)

   accuracy = [str(accuracy)] 
   F1_score = [str(F1_Score)]
   best_param= [str(best_param)]
   columns = ['Classifier', 'accuracy','F1_score','best_param']   
   #classifiers_2 =['LogisticRegression-MaxIter','LinearSVC-MaxIter','RandomForestClassifier-MaxDepth','MultilayerPerceptronClassifier-MaxIter']
   result = spark.createDataFrame(zip(classfier,accuracy,F1_score,best_param), schema=columns)
   result.show()

import matplotlib.pyplot as plt

accuracy = [str(accuracy)] 
F1_score = [str(F1_Score)]

plt.figure()
x_val = ['Logistic','LinearSVC','RandomForest','Multilayer']
y_val = accuracy

plt.title('accuracy v/s models')
plt.xlabel('models')
plt.ylabel('metric')
plt.plot(x_val, y_val)
plt.plot(x_val, F1_score)

plt.legend(['accuracy'])
plt.legend(['accuracy','F1-score'])

