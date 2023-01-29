
from pyspark.sql import SparkSession 
spark = SparkSession \
    .builder \
    .appName("2020 US Election Tweets Analysis") \
    .getOrCreate()
data_df = spark.read.option("header",True) \
     .csv("/user/sbyrapu/input/hashtag_joebiden.csv")

from pyspark.sql.functions import col,isnan, when, count
records=data_df.count()
df_missing_count= data_df.select([(count((when(isnan(c)|col(c).isNull(),c)))/records).alias(c) for c in data_df.columns])

df_missing_count.show()

#print(data_df.count())
df=data_df.dropna()
#print(df.count())

df.show()

from pyspark.sql.types import StringType,ArrayType,StructType,StructField,IntegerType
from pyspark.sql.functions import udf, regexp_replace, lower, split, trim, explode, lit, col, collect_list
from pyspark.ml.feature import StopWordsRemover 

df_clean = df.select('tweet_id','tweet',  regexp_replace("tweet", "\[.*?\]", " ").alias('text')) # remove punc
df_clean = df_clean.select('tweet_id','tweet',  regexp_replace("text", "https?://\S+|www\.\S+", " ").alias('text'))
df_clean = df_clean.select('tweet_id','tweet',  regexp_replace("text", "<.*?>+", " ").alias('text'))
df_clean = df_clean.select('tweet_id','tweet',  regexp_replace("text", "\n", " ").alias('text'))
df_clean = df_clean.select('tweet_id','tweet', regexp_replace("text", "\w*\d\w*", " ").alias('text')) # remove alpha-numeric characters
df_clean = df_clean.select('tweet_id','tweet', regexp_replace("text", "@[A-Za-z0-9]+", " ").alias('text')) 
df_clean = df_clean.select('tweet_id','tweet', regexp_replace("text", "#", " ").alias('text'))
df_clean = df_clean.select('tweet_id','tweet', regexp_replace("text", "[^\w]", " ").alias('text'))
df_clean = df_clean.select('tweet_id','tweet', trim("text").alias('text')) # remove spaces from start and end
df_clean = df_clean.select('tweet_id','tweet', lower("text").alias('text')) # lower

from textblob import TextBlob
def sentiment_analysis(tweet_words):
    # Determine polarity
    polarity = TextBlob(" ".join(tweet_words)).sentiment.polarity
    #polarity = TextBlob(tweet_words).sentiment.polarity
    # Classify overall sentiment
    if polarity > 0:
        # positive
        sentiment = 1
    elif polarity == 0:
        # neutral
        sentiment = 0
    else:
        # negative
        sentiment = 2
    return sentiment

df_clean=df_clean.select('tweet_id','text')
df_clean.show()

from pyspark.ml.feature import Tokenizer,CountVectorizer

tokenizer = Tokenizer(outputCol="words1")
tokenizer.setInputCol("text")
df_tokens=tokenizer.transform(df_clean)

remover = StopWordsRemover().setInputCol('words1').setOutputCol('words')
df_stop= remover.transform(df_tokens).select("tweet_id","words")

df_stop.show()

my_udf = udf(lambda x: sentiment_analysis(x), StringType())
senti_udf=df_stop.withColumn('sentiment', my_udf(df_stop.words) )

senti_udf.show()

from pyspark.sql.functions import col,when
senti_udf2=senti_udf.withColumn('sentiment', 
                     when(senti_udf.sentiment==0, 2).otherwise(col('sentiment')))

senti_udf2.show()

#pip install langdetect

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from langdetect import detect
def language_detection(tweet_words):
    tweet_words=" ".join(tweet_words)
    if(len(tweet_words)>0):
      lang = detect(tweet_words)
    else:
      lang='no'
    return lang

my_udf = udf(lambda x: language_detection(x), StringType())
lang_udf=senti_udf2.withColumn('lang', my_udf(senti_udf2.words) )

lang_udf.show()
df_lang=lang_udf.filter((lang_udf.lang=='en'))
#df_lang.show()
df_new=df_lang.drop('lang')


from pyspark.ml.feature import CountVectorizer,IDF

cv=CountVectorizer(inputCol="words", outputCol="rawfeatures")
cv_model=cv.fit(df_new)
featurizedData = cv_model.transform(df_new)

idf = IDF(inputCol="rawfeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
vectorized_tokens = idfModel.transform(featurizedData)

from pyspark.ml.clustering import LDA
num_topics = 10
lda = LDA(k=num_topics, maxIter=10)
lda_model = lda.fit(vectorized_tokens)
transformed = lda_model.transform(vectorized_tokens)
lda_model.describeTopics().show()

# extract vocabulary from CountVectorizer
vocab = cv_model.vocabulary
topics = lda_model.describeTopics()   
topics_rdd = topics.rdd
topics_words = topics_rdd\
       .map(lambda row: row['termIndices'])\
       .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
       .collect()
for idx, topic in enumerate(topics_words):
    print("topic: {}".format(idx))
    print("*"*25)
    for word in topic:
       print(word)
    print("*"*25)

#pip install pyLDAvis

#import pyLDAvis
def format_data_to_pyldavis(df_filtered, cv_model, transformed, lda_model):
    xxx = df_filtered.select((explode(df_filtered.words)).alias("words")).groupby("words").count()
    word_counts = {r['words']:r['count'] for r in xxx.collect()}
    word_counts = [word_counts[w] for w in cv_model.vocabulary]


    data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T, 
            'doc_topic_dists': np.array([x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [r[0] for r in df_filtered.select(size(df_filtered.words_filtered)).collect()],
            'vocab': cv_model.vocabulary,
            'term_frequency': word_counts}

    return data

def filter_bad_docs(data):
    bad = 0
    doc_topic_dists_filtrado = []
    doc_lengths_filtrado = []

    for x,y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x)==0:
            bad+=1
        elif np.sum(x) != 1:
            bad+=1
        elif np.isnan(x).any():
            bad+=1
        else:
            doc_topic_dists_filtrado.append(x)
            doc_lengths_filtrado.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtrado
    data['doc_lengths'] = doc_lengths_filtrado

# FORMAT DATA AND PASS IT TO PYLDAVIS
#data = format_data_to_pyldavis(df_new, cv_model, transformed, lda_model)
#filter_bad_docs(data) # this is, because for some reason some docs apears with 0 value in all the vectors, or the norm is not 1, so I filter those docs.
#py_lda_prepared_data = pyLDAvis.prepare(**data)
#pyLDAvis.display(py_lda_prepared_data)

df_new=vectorized_tokens.select('features','sentiment')

from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler

#Encoding the columns using StringIndexer
print("Encoding the columns")
inputs = ["sentiment"]
outputs = ["label"]
stringIndexer = StringIndexer(inputCols=inputs, outputCols=outputs)
model = stringIndexer.fit(df_new)
result = model.transform(df_new)
result.show()

result=result.select('features','label')
result=result.withColumnRenamed('features','features1')

#Prepare training and test data.
print("random split (70%,30%) of the data into training and test.")
splits = result.randomSplit([0.7, 0.3], None)
x_train = splits[0]
x_test = splits[1]



from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
print("converting features into single vector list")
features_list = ['features1']
assembler = VectorAssembler(inputCols=features_list,outputCol='features')
train = assembler.transform(x_train).select('features','label')
test = assembler.transform(x_test).select('features','label')


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
classifier= LogisticRegression()
paramGrid = (ParamGridBuilder() \
              .addGrid(classifier.maxIter, [10, 15,20])
              .build())
folds=10

evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC")
crossval = CrossValidator(estimator=classifier,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=folds) 

fitModel = crossval.fit(train)
bestmodel = fitModel.bestModel
best_param=bestmodel.getMaxIter()

predictions = fitModel.transform(test)
results = predictions.select(['prediction', 'label'])

print("area under roc=",evaluator.evaluate(results))

transformed=transformed.select('transformed','label')
#Prepare training and test data.
print("random split (70%,30%) of the data into training and test.")
splits = transformed.randomSplit([0.7, 0.3], None)
x_train = splits[0]
x_test = splits[1]

from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
print("converting features into single vector list")
features_list = ['transformed']
assembler = VectorAssembler(inputCols=features_list,outputCol='features1')
train = assembler.transform(x_train).select('features1','label')
test = assembler.transform(x_test).select('features1','label')



classifier= LogisticRegression()
paramGrid = (ParamGridBuilder() \
              .addGrid(classifier.maxIter, [10, 15,20])
              .build())
folds=10

evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC")
crossval = CrossValidator(estimator=classifier,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=folds) 

fitModel = crossval.fit(train)
bestmodel = fitModel.bestModel
best_param=bestmodel.getMaxIter()

predictions = fitModel.transform(test)
results = predictions.select(['prediction', 'label'])

print("area under roc for LDA=",evaluator.evaluate(results))



