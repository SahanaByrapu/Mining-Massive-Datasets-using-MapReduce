
import os
import math
import time


from pyspark.sql import SparkSession, Row, SQLContext

# if__name__=="__main__":
#   if len(sys.argv)!=2:
#     print("Usage:wordcount<file>",file=sys.stderr)
#     exit(-1)

spark = SparkSession.builder.appName("PythonWordCount").getOrCreate()

sc = spark.sparkContext

files=sc.wholeTextFiles('texts')

#files.collect()

files.take(1)

def keepdate(s):
  return s[-8:-4]

def clean(s):
  
  return s[:10]

# year_text = files.map(lambda x: (keepdata(x[0]), clean(x[1])))
# year_text.take(4)

df = files.toDF(["year","text"])

from pyspark.sql.types import StringType,ArrayType,StructType,StructField,IntegerType, FloatType
from pyspark.sql.functions import udf, regexp_replace, lower, split, trim, explode, lit, col, collect_list
from pyspark.ml.feature import StopWordsRemover 


getdata_udf = udf(keepdate, StringType())
clean_udf = udf(clean,StringType())

# year_text_df = df.select(getdata_udf('year').alias('year'), clean_udf('text').alias('text') )
year_text_df = df.select(getdata_udf('year').alias('year'),"text" )

year_text1 = year_text_df.select('year', regexp_replace("text", "[\n\t]", " ").alias('text')) # remove new lines and tabs
year_text1 = year_text1.select('year', regexp_replace("text", "<p>|</p>", " ").alias('text')) # remove html p
year_text1 = year_text1.select('year', regexp_replace("text", "[\-,!\(\)\"\'\?;:]", " ").alias('text')) # remove punc
year_text1 = year_text1.select('year', regexp_replace("text", "\s+", " ").alias('text')) # remove spaces
year_text1 = year_text1.select('year', trim("text").alias('text')) # remove spaces from start and end
year_text1 = year_text1.select('year', lower("text").alias('text')) # lower spaces

year_text1.show()

text_rdd = year_text1.rdd.map(lambda x: x[1])

sentences_rdd = text_rdd.map(lambda x: x.split(".")).flatMap(lambda x: x)

df_sentences = SQLContext(sc).createDataFrame([Row(sent=w) for w in sentences_rdd.collect()])

df_sentences.show()

df_clean_sentences = df_sentences.select(trim("sent").alias('sent'))

word_array_df = df_clean_sentences.select(split("sent", " ").alias('sent'))

remover = StopWordsRemover()
remover.setInputCol('sent')
remover.setOutputCol('words')
preprocess_df = remover.transform(word_array_df).select("words")

preprocess_df.show()

words_all = preprocess_df.rdd.flatMap(lambda x: x)

words_list = words_all.map(lambda x: [(w, time.time()) for w in x]).flatMap(lambda x: x)

words_list.take(1)

word_id_df = SQLContext(sc).createDataFrame([Row(word=word, id=id) for (word, id) in words_list.collect()])

word_id_df.show()

left = word_id_df.toDF('left', 'id')
right = word_id_df.toDF('right', 'id')

word_pairs = left.join(right, (left.id == right.id) & (left.left < right.right)).drop('id')

word_pairs_sum = word_pairs.groupBy("left", "right").count()

all_pair_sum = word_pairs_sum.groupBy().sum().collect()[0][0]

word_pairs_imp = word_pairs_sum.filter("count > 5").orderBy("count", ascending=False)

word_pairs_imp.show()

prob_sum_left = word_id_df.drop("id").groupBy("word").count().withColumnRenamed('count', 'left_sum')
prob_total_left = word_id_df.drop("id").groupBy("word").count().groupBy().sum().collect()[0][0]
word_pairs_new = word_pairs_imp.join(prob_sum_left, on=(word_pairs_imp.left == prob_sum_left.word))
word_pairs_new = word_pairs_new.drop('word')
word_pairs_new.show()

word_pairs_new.registerTempTable('word_pairs_new')
prob_df = SQLContext(sc).sql(f'SELECT *, count/{all_pair_sum} as AnB, left_sum/{prob_total_left} as A FROM word_pairs_new')
prob_df.show()

prob_df.registerTempTable('prob_df')
pba_df =  SQLContext(sc).sql(f'SELECT *, AnB/A as PBA FROM prob_df')
pba_df.show()

pba_df.registerTempTable('pba_df')
lift_df =  SQLContext(sc).sql(f'SELECT *, AnB/(A*A) as Lift FROM pba_df')

lift_final = lift_df.filter("Lift > 3").drop('AnB').drop('A').drop('left_sum').drop('count').orderBy("Lift", ascending=False)
lift_final.show()

