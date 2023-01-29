
import os
import math
from pyspark.sql import SparkSession

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

from pyspark.sql.types import StringType,ArrayType,StructType,StructField,IntegerType
from pyspark.sql.functions import udf, regexp_replace, lower, split, trim, explode, lit, col, collect_list
from pyspark.ml.feature import StopWordsRemover 


getdata_udf = udf(keepdate, StringType())
clean_udf = udf(clean,StringType())

# year_text_df = df.select(getdata_udf('year').alias('year'), clean_udf('text').alias('text') )
year_text_df = df.select(getdata_udf('year').alias('year'),"text" )

year_text1 = year_text_df.select('year', regexp_replace("text", "[\n\t]", " ").alias('text')) # remove new lines and tabs
year_text1 = year_text1.select('year', regexp_replace("text", "<p>|</p>", " ").alias('text')) # remove html p
year_text1 = year_text1.select('year', regexp_replace("text", "[\-\.,!\(\)\"\'\?;]", " ").alias('text')) # remove punc
year_text1 = year_text1.select('year', regexp_replace("text", "\s+", " ").alias('text')) # remove spaces
year_text1 = year_text1.select('year', trim("text").alias('text')) # remove spaces from start and end
year_text1 = year_text1.select('year', lower("text").alias('text')) # lower spaces

year_text1 = year_text1.select('year', split("text", " ").alias('text')) # split

year_text1.rdd.take(1)

remover = StopWordsRemover()
remover.setInputCol('text')
remover.setOutputCol('words')
preprocess_df=remover.transform(year_text1).select("year", "words")

preprocess_df.show()

year_word_df = preprocess_df.select("year", explode("words").alias("word"))

year_word_df.show()

year_word_2009_df = year_word_df.filter(year_word_df.year >= 2009)
year_word_2009_df = year_word_2009_df.withColumn("count", lit(1))

year_word_2009_df.show() #select("year").distinct().collect()

year_word_groups_df = year_word_2009_df.groupBy('year','word').sum('count').withColumnRenamed("sum(count)", "count")

year_word_groups_df.show()

def grouper(year):
  years = range(2009,2022, 4)
  for y in years:
    if int(year) < y+4:
      return str(y) + "-" + str(y + 4)

grouper_udf = udf(grouper, StringType())

# grouper_udf = udf(grouper, ArrayType(StructType([
#     StructField("start", IntegerType(), False),
#     StructField("end", IntegerType(), False)
# ])))

years_word_df = year_word_groups_df.select(grouper_udf('year').alias("years"), 'word', "count")

years_word_df.groupBy('years','word').avg('count').withColumnRenamed("avg(count)", "average").orderBy(col("average").desc()).show()

word_years_agg = years_word_df.select("word", "years", "count").groupBy("word", "years").agg(collect_list("count").alias("counts"))

mean = lambda data: float(sum(data) / len(data))
variance = lambda data: mean([(x - mean(data)) ** 2 for x in data])
stddev = lambda data: math.sqrt(variance(data))

def stdev(values):
  if len(values) > 1:
    return round(stddev(values), 2)
  else:
    return round(values[0], 2)
    
import statistics
def avglist(l):
  return statistics.mean(l)

stdev_udf = udf(stdev, StringType())
avg_udf =  udf(avglist, StringType())

word_years_agg.show()

word_years_stddev = word_years_agg.select("word", "years", stdev_udf('counts').alias('stddev'), avg_udf('counts').alias('avg'))

word_years_stddev.show()

#1.2

def nextyear(string):
  return int(string.split("-")[1]) + 1

nextyear_udf = udf(nextyear, IntegerType())

word_years_nextyear_stddev = word_years_stddev.select("word", "years", nextyear_udf("years").alias('nextyear'), "stddev", "avg")

word_years_nextyear_stddev.show()

year_word_groups_df.show()

joined = word_years_nextyear_stddev.join(year_word_groups_df,
                                         (word_years_nextyear_stddev.word == year_word_groups_df.word)
                                         & (word_years_nextyear_stddev.nextyear == year_word_groups_df.year))

joined.show()

final_result = joined.filter(joined["count"] > (joined.avg + 2 * joined.stddev))



final_result.show()



