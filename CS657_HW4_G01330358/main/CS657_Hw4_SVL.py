def filter_func(x):
    pair = x[0]
    sim = x[1]
    return (pair[0] == movieID or pair[1] == movieID) and sim[0] > scoreThreshold and sim[1] > coOccurenceThreshold

filteredResults = moviePairSimilarities.filter(filter_func)
filteredResults.take(5)

# Sort by quality score.
results = filteredResults.take(10)

print("Top 10 similar movies for " + str([movieID]))
for result in results:
    (pair, sim) = result
    similarMovieID = pair[0]
    if (similarMovieID == movieID):
        similarMovieID = pair[1]
        
        movie_name = movies.filter(movies.movieId == similarMovieID).rdd.collect()[0][1]
        print(f'{movie_name:20}\nscore: {str(sim[0])} \tstrength: {str(sim[1])}')

# part 3

rating.show()

movies.show()

from pyspark.sql import functions as F

from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml import Model, Estimator
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer

class ArrayStringIndexerModel(Model
                              ,DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, indexer, inputCol: str, outputCol: str):
        super(ArrayStringIndexerModel, self).__init__()
        self.indexer = indexer
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df: DataFrame=[]) -> DataFrame:

        # Creating always increasing id (as in fit)
        df = df.withColumn("id_temp_added", F.monotonically_increasing_id())\

        # Exploding "inputCol" and saving to the new dataframe (as in fit)
        df2 = df.withColumn('inputExpl', F.explode(self.inputCol)).select('id_temp_added', 'inputExpl')

        # Transforming with fitted "indexed"
        indexed_df = self.indexer.transform(df2)

        # Converting indexed back to array
        indexed_df = indexed_df.groupby('id_temp_added').agg(F.collect_list(F.col(self.outputCol)).alias(self.outputCol))

        # Joining to the main dataframe
        df = df.join(indexed_df, on='id_temp_added', how='left')

        # dropping created id column
        df = df.drop('id_temp_added')

        return df


class ArrayStringIndexer(Estimator
                ,DefaultParamsReadable, DefaultParamsWritable):
    """
    A custom Transformer which applies string indexer to the array of strings
    (explodes, applies StirngIndexer, aggregates back)
    """

    def __init__(self, inputCol: str, outputCol: str):
        super(ArrayStringIndexer, self).__init__()
       # self.indexer = None
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _fit(self, df: DataFrame = []) -> ArrayStringIndexerModel:
        # Creating always increasing id
        df = df.withColumn("id_temp_added", F.monotonically_increasing_id())

        # Exploding "inputCol" and saving to the new dataframe
        df2 = df.withColumn('inputExpl', F.explode(self.inputCol)).select('id_temp_added', 'inputExpl')

        # Indexing self.indexer and self.indexed dataframe with exploded input column
        indexer = StringIndexer(inputCol='inputExpl', outputCol=self.outputCol)
        indexer = indexer.fit(df2)

        # Returns ArrayStringIndexerModel class with fitted StringIndexer, input and output columns
        return ArrayStringIndexerModel(indexer=indexer, inputCol=self.inputCol, outputCol=self.outputCol)

joined_df = rating.join(movies, ['movieId'], 'right').select("userId", "genres", "rating").limit(100)

joined_df.show()

userid_genres_rating = joined_df.withColumn("genre_array", F.split("genres",'\|')).drop("genres")
userid_genres_rating = userid_genres_rating.withColumn("userId", F.array("userId"))
userid_genres_rating.show()

tags_indexer = ArrayStringIndexer(inputCol="genre_array", outputCol="genre_array_indexed")
userid_generes_rating_feat = tags_indexer.fit(userid_genres_rating).transform(userid_genres_rating)
userid_generes_rating_feat.show()

def func(arr):
  vector = [0] * 19
  for i in arr:
      vector[int(i)] = 1
  return vector

from pyspark.sql.types import ArrayType, IntegerType

udfvector = F.udf(lambda x: func(x), ArrayType(IntegerType(), True))

userid_generes_rating_vector = userid_generes_rating_feat.withColumn('vector', udfvector(F.col('genre_array_indexed')))

userid_generes_rating_vector.show()

features_rating = userid_generes_rating_vector.withColumn("features_array", F.concat(F.col("userId"), F.col("vector"))).drop("genre_array", "userId", "genre_array_indexed")
features_rating.show()

from pyspark.ml.functions import array_to_vector
features_rating1 = features_rating.withColumn("features", array_to_vector('features_array'))

features_rating_clean = features_rating1.filter("rating <= 5.0").filter("rating >= 0.0")

features_rating_clean.dtypes

training_df,test_df = features_rating_clean.randomSplit([0.75,0.25])

from pyspark.ml.regression import LinearRegression
log_reg = LinearRegression(labelCol='rating', featuresCol='features').fit(training_df)

test_results = log_reg.evaluate(test_df).predictions

test_results.show()

test_results.write.mode('overwrite').csv('/content/preds_df_SVL')