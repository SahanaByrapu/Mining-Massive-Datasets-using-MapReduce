import sys
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SparkSession 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from math import sqrt
#conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
#sc = SparkContext(conf = conf)
#sc.setCheckpointDir('checkpoint')

#data = sc.textFile("/user/sbyrapu/input/ml-20m/ratings.csv")
#header = data.first() #extract header
#data = data.filter(lambda line: line != header)

#Splitting the dataset into train(0.8) and test(0.2)
#x_train, x_test = data.randomSplit([0.8, 0.2], 17)
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
x_train=sc.textFile("/user/sbyrapu/input/train_df_final")
x_test = sc.textFile("/user/sbyrapu/input/test_df_final")

movies=sc.textFile("/user/sbyrapu/input/ml-20m/movies.csv")

# cache data
x_train.cache()
x_test.cache()

def makePairs(row):
    (user, ratings) = row
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates(row):
    (userID, ratings) = row
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):  
    numPairs = 0
    sum_xx = 0
    sum_yy = 0
    sum_xy = 0
    
    for row in ratingPairs:
        ratingX, ratingY = row
        ratingX = float(ratingX)
        ratingY = float(ratingY)
        sum_xx += float(ratingX) * float(ratingX)
        sum_yy += float(ratingY) * float(ratingY)
        sum_xy += float(ratingX) * float(ratingY)
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

def computeMovieMeanRating(ratings):
    counts_by_movieId = ratings.countByKey()
    sum_ratings = ratings.reduceByKey(lambda x, y: x+y)
    movie_avgs = sum_ratings.map(lambda x: (x[0], x[1]/counts_by_movieId[x[0]]))

    movie_avg_dict = movie_avgs.collectAsMap()
    return movie_avg_dict

def normalizeRating(x, mean):
    movie_id, rating = x
    norm_rating = rating - mean
    return (movie_id, norm_rating)

#ratings=x_train
ratings = x_test.map(lambda l: l.split(',')).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))
print("\nEvaluate recommendation model...")
#print(ratings.values())
movie_mean_ratings = computeMovieMeanRating(ratings.values())
normalized_ratings = ratings.mapValues(lambda x: normalizeRating(x, movie_mean_ratings[x[0]]))
ratings = normalized_ratings

# Emit every movie rated together by the same user.
# Self-join to find every combination.
ratingsPartitioned = ratings.partitionBy(100)
joinedRatings = ratingsPartitioned.join(ratingsPartitioned)

# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(100)


# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()


moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).persist()

#We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
#moviePairSimilarities = moviePairRatings_df.withColumn('movipair_cos_sim',computeCosineSimilarity(moviePairRatings_df._2.data.element)).cache()
#model_df=moviePairSimilarities.toDF()

scoreThreshold = 0.85
coOccurenceThreshold = 10

def filter_func(x):
    pair = x[0]
    sim = x[1]
    return (pair[0] == 83 or pair[1] == 83) and sim[0] > scoreThreshold and sim[1] > coOccurenceThreshold

filteredResults = moviePairSimilarities.filter(filter_func)
filteredResults.take(5)

# Sort by quality score.
results = filteredResults.take(10)

print("Top 10 similar movies for " + str([83]))
for result in results:
    (pair, sim) = result
    similarMovieID = pair[0]
    if (similarMovieID == 83):
        similarMovieID = pair[1]
        
        movie_name = movies.filter(movies.movieId == similarMovieID).rdd.collect()[0][1]
        print(f'{movie_name:20}\nscore: {str(sim[0])} \tstrength: {str(sim[1])}')



#moviePairSimilarities_df=moviePairSimilarities.toDF()
def computeScores(ratings):  
    sum_yy = 0
    sum_xy = 0

    from pyspark.sql.types import DoubleType
    preds_df=spark.createDataFrame([], ['userId','movieId','rating','score'])
    preds=[]
    
    for row in ratings:
        print("row!!!")
        print(row)
        pair=tuple()
        pair=row
        userId=pair[0]
        movieId=pair[1][0]
        ratingY=pair[1][1]
        #print("userId=",userId,"movieId=",movieId,"ratingY=",ratingY)
        userId=int(userId)
        movieId = int(movieId)
        ratingY = float(ratingY)
        filteredresults= moviePairSimilarities.filter(lambda x: (x[0][0] == movieId) \
         or (x[0][1] == movieId)and x[1][0]> scoreThreshold and x[1][1] > coOccurenceThreshold)
        results= filteredresults.map(lambda x: (x[1][0], x[0])).sortByKey(ascending = False).take(10)


        for result in results:
            (sim, pair_res) = result
            res_sim=ratings.filter(lambda x: (int(x[0])==userId) and (float(x[1][0]) == float(pair_res[1])))
            rating_sim=res_sim.map(lambda x: float(x[1][1])).collect()
            print(rating_sim)
            if(len(rating_sim)>0):
             sim= float(sim)
             sum_xy=sum_xy+sum([sim*val for val in rating_sim])
             sum_yy+=sim

        numerator = sum_xy
        denominator = sum_yy

        score = 0
        if (denominator):
          score = (numerator / (float(denominator)))
       
        preds_df=preds_df.union([userId,movieId,ratingY,score])
        #return score
        

    return (preds_df)

ratings_test = x_test.map(lambda l: l.split(',')).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

#res=ratings_test.take(5)
#print(res.take(5))
print("Scores!!!!")
result_df=computeScores(ratings_test)
#print(result_df.show(n=20))


predictions=result_df.select("rating","score")
predictions=predictions.na.drop()



#preds=ratings_test.mapValues(computeScores).persist()

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
evaluator.setLabelCol("rating").setPredictionCol("score")
mae=evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
MSE=evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
RMSE=evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print('------------------------The out-of-sample MSE, RMSE, MAE of rating predictions is--------------------',MSE,RMSE,mae)



result_df.write.mode("overwrite").csv("/user/sbyrapu/input/preds_df_Item_Item_CF")



