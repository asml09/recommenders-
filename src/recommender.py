import logging
import numpy as np
import pandas as pd

import pyspark
from pyspark.sql import SparkSession
# from pyspark.sql.functions import lit
from pyspark.ml.recommendation import ALS


class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        # ...


    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        s_df = spark.createDataFrame(ratings)

        model = ALS(
            itemCol='movie',
            userCol='user',
            ratingCol='rating',
            nonnegative=True,    
            regParam=0.1,
            rank=10)
        
        self.recommender = model.fit(s_df)

        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        # requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        s_df = spark.createDataFrame(requests)
        predictions = self.recommender.transform(s_df)
        predictions = predictions.toPandas()
        self.logger.debug("finishing predict")
        return(predictions)


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
