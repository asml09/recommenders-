import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Loading Data ###
## Loading CSV files & converting to spark
p_df_train = pd.read_csv('data/training.csv')
p_df_f_test = pd.read_csv('data/fake_testing.csv')
p_df_requests = pd.read_csv('data/requests.csv')

def pd_to_spark(p_df):
    spark = SparkSession.builder.getOrCreate()
    s_df = spark.createDataFrame(p_df)
    return s_df

orig_train = pd_to_spark(p_df_train)
orig_f_test = pd_to_spark(p_df_f_test)
requests = pd_to_spark(p_df_requests)

## loading dat files & converting to spark
movie_data = pd.read_csv('data/movies.dat',
                                   sep="\t|::",
                                   names=['movie_id','title','genres'], 
                                   header=None, 
                                   engine="python")

user_data = pd.read_csv('data/users.dat',
                                   sep="\t|::",
                                   names=['user','gender','age','occupation','zip'], 
                                   header=None, 
                                   engine="python")


### density check function
def density(p_df_train):
    # get density from original data
    df = p_df_train
    n_rantings = df.count()
    n_users = df.select('user').distinct().count()
    n_movies = df.select('movie').distinct().count()
    density = n_rantings / (n_users * n_movies)
    return density

### tran test split
def traintestsplit():
    train, test = orig_train.randomSplit([0.8, 0.2], seed=427471138)
    return train, test


### Modelling
## initial model

# instantiate the model and set its parameters
als_model = ALS(
    itemCol='movie',
    userCol='user',
    ratingCol='rating',
    nonnegative=True,    
    regParam=0.1,
    rank=10)

# fitting
recommender = als_model.fit(train)

# transforming for the prediction
predictions = recommender.transform(test)
predictions.show()

