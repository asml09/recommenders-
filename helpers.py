import pandas as pd 
import numpy as np 



def get_movies_seen_first(ratings):
    '''
    Function to get movie_id for movies that are most often seen first. 
    ratings_matrix must have correct columns

    Input:
    ratings_matrix(pd.DataFrame) - ratings matrix with columns: 
                                    user, movie, rating, timestamp
    Output:
    movie_ids (np.array) - numpy array of movies that are seen first most often
    '''

    time_first_movie = ratings.groupby('user')['timestamp'].min()

    movies_first = {}
    for user in time_first_movie.index:
        movie = ratings[
            (ratings.user == user) & 
            (ratings.timestamp == time_first_movie[user])
        ]['movie']
        if len(movie)> 1:
            for m in movie:
                if m in movies_first:
                    movies_first[m] += 1
                else:
                    movies_first[m] = 1
        else:
            if m in movies_first:
                movies_first[m] += 1
            else:
                movies_first[m] = 1
    
    movies_first_df = pd.Series(movies_first.values(), index=movies_first.keys())
    return movies_first_df




