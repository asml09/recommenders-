# recommenders-

We started off with making a factorization recommender. We wanted to identify cold start cases and how they would do with various models. To do this, we implemented a funciton that finds which movies exist in the test set but not the train set. We also considered a few ideas for what to do in cold start cases. One of these is, if a user has hardly rated any movies, to go off of the average rating that everyone else gave rather than use the little data we have. 
