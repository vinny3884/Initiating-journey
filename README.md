# Initiating-journey
The code loads the ratings and movies data into Pandas dataframes, merges them, and creates a pivot table of user ratings. It then fills missing ratings with 0 and computes the cosine similarity matrix between users using the cosine_similarity function from the sklearn.metrics.pairwise module. The get_similar_users function takes a user id and the cosine similarity matrix and returns the top 10 similar user ids for that user. Finally, the code calls get_similar_users with user id 1 and prints the similar user ids.

