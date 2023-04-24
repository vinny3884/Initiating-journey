#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


# Load the ratings data
ratings_df = pd.read_csv('ratings.csv')
# Load the movies data
movies_df = pd.read_csv('movies.csv')


# In[4]:


# View the first 5 rows of the ratings data
print(ratings_df.head())
# View the first 5 rows of the movies data
print(movies_df.head())


# In[5]:


# Merge the ratings and movies dataframes
ratings_movies_df = pd.merge(ratings_df, movies_df, on='movieId')


# In[6]:


# Remove the 'timestamp' column
ratings_movies_df.drop('timestamp', axis=1, inplace=True)


# In[8]:


ratings_movies_df.head()


# In[9]:


# Create a pivot table
user_movie_ratings = ratings_movies_df.pivot_table(index='userId', columns='movieId', values='rating')


# In[13]:


user_movie_ratings.head()


# In[ ]:





# In[12]:


# Replace NaN with 0
user_movie_ratings.fillna(0, inplace=True)


# In[14]:


# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(user_movie_ratings)


# In[15]:


def get_similar_users(user_id, cosine_sim_matrix):
    # Get the row index for the user id
    user_index = user_id - 1
    # Get the cosine similarity scores for the user
    user_scores = cosine_sim_matrix[user_index]
    # Sort the scores in descending order and get the top 10 users
    similar_users = np.argsort(-user_scores)[:10]
    # Return the similar user ids
    return similar_users + 1

# Get the similar users for user 1
similar_users = get_similar_users(1, cosine_sim_matrix)

# Print the similar user ids
print(similar_users)

