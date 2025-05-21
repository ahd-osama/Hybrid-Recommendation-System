#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv("data/cleaned_movies.csv")


# In[ ]:


rating_data = pd.read_csv("data/ratings.csv")


# In[2]:


# Initialize TF-IDF Vectorizer (you can tune parameters if needed)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the 'text' column into TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies_data['text'])

# Compute cosine similarity matrix between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")


# In[12]:


# Helper: map movie titles to their index in movies_data
title_to_index = pd.Series(movies_data.index, index=movies_data['title']).drop_duplicates()

def get_content_recommendations(title, top_n=10):
    # Check if movie exists in dataset
    if title not in title_to_index:
        return f"Movie '{title}' not found in dataset."
    
    idx = title_to_index[title]
    
    # Get pairwise similarity scores for this movie to all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself and get top N similar movies
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movie titles and genres
    recommended = movies_data.iloc[movie_indices][['title', 'genres']]
    recommended['similarity_score'] = [i[1] for i in sim_scores]
    
    return recommended.reset_index(drop=True)

# Example usage:
print("top similarity scores for this movie")
get_content_recommendations('toy story (1995)', top_n=5)


# This function takes a movie title and finds the most similar movies based on precomputed cosine similarity scores from TF-IDF features. It excludes the input movie itself and returns the top N most similar movies with their titles, genres, and similarity scores. This enables content-based recommendations by comparing movies’ textual metadata.
# 
# 

# In[10]:


def content_based_recommendations_for_user(user_id, rating_threshold=4, top_n=10):
    # Movies the user liked
    liked_movies = rating_data[(rating_data['userId'] == user_id) & (rating_data['rating'] >= rating_threshold)]['movieId'].tolist()
    
    if not liked_movies:
        return pd.DataFrame(columns=['title', 'genres', 'similarity_score']).append({'title': 'No liked movies found for user.', 'genres': '', 'similarity_score': 0}, ignore_index=True)
    
    sim_scores = np.zeros(len(movies_data))
    
    for movie_id in liked_movies:
        try:
            idx = movies_data[movies_data['movieId'] == movie_id].index[0]
            sim_scores += cosine_sim[idx]
        except IndexError:
            continue
    
    sim_scores = sim_scores / len(liked_movies)
    
    # Remove movies the user already rated
    rated_movie_indices = movies_data[movies_data['movieId'].isin(liked_movies)].index
    sim_scores[rated_movie_indices] = -1  # exclude already liked/rated movies
    
    recommended_indices = sim_scores.argsort()[::-1][:top_n]
    recommended = movies_data.iloc[recommended_indices][['title', 'genres']].copy()
    recommended['similarity_score'] = sim_scores[recommended_indices]
    
    return recommended.reset_index(drop=True)

# Example usage:
# user_id = 1
# recommendations = content_based_recommendations_for_user(user_id=user_id, top_n=5)
# print(f"Content-based recommendations for user {user_id}:\n")
# recommendations

