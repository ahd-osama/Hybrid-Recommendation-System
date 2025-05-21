#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import pickle

# Load or import your data and model:
movies_data = pd.read_csv("data/cleaned_movies.csv")  # example
rating_data = pd.read_csv("data/ratings.csv")         # example
cosine_sim = np.load("ui/..\data\cosine_sim.npy")           # example saved numpy array
with open("data/svd_model.pkl", 'rb') as f:
    svd = pickle.load(f)


# In[6]:


import numpy as np
import pandas as pd

def normalize(arr):
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def hybrid_recommendations(user_id, top_n=10, weight_cf=0.7, weight_cb=0.3, rating_threshold=4):
    # 1. Movies user liked (content side)
    liked_movies = rating_data[(rating_data['userId'] == user_id) & (rating_data['rating'] >= rating_threshold)]['movieId'].tolist()
    
    if not liked_movies:
        return f"No liked movies found for user {user_id}."
    
    # 2. Content-based similarity scores: average similarity across liked movies
    sim_scores = np.zeros(len(movies_data))
    for movie_id in liked_movies:
        try:
            idx = movies_data[movies_data['movieId'] == movie_id].index[0]
            sim_scores += cosine_sim[idx]
        except IndexError:
            continue
    sim_scores /= len(liked_movies)
    
    # Normalize content similarity scores
    sim_scores_norm = normalize(sim_scores)
    
    # 3. Collaborative filtering scores: predicted ratings for all movies by user
    cf_scores = []
    for idx, movie_id in enumerate(movies_data['movieId']):
        pred = svd.predict(user_id, movie_id)
        cf_scores.append(pred.est)
    cf_scores = np.array(cf_scores)
    
    # Normalize collaborative filtering scores
    cf_scores_norm = normalize(cf_scores)
    
    # 4. Combine scores with weights
    hybrid_scores = weight_cf * cf_scores_norm + weight_cb * sim_scores_norm
    
    # 5. Exclude movies already rated by user
    rated_movies = rating_data[rating_data['userId'] == user_id]['movieId'].tolist()
    rated_indices = movies_data[movies_data['movieId'].isin(rated_movies)].index
    hybrid_scores[rated_indices] = -1  # exclude
    
    # 6. Get top N recommendations
    recommended_indices = hybrid_scores.argsort()[::-1][:top_n]
    recommendations = movies_data.iloc[recommended_indices][['title', 'genres']].copy()
    recommendations['hybrid_score'] = hybrid_scores[recommended_indices]
    
    return recommendations.reset_index(drop=True)


