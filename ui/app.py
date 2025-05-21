import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from models.Content_based import get_content_recommendations
from models.hybrid_model import hybrid_recommendations

st.set_page_config(page_title="Movie Recommender", layout='wide')

with st.sidebar:
    selected = option_menu(
        menu_title="Choose a model",  
        options=["Collaberative Filtering", "Content-Based", "Hybrid"],   
        menu_icon="film",  
        default_index=0,
        styles={
            "nav-link-selected": {
                "background-color": "#0a7cc9",  
            },
        }
    )

st.title("🎬 Movie Recommendation System")
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
    except pickle.UnpicklingError:
        st.error("⚠️ File exists but could not be unpickled. Is it a valid pickle file?")
    except Exception as e:
        st.error(f"🔥 Unexpected error: {e}")
    return None

def show_recommendations(title, recs):
    st.subheader(f"Top Recommendations based on {title}:")
    for movie in recs:
        st.write(f"• {movie}")

def get_top_n_recommendations(model, user_id, movies_df, rating_data, n=5):
    # Get movies the user has already rated
    rated_movies = rating_data[rating_data['userId'] == user_id]['movieId'].tolist()
    
    # Predict ratings for all movies not rated yet
    predictions = []
    for movie_id in movies_df['movieId']:
        if movie_id not in rated_movies:
            pred = model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
    
    # Sort by predicted rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top n movie titles and scores
    top_n = [(movies_df[movies_df['movieId'] == movie_id]['title'].values[0], rating) 
             for movie_id, rating in predictions[:n]]
    return top_n

# Collaborative Filtering 
if selected == "Collaberative Filtering":
    st.header("Collaborative Filtering")

    movies_df = pd.read_csv("/Users/ahdosama/Documents/Hybrid-Movie-Recommendation-System-Project/data/movies.csv")
    rating_data = pd.read_csv("/Users/ahdosama/Documents/Hybrid-Movie-Recommendation-System-Project/data/ratings.csv")

    user_id = st.number_input("Enter your User ID:", min_value=1)

    model = load_model("/Users/ahdosama/Documents/Hybrid-Movie-Recommendation-System-Project/data/svd_model.pkl")

    if st.button("Recommend Movies"):
        if model:
            try:
                recommendations = get_top_n_recommendations(model, user_id, movies_df, rating_data, n=5)
                st.subheader("Top Collaborative Filtering Recommendations:")
                recommendations_df = pd.DataFrame(recommendations, columns=['Movie Title', 'Predicted Rating'])
                st.dataframe(recommendations_df)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
        else:
            st.warning("Collaborative model not loaded.")

# Content-Based Filtering
elif selected == "Content-Based":
    st.header("Content-Based Filtering")

    movies = pd.read_csv("/Users/ahdosama/Documents/Hybrid-Movie-Recommendation-System-Project/data/cleaned_movies.csv")
    selected_title = st.selectbox("Choose a movie:", movies['title'].tolist())

    if st.button("Recommend Similar Movies"):
        recommendations = get_content_recommendations(selected_title, top_n=5)
        st.subheader("Similar Movies:")
        st.dataframe(recommendations)

# Hybrid Recommendation
elif selected == "Hybrid":
    st.header("Hybrid Recommendation")

    user_id = st.number_input("Enter your User ID:", min_value=1, key="hybrid_user")
    movie_title = st.text_input("Enter a movie you like:", key="hybrid_movie")
    weight_cf = st.slider("Collaborative Filtering Weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    weight_cb = st.slider("Content-Based Filtering Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    # Normalize weights to always sum to 1
    total_weight = weight_cf + weight_cb
    if total_weight == 0:
        weight_cf, weight_cb = 0.7, 0.3  
    else:
        weight_cf /= total_weight
        weight_cb /= total_weight

    if st.button("Recommend Movies"):
        try:
            recommendations = hybrid_recommendations(user_id, top_n=5, weight_cf=weight_cf, weight_cb=weight_cb, rating_threshold=4)
            st.dataframe(recommendations)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
