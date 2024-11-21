# app.py

import streamlit as st
import pandas as pd
import model  # Importing model.py

# Load data and models
movies = model.load_and_process_data()
movies, models_and_encoders = model.create_cluster_models(movies)

st.title('Movie Recommendation System')
st.write('Get movie recommendations based on your favorite movies!')

# Option to choose recommendation method
option = st.selectbox(
    'Choose a recommendation method:',
    ('Genres', 'Keywords', 'Cast')
)

# Movie selection
movie_titles = movies['title'].tolist()
selected_movie_title = st.selectbox('Select a movie:', movie_titles)

# Number of recommendations
num_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=20, value=5)

# Get Recommendations button
if st.button('Get Recommendations'):
    if option == 'Genres':
        recommendations = model.recommendationsOnGenres(selected_movie_title, movies, num_recommendations)
    elif option == 'Keywords':
        recommendations = model.recommendationsOnKeywords(selected_movie_title, movies, num_recommendations)
    elif option == 'Cast':
        recommendations = model.recommendationsOnCast(selected_movie_title, movies, num_recommendations)
    else:
        recommendations = pd.DataFrame()

    if not recommendations.empty:
        st.write('Recommendations:')
        st.table(recommendations[['title']])
    else:
        st.write('No recommendations found.')
