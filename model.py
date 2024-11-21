# model.py

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.cluster import KMeans
# from ast import literal_eval
# import streamlit as st  # Import Streamlit to use caching decorators

# @st.cache_data
# def load_and_process_data():
#     # Load and merge the data
#     dataset1 = pd.read_csv('tmdb_5000_credits.csv')
#     dataset2 = pd.read_csv('tmdb_5000_movies.csv')

#     dataset1.columns = ['id', 'title', 'cast', 'crew']
#     movies = dataset2.merge(dataset1, on='id')

#     # Rename columns to resolve conflicts
#     movies.rename(columns={'title_x': 'title'}, inplace=True)
#     movies.drop(columns=['title_y'], inplace=True)

#     # Define helper functions
#     def extractFeature(obj):
#         if isinstance(obj, str):
#             obj = literal_eval(obj)
#         if isinstance(obj, list):
#             return [d['name'] for d in obj]
#         return []

#     def topCastNames(cast_list, top_n=5):
#         if isinstance(cast_list, list):
#             names = [member['name'] for member in cast_list[:top_n]]
#             return names
#         return []

#     # Process features
#     movies['genres'] = movies['genres'].apply(extractFeature)
#     movies['keywords'] = movies['keywords'].apply(extractFeature)
#     movies['cast'] = movies['cast'].apply(literal_eval)
#     movies['castNames'] = movies['cast'].apply(topCastNames)

#     return movies

# @st.cache_resource
# def create_cluster_models(movies, k=100):
#     # Prepare data for clustering
#     genres = movies['genres']
#     keywords = movies['keywords']
#     cast = movies['castNames']

#     # Encode features
#     genreMlb = MultiLabelBinarizer()
#     encodedGenres = genreMlb.fit_transform(genres)

#     keywordsMlb = MultiLabelBinarizer()
#     encodedKeywords = keywordsMlb.fit_transform(keywords)

#     castMlb = MultiLabelBinarizer()
#     encodedCast = castMlb.fit_transform(cast)

#     # Create and fit KMeans models
#     genresKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     genresKmeans.fit(encodedGenres)

#     keywordsKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     keywordsKmeans.fit(encodedKeywords)

#     castKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     castKmeans.fit(encodedCast)

#     # Assign cluster labels to movies
#     movies['genresClusters'] = genresKmeans.labels_
#     movies['keywordsClusters'] = keywordsKmeans.labels_
#     movies['castClusters'] = castKmeans.labels_

#     # Optional: Store encoders and models
#     models_and_encoders = {
#         'genres': {'kmeans': genresKmeans, 'mlb': genreMlb},
#         'keywords': {'kmeans': keywordsKmeans, 'mlb': keywordsMlb},
#         'cast': {'kmeans': castKmeans, 'mlb': castMlb}
#     }

#     return movies, models_and_encoders

# model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from ast import literal_eval
import pickle

def load_and_process_data():
    # Load and merge the data
    dataset1 = pd.read_csv('tmdb_5000_credits.csv')
    dataset2 = pd.read_csv('tmdb_5000_movies.csv')

    dataset1.columns = ['id', 'title', 'cast', 'crew']
    movies = dataset2.merge(dataset1, on='id')

    # Rename columns to resolve conflicts
    movies.rename(columns={'title_x': 'title'}, inplace=True)
    movies.drop(columns=['title_y'], inplace=True)

    # Define helper functions
    def extractFeature(obj):
        if isinstance(obj, str):
            obj = literal_eval(obj)
        if isinstance(obj, list):
            return [d['name'] for d in obj]
        return []

    def topCastNames(cast_list, top_n=5):
        if isinstance(cast_list, list):
            names = [member['name'] for member in cast_list[:top_n]]
            return names
        return []

    # Process features
    movies['genres'] = movies['genres'].apply(extractFeature)
    movies['keywords'] = movies['keywords'].apply(extractFeature)
    movies['cast'] = movies['cast'].apply(literal_eval)
    movies['castNames'] = movies['cast'].apply(topCastNames)

    return movies

def create_cluster_models(movies, k=100):
    # Prepare data for clustering
    genres = movies['genres']
    keywords = movies['keywords']
    cast = movies['castNames']

    # Encode features
    genreMlb = MultiLabelBinarizer()
    encodedGenres = genreMlb.fit_transform(genres)

    keywordsMlb = MultiLabelBinarizer()
    encodedKeywords = keywordsMlb.fit_transform(keywords)

    castMlb = MultiLabelBinarizer()
    encodedCast = castMlb.fit_transform(cast)

    # Create and fit KMeans models
    genresKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    genresKmeans.fit(encodedGenres)

    keywordsKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    keywordsKmeans.fit(encodedKeywords)

    castKmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    castKmeans.fit(encodedCast)

    # Assign cluster labels to movies
    movies['genresClusters'] = genresKmeans.labels_
    movies['keywordsClusters'] = keywordsKmeans.labels_
    movies['castClusters'] = castKmeans.labels_

    return movies

def save_precomputed_data(movies):
    # Save the movies DataFrame with cluster labels
    movies.to_csv('movies_with_clusters.csv', index=False)



# Recommendation functions remain the same
def recommendationsOnGenres(movieTitle, movies, count=5):
    selectedMovie = movies[movies['title'] == movieTitle]
    if selectedMovie.empty:
        print(f"Movie '{movieTitle}' not found.")
        return pd.DataFrame()
    selectedCluster = selectedMovie['genresClusters'].values[0]
    clusterMovies = movies[movies['genresClusters'] == selectedCluster]
    recommendations = clusterMovies[clusterMovies['title'] != movieTitle]
    recommendations = recommendations.sort_values(by='popularity', ascending=False)
    return recommendations.head(count)

def recommendationsOnKeywords(movieTitle, movies, count=5):
    selectedMovie = movies[movies['title'] == movieTitle]
    if selectedMovie.empty:
        print(f"Movie '{movieTitle}' not found.")
        return pd.DataFrame()
    selectedCluster = selectedMovie['keywordsClusters'].values[0]
    clusterMovies = movies[movies['keywordsClusters'] == selectedCluster]
    recommendations = clusterMovies[clusterMovies['title'] != movieTitle]
    recommendations = recommendations.sort_values(by='popularity', ascending=False)
    return recommendations.head(count)

def recommendationsOnCast(movieTitle, movies, count=5):
    selectedMovie = movies[movies['title'] == movieTitle]
    if selectedMovie.empty:
        print(f"Movie '{movieTitle}' not found.")
        return pd.DataFrame()
    selectedCluster = selectedMovie['castClusters'].values[0]
    clusterMovies = movies[movies['castClusters'] == selectedCluster]
    recommendations = clusterMovies[clusterMovies['title'] != movieTitle]
    recommendations = recommendations.sort_values(by='popularity', ascending=False)
    return recommendations.head(count)

# Run this part only when computing clusters offline
if __name__ == '__main__':
    movies = load_and_process_data()
    movies = create_cluster_models(movies)
    save_precomputed_data(movies)
