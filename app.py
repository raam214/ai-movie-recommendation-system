import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD DATA (CSV ONLY) ----------------


@st.cache_data
def load_movies():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[[
        "movie_id", "title", "overview",
        "genres", "keywords", "cast", "crew"
    ]]

    movies.dropna(inplace=True)
    return movies


movies = load_movies()

# ---------------- FEATURE ENGINEERING ----------------


def combine(row):
    return (
        row["overview"] + " " +
        row["genres"] + " " +
        row["keywords"] + " " +
        row["cast"] + " " +
        row["crew"]
    )


movies["tags"] = movies.apply(combine, axis=1)

# ---------------- SIMILARITY ----------------


@st.cache_data
def compute_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    return cosine_similarity(vectors)


similarity = compute_similarity(movies)

# ---------------- RECOMMENDER ----------------


def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]


# ---------------- UI ----------------
st.title("🎬 AI Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies")
    for movie in recommendations:
        st.write("🎥", movie)
