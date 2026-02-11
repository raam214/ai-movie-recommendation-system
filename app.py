import base64
import streamlit as st
import pandas as pd
import requests
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Movie Recommendation System",
    layout="wide"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")

# ---------------- LOAD & PROCESS DATA ----------------


@st.cache_data
def load_and_prepare_data():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies = movies.merge(credits, on="title")

    movies = movies[[
        "movie_id", "title", "overview",
        "genres", "keywords", "cast", "crew"
    ]]

    import ast

    def convert(text):
        return [i["name"] for i in ast.literal_eval(text)]

    def convert_cast(text):
        return [i["name"] for i in ast.literal_eval(text)[:3]]

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].fillna("")

    movies["tags"] = (
        movies["overview"] + " " +
        movies["genres"].apply(lambda x: " ".join(x)) + " " +
        movies["keywords"].apply(lambda x: " ".join(x)) + " " +
        movies["cast"].apply(lambda x: " ".join(x)) + " " +
        movies["crew"].apply(lambda x: " ".join(x))
    )

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return movies, similarity


movies, similarity = load_and_prepare_data()

# ---------------- BACKGROUND ----------------


def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background(os.path.join(BASE_DIR, "assets", "background.jpg"))

# ---------------- TMDB API ----------------
TMDB_API_KEY = "f056f7cdd9bfc9d260f4ccf24c68d693"


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except:
        return None

# ---------------- RECOMMENDER ----------------


def recommend(movie_title):
    index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters


# ---------------- UI ----------------
st.markdown(
    """
    <h1 style="
        text-align:center;
        color:#E50914;
        text-shadow: 2px 2px 8px black;
        font-size:48px;
        font-weight:800;
    ">
        🎬 AI Movie Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center; font-size:18px; font-weight:600; color:pink;">
        Netflix-style content based recommender
    </p>
    <p style="text-align:center; font-size:13px; color:#ccc; margin-top:-8px;">
        Built with ❤️ using NLP & Streamlit · Created by <b>@214_raam</b>
    </p>
    """,
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i], width=230)
            else:
                st.write("Poster not available")
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>{names[i]}</p>",
                unsafe_allow_html=True
            )

st.markdown("<div style='height:45vh'></div>", unsafe_allow_html=True)
