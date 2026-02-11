import base64
import streamlit as st
import pandas as pd
import requests
import os
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Movie Recommendation System",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")

TMDB_API_KEY = "f056f7cdd9bfc9d260f4ccf24c68d693"

# ================= BACKGROUND =================


def set_background(image_path):
    with open(image_path, "rb") as f:
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

# ================= DATA LOAD =================


@st.cache_data
def load_and_prepare_data():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    movies = movies.merge(credits, on="title")

    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]

    def convert(text):
        return [i["name"] for i in ast.literal_eval(text)]

    def convert_cast(text):
        return [i["name"] for i in ast.literal_eval(text)[:3]]

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

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

# ================= TMDB HELPERS =================


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None


def get_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {"api_key": TMDB_API_KEY}
        data = requests.get(url, params=params, timeout=5).json()
        for v in data.get("results", []):
            if v["type"] == "Trailer" and v["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={v['key']}"
    except:
        pass
    return None


def get_tmdb_link(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"

# ================= RECOMMENDER =================


def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    return movie_list


# ================= UI =================
st.markdown(
    """
    <h1 style="text-align:center; color:#E50914; text-shadow:2px 2px 8px black;">
        🎬 AI Movie Recommendation System
    </h1>

    <p style="text-align:center; font-size:16px; font-weight:600; color:pink;">
        ML & NLP based content recommendation engine
    </p>

    <p style="text-align:center; font-size:13px; color:#ddd;">
        Built with ❤️ using Machine Learning & NLP · Created by <b>@214_raam</b>
    </p>
    """,
    unsafe_allow_html=True
)


selected_movie = st.selectbox("Select a movie", movies["title"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    # Mobile-safe: Streamlit auto-stacks columns on small screens
    cols = st.columns(5)

    for col, rec in zip(cols, recommendations):
        i = rec[0]
        movie_id = movies.iloc[i].movie_id
        title = movies.iloc[i].title

        with col:
            poster = fetch_poster(movie_id)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:320px; background:#111; border-radius:8px;'></div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>{title}</p>",
                unsafe_allow_html=True
            )

            trailer = get_trailer(movie_id)
            if trailer:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <a href="{trailer}" target="_blank"
                           style="display:inline-block;
                                  margin:6px 0;
                                  padding:6px 12px;
                                  background:#E50914;
                                  color:white;
                                  text-decoration:none;
                                  border-radius:6px;
                                  font-weight:600;">
                            🎬 Play Trailer
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <a href="{get_tmdb_link(movie_id)}" target="_blank"
                       style="display:inline-block;
                              margin-top:6px;
                              padding:6px 12px;
                              background:black;
                              color:#E50914;
                              border:1px solid #E50914;
                              text-decoration:none;
                              border-radius:6px;
                              font-weight:600;">
                        ▶ Watch Options
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

# ================= FOOTER SPACE =================
st.markdown("<div style='height:30vh'></div>", unsafe_allow_html=True)
