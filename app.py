import base64
import streamlit as st
import pickle
import requests
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Movie Recommendation System",
    layout="wide"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, "model", "movies.pkl")
similarity_path = os.path.join(BASE_DIR, "model", "similarity.pkl")

# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    with open(similarity_path, "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


movies, similarity = load_data()

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


# ---------------- TMDB API KEY ----------------
TMDB_API_KEY = "f056f7cdd9bfc9d260f4ccf24c68d693"


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        }
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=10)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        return None

# ---------------- RECOMMENDATION LOGIC ----------------


def recommend(movie_title):
    index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


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
    <p style="text-align:center; font-size:18px; font-weight:600; color:black;">
        Netflix-style content based recommender
    </p>
    <p style="text-align:center; font-size:13px; color:#222; margin-top:-6px;">
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

# ================= FOOTER (ALWAYS AT BOTTOM) =================
# ---------- FORCE SPACE BEFORE FOOTER ----------
# Push footer to bottom using viewport height
st.markdown(
    "<div style='height: 45vh;'></div>",
    unsafe_allow_html=True
)
