"""
PROJECT: Movie Recommendation System (Intermediate Level)
-----------------------------------------------------------
Goal: Given a movie a user likes, recommend similar movies using
content-based filtering (NLP + similarity, not just a rating average).

Why this project is a good resume piece:
- Uses real NLP techniques: text vectorization (TF-IDF) and similarity
  search (cosine similarity) -- concepts that show up constantly in
  interviews and in production recommender/search systems.
- Demonstrates feature engineering with mixed data (text + categorical
  + numeric) combined into a single "content soup".
- Naturally extends into a Streamlit app (you've already built one for
  your YouTube Trend Console, so wiring a UI on top of this will be
  quick) and into more advanced versions later (collaborative
  filtering, matrix factorization, embeddings from a language model).

Dataset: A small hand-crafted movie catalog is included so the script
runs with zero downloads. Swap `MOVIES` below with the public
TMDB 5000 Movies dataset (Kaggle) for a bigger, more resume-credible
version -- same code, just load genres/keywords/overview columns from
the CSV instead of the list below.
"""

from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# 1. DATA (replace with pd.read_csv("tmdb_5000_movies.csv") for the
#    real dataset in your portfolio version)
# ---------------------------------------------------------------------
MOVIES = [
    {"title": "The Matrix", "genres": "Action Sci-Fi", "director": "Wachowski",
     "overview": "A hacker discovers reality is a simulation and joins a rebellion against machines."},
    {"title": "Inception", "genres": "Action Sci-Fi Thriller", "director": "Nolan",
     "overview": "A thief who steals secrets through dream-sharing technology is given a chance to erase his past."},
    {"title": "Interstellar", "genres": "Adventure Drama Sci-Fi", "director": "Nolan",
     "overview": "A team of explorers travel through a wormhole in space to ensure humanity's survival."},
    {"title": "The Dark Knight", "genres": "Action Crime Drama", "director": "Nolan",
     "overview": "Batman faces the Joker, a criminal mastermind who plunges Gotham into anarchy."},
    {"title": "John Wick", "genres": "Action Crime Thriller", "director": "Stahelski",
     "overview": "An ex-hitman comes out of retirement to track down the gangsters that took everything from him."},
    {"title": "The Avengers", "genres": "Action Adventure Sci-Fi", "director": "Whedon",
     "overview": "Earth's mightiest heroes assemble to stop an alien invasion led by a powerful villain."},
    {"title": "Titanic", "genres": "Drama Romance", "director": "Cameron",
     "overview": "A young couple from different social classes fall in love aboard a doomed ocean liner."},
    {"title": "The Notebook", "genres": "Drama Romance", "director": "Cassavetes",
     "overview": "A poor young man and a rich young woman fall passionately in love despite their families."},
    {"title": "Avatar", "genres": "Action Adventure Sci-Fi", "director": "Cameron",
     "overview": "A paraplegic marine on an alien planet becomes torn between orders and protecting the world."},
    {"title": "Gladiator", "genres": "Action Adventure Drama", "director": "Scott",
     "overview": "A betrayed Roman general fights his way back to seek revenge against a corrupt emperor."},
    {"title": "The Shawshank Redemption", "genres": "Drama", "director": "Darabont",
     "overview": "Two imprisoned men bond over years, finding solace and eventual redemption through decency."},
    {"title": "Se7en", "genres": "Crime Drama Thriller", "director": "Fincher",
     "overview": "Two detectives hunt a serial killer who uses the seven deadly sins as his motives."},
]

df = pd.DataFrame(MOVIES)

# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING: build a combined "content soup" per movie
# ---------------------------------------------------------------------
def build_content_soup(row: pd.Series) -> str:
    # Repeat genres/director so they carry more weight than plain overview words
    return " ".join([
        row["overview"],
        (row["genres"] + " ") * 3,
        (row["director"] + " ") * 2,
    ])

df["content"] = df.apply(build_content_soup, axis=1)

# ---------------------------------------------------------------------
# 3. VECTORIZE WITH TF-IDF
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# ---------------------------------------------------------------------
# 4. COMPUTE PAIRWISE COSINE SIMILARITY
# ---------------------------------------------------------------------
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_index = pd.Series(df.index, index=df["title"])


def recommend(title: str, top_n: int = 5) -> pd.DataFrame:
    """Return the top_n movies most similar in content to `title`."""
    if title not in title_to_index:
        raise ValueError(f"'{title}' not found in catalog. Available: {list(df['title'])}")

    idx = title_to_index[title]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]  # exclude itself

    result = df.iloc[[i for i, _ in scores]][["title", "genres", "director"]].copy()
    result["similarity"] = [round(score, 3) for _, score in scores]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------------------
if __name__ == "__main__":
    for query in ["Inception", "Titanic", "John Wick"]:
        print(f"\nBecause you liked '{query}', you might also like:")
        print(recommend(query, top_n=4).to_string(index=False))
