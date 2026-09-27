# Movie Recommendation System

A content-based movie recommender that suggests similar movies using
TF-IDF vectorization and cosine similarity over plot, genre, and director.

## What it does
- Builds a combined "content soup" per movie (overview + weighted genres
  + director)
- Vectorizes it with `TfidfVectorizer`
- Computes pairwise `cosine_similarity` between all movies
- Returns the top-N most similar movies for any given title

## Tech stack
Python, pandas, scikit-learn (TF-IDF, cosine similarity)

## How to run
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python movie_recommender.py
```

## Example output
```
Because you liked 'Inception', you might also like:
       title                  genres  director  similarity
  The Matrix           Action Sci-Fi Wachowski       0.398
      Avatar Action Adventure Sci-Fi   Cameron       0.365
Interstellar  Adventure Drama Sci-Fi     Nolan       0.364
```

## Dataset
Ships with a small hand-crafted movie catalog so it runs with zero setup.
For a bigger, more resume-credible version, swap in the public
[TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
dataset from Kaggle.

## Future improvements
- Scale up to the full TMDB dataset
- Replace TF-IDF with sentence embeddings for better semantic matching
- Wrap in a Streamlit UI for an interactive demo
