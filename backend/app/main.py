import os
import pickle
import httpx
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is not set")

app = FastAPI(
    title="Movie Recommendation System",
    description="A simple movie recommendation system using the TMDB API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

DF_PATH = MODELS_DIR / "movie_df.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
INDICES_PATH = MODELS_DIR / "indices.pkl"