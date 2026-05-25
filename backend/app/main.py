import os
import pickle
import httpx
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional, Any
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from utils.util_functions import tfidf_movie_recommendations
from schema.schema import TFIDFRecommendations

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


df: Optional[pd.DataFrame] = None
indices: Optional[pd.Series] = None
tfidf: Optional[Any] = None
tfidf_matrix: Optional[Any] = None
client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
def startup_event():
    global df, indices, tfidf, tfidf_matrix
    df = pd.read_pickle(DF_PATH)
    indices = pd.read_pickle(INDICES_PATH)
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MATRIX_PATH, "rb"))
    
    global client
    client = httpx.AsyncClient()
    
    return {"message": "Startup event completed"}

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

@app.get("/")
def health_check():
    return JSONResponse(status_code=200, content={"message": "OK"})


@app.get("/recommend/tfidf", response_model=List[TFIDFRecommendations])
def get_tfidf_recommendations(
    title: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=50),
) -> List[TFIDFRecommendations]:
    
    if df is None or indices is None or tfidf is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    recommendations = tfidf_movie_recommendations(title, indices, tfidf_matrix, df, n = n)
    return [
        TFIDFRecommendations(title= recommendations[i][0], score= recommendations[i][1]) for i in range(len(recommendations))
    ]
