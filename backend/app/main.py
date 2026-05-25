from utils.tmdb_client import tmdb_cards_from_results
from schema.schema import MovieCard
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
from utils.tmdb_client import tmdb_get

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
async def startup_event():
    global df, indices, tfidf, tfidf_matrix, client
    df = pd.read_pickle(DF_PATH)
    indices = pd.read_pickle(INDICES_PATH)
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MATRIX_PATH, "rb"))

    client = httpx.AsyncClient()

    return {"message": "Startup event completed"}


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


@app.get("/")
def health_check():
    return JSONResponse(status_code=200, content={"message": "OK"})


@app.get("/home", response_model=List[MovieCard])
async def home(
    client: httpx.AsyncClient,
    category: str = Query(..., min_length=1),
    limit: int = Query(24, ge=1, le=100),
) -> List[MovieCard]:

    try:
        if category == "trending":
            data = await tmdb_get(
                path="trending/movie/week",
                params={
                    "language": "en-US",
                },
                client=client,
            )

            return tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in ["popular", "top_rated", "upcoming", "now_playing"]:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        data = await tmdb_get(
            path=f"/movie/{category}",
            params={
                "language": "en-US",
            },
            client=client,
        )

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {type(e).__name__} | {repr(e)}",
        )


@app.get("/recommend/tfidf", response_model=List[TFIDFRecommendations])
def get_tfidf_recommendations(
    title: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=50),
) -> List[TFIDFRecommendations]:

    if df is None or indices is None or tfidf is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    recommendations = tfidf_movie_recommendations(title, indices, tfidf_matrix, df, n=n)
    return [
        TFIDFRecommendations(title=recommendations[i][0], score=recommendations[i][1])
        for i in range(len(recommendations))
    ]
