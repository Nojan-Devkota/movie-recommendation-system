from typing import Any, Dict, List, Tuple
import pandas as pd
from pathlib import Path
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

DF_PATH = MODELS_DIR / "movie_df.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
INDICES_PATH = MODELS_DIR / "indices.pkl"


def _norm_title(title:str) -> str:
    return title.lower().strip()

def get_local_idx_by_title(title: str, indices: pd.Series) -> int:
    title = _norm_title(title)
    
    if title not in indices:
        raise HTTPException(status_code=404, detail=f"Movie title '{title}' not found in local database")
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    return int(idx)

def tfidf_movie_recommendations(title: str, indices: pd.Series, tfidf_matrix: Any, df: pd.DataFrame, n: int = 10) -> List[Tuple[str, float]]:
    idx = get_local_idx_by_title(title, indices)
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    
    return [(df['title'].iloc[i], sim_score[i]) for i in similar_idx]