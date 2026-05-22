from pydantic import BaseModel
from typing import List, Optional


class MovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    
class MovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: str
    release_date: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: List[dict]
    
class TFIDFRecommendations(BaseModel):
    title: str
    score: float
    tmdb:Optional[MovieCard] = None
    
class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecommendations]
    genre_recommendations: List[MovieCard]
    
