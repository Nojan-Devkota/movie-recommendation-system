import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

from schema.schema import MovieCard, MovieDetails

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is not set")


def make_img_url(path: str) -> str:
    if not path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{path}"


async def tmdb_get(
    path: str, params: dict[str, Any], client: httpx.AsyncClient
) -> dict[str, Any]:

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    url = f"{TMDB_BASE_URL}/{path}"

    try:
        response = await client.get(url, params=q)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to fetch data from TMDB: {type(e).__name__} | {repr(e)}",
        )

    return response.json()


def tmdb_cards_from_results(
    results: List[Dict[str, Any]], limit: int = 20
) -> List[MovieCard]:

    outputs: List[MovieCard] = []
    try:
        for i in (results or [])[:limit]:
            if i.get("id"):
                outputs.append(
                    MovieCard(
                        tmdb_id=int(i.get("id")),
                        title=i.get("title"),
                        poster_path=make_img_url(i.get("poster_path")),
                        release_date=i.get("release_date"),
                        vote_average=i.get("vote_average"),
                    )
                )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create movie cards: {type(e).__name__} | {repr(e)}",
        )

    return outputs


async def tmdb_movie_details(movie_id: int, client: httpx.AsyncClient) -> MovieDetails:
    data = await tmdb_get(path=f"movie/{movie_id}", params={}, client=client)
    try:
        return MovieDetails(
            tmdb_id=int(data.get("id")),
            title=data.get("title"),
            overview=data.get("overview"),
            release_date=data.get("release_date"),
            poster_path=make_img_url(data.get("poster_path")),
            backdrop_path=make_img_url(data.get("backdrop_path")),
            genres=data.get("genres", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch movie details: {type(e).__name__} | {repr(e)}",
        )

async def tmdb_search(
    query: str,
    client: httpx.AsyncClient,
    page: int = 1
) -> Dict[str, Any]:
    data = await tmdb_get(
        path="search/movie",
        client=client,
        params={
            "query": query,
            "language": "en-US",
            "page": page,
        },
    )
    
    return data

async def tmdb_search_first(
    query: str,
    client: httpx.AsyncClient,
    page: int = 1
) -> Optional[Dict]:
    data = await tmdb_search(query, client, page)
    if data.get("results"):
        return tmdb_cards_from_results(data.get("results"))[0]
    return None
    
async def attach_tmdb_card_by_title(
    title: str,
    client: httpx.AsyncClient,
) -> Optional[MovieCard]:
    try:
        data = await tmdb_search_first(title, client)
        if not data:
            raise None

        return MovieCard(
            tmdb_id=data.get("id"),
            title=data.get("title"),
            poster_path=make_img_url(data.get("poster_path")),
            release_date=data.get("release_date"),
            vote_average=data.get("vote_average"),
        )

    except Exception as e:
        return None
