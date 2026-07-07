"""Ranking-as-a-service API. Run with: uvicorn api.main:app --reload"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from api.schemas import Event
from models.embedding import EmbeddingRanker
from models.movielens import load_movielens_events, load_movielens_item_text
from models.popularity import PopularityRanker

rankers: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    interactions = load_movielens_events()
    item_text = load_movielens_item_text()
    rankers["popularity"] = PopularityRanker().fit(interactions)
    rankers["embedding"] = EmbeddingRanker().fit(interactions, item_text)
    yield


app = FastAPI(title="Shaped-Style Ranking API", lifespan=lifespan)


class RankRequest(BaseModel):
    events: list[Event]
    k: int = 10
    ranker: Literal["embedding", "popularity"] = "embedding"


class RankResponse(BaseModel):
    ranker: str
    ranked_items: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest):
    seen_item_ids = [e.item_id for e in request.events]

    ranked_items = rankers[request.ranker].rank(seen_item_ids, k=request.k)
    used_ranker = request.ranker
    if not ranked_items and request.ranker == "embedding":
        # cold start: no known items to build a user vector from
        used_ranker = "popularity"
        ranked_items = rankers["popularity"].rank(seen_item_ids, k=request.k)

    return RankResponse(ranker=used_ranker, ranked_items=ranked_items)
