"""FastAPI service. Run: PYTHONPATH=src uvicorn homematch.api:app --reload"""
import json
import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path

import openai
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from homematch.config import OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, require_api_key
from homematch.recommend import stream_recommendation
from homematch.search import hybrid_search

# ---------- logging: one JSON line per request, easy to analyze later ----------
LOG_PATH = Path("logs/requests.jsonl")
logger = logging.getLogger("homematch")
if not logger.handlers:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(LOG_PATH))
    logger.addHandler(logging.StreamHandler())


def log_event(event: dict) -> None:
    logger.info(json.dumps(event))


# ---------- app ----------
app = FastAPI(title="HomeMatch", description="Natural-language real-estate search with hybrid retrieval")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


@lru_cache
def get_client() -> openai.OpenAI:
    """Created once and reused. Tests replace this with a fake."""
    return openai.OpenAI(api_key=require_api_key())


def ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def run_search(client, req: SearchRequest, request_id: str):
    try:
        return hybrid_search(client, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, req.query, k=req.k)
    except openai.OpenAIError as err:
        log_event({"request_id": request_id, "error": type(err).__name__, "query": req.query})
        raise HTTPException(status_code=502, detail="Upstream model error, please retry") from err


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest, client=Depends(get_client)) -> dict:
    """Retrieval only: parsed filters + matching listings (no LLM-written answer)."""
    request_id, start = uuid.uuid4().hex[:8], time.perf_counter()
    parsed, results = run_search(client, req, request_id)
    log_event({
        "request_id": request_id, "endpoint": "search", "query": req.query,
        "filters": parsed.model_dump(exclude_none=True),
        "result_ids": [r.listing_id for r in results], "latency_ms": ms_since(start),
    })
    return {
        "request_id": request_id,
        "parsed_filters": parsed.model_dump(exclude_none=True),
        "results": [{"id": r.listing_id, "distance": round(r.distance, 4), **r.metadata} for r in results],
    }


@app.post("/recommend")
def recommend(req: SearchRequest, client=Depends(get_client)) -> StreamingResponse:
    """Hybrid search, then stream an LLM-written recommendation grounded in the results."""
    request_id, start = uuid.uuid4().hex[:8], time.perf_counter()
    parsed, results = run_search(client, req, request_id)
    retrieval_ms = ms_since(start)

    def body():
        chunks, first_token_ms = [], None
        try:
            for token in stream_recommendation(client, OPENAI_MODEL, req.query, results):
                if first_token_ms is None:
                    first_token_ms = ms_since(start)
                chunks.append(token)
                yield token
        finally:  # runs even if the client disconnects mid-stream
            log_event({
                "request_id": request_id, "endpoint": "recommend", "query": req.query,
                "filters": parsed.model_dump(exclude_none=True),
                "result_ids": [r.listing_id for r in results],
                "retrieval_ms": retrieval_ms, "first_token_ms": first_token_ms,
                "total_ms": ms_since(start), "answer": "".join(chunks),
            })

    return StreamingResponse(body(), media_type="text/plain", headers={"X-Request-ID": request_id})
