import openai
import pytest
from fastapi.testclient import TestClient

from homematch import api
from homematch.query_parser import ParsedQuery
from homematch.search import SearchResult

META = {"price": 700000, "bedrooms": 4, "bathrooms": 2.0, "size_sqft": 2000,
        "property_type": "house", "neighborhood": "Oak Park"}
RESULTS = [SearchResult("L0001", 0.41, META, "doc text")]


def fake_hybrid(client, chat_model, embedding_model, query, k=5):
    return ParsedQuery(max_price=800000, semantic_query="fireplace"), RESULTS


def fake_stream(client, model, query, results):
    yield "Try "
    yield "L0001."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(api, "stream_recommendation", fake_stream)
    api.app.dependency_overrides[api.get_client] = lambda: object()  # no real OpenAI client
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_filters_and_results(client):
    body = client.post("/search", json={"query": "house under 800k"}).json()
    assert body["parsed_filters"]["max_price"] == 800000
    assert body["results"][0]["id"] == "L0001"


def test_recommend_streams_text(client):
    response = client.post("/recommend", json={"query": "house under 800k"})
    assert response.status_code == 200
    assert response.text == "Try L0001."
    assert "x-request-id" in response.headers


def test_rejects_too_short_query(client):
    assert client.post("/search", json={"query": "a"}).status_code == 422


def test_upstream_error_returns_502(client, monkeypatch):
    def broken(*args, **kwargs):
        raise openai.OpenAIError("service down")

    monkeypatch.setattr(api, "hybrid_search", broken)
    assert client.post("/search", json={"query": "house under 800k"}).status_code == 502
