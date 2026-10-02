# HomeMatch

Natural-language real-estate search. A buyer describes what they want
("4 bedroom house with a fireplace under $800k"), and HomeMatch finds matching listings
with **hybrid retrieval** and streams back a grounded recommendation.

## How it works

```
user query
   │
   ▼  LLM query parser ──► exact filters (price, bedrooms, type, size)
   │                    └► semantic query ("fireplace")
   ▼
Chroma: metadata filter → vector similarity ranking (OpenAI embeddings, cosine)
   │
   ▼
LLM writes a recommendation using ONLY the retrieved listings (streamed via FastAPI)
```

Plain vector search understands meaning but can't compare numbers, so it happily returns a
$7.5M house for an "under $800k" query. Hybrid search applies hard constraints exactly and
uses embeddings only for soft preferences.

## Results

Evaluated on 20 hand-labeled queries (k=5). Ground truth is computed from structured listing
data, and expected constraints are written by hand so the LLM never grades itself.

| Metric | Vector search | Hybrid search |
|---|---|---|
| Constraint precision@5 | 0.81 | **1.00** |
| Full precision@5 (constraints + features) | 0.59 | **0.73** |
| Recall@5 | 0.70 | **0.89** |

Query parser extracted the exact expected filters on 20/20 queries.
Details: `results/eval_report.md`.

**Limitations:** listings are LLM-generated (95 listings), and the eval set is small.
Soft features (e.g. "fireplace") are ranked by similarity, not guaranteed. Next step:
extract must-have features and filter on them too.

## Project structure

```
src/homematch/
  schemas.py       listing data model (Pydantic)
  generate.py      LLM listing generation with validation + retries
  query_parser.py  natural language → filters + semantic query
  filters.py       filters → Chroma `where` clause
  index.py         embeddings + Chroma index
  search.py        vector_search (baseline) and hybrid_search
  evaluate.py      metrics against ground truth
  recommend.py     grounded, streamed recommendation
  api.py           FastAPI service with JSON request logging
scripts/           generate data, build index, demo, run evaluation
tests/             unit + API tests (no API calls; OpenAI is faked)
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # add your OpenAI key
export PYTHONPATH=src
python scripts/generate_listings.py --per-scenario 10  # optional: data/listings.jsonl is included
python scripts/build_index.py
python scripts/run_eval.py
uvicorn homematch.api:app --reload
```

Open http://localhost:8000/docs for interactive API docs, or:

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "condo under $500k near transit"}'

curl -N -X POST localhost:8000/recommend -H "Content-Type: application/json" \
     -d '{"query": "4 bedroom house with a fireplace under $800k"}'
```

Each request is logged as one JSON line in `logs/requests.jsonl` (filters, retrieved ids,
retrieval latency, time to first token, total latency, answer).

## Run with Docker

```bash
docker build -t homematch .
docker run --env-file .env -p 8000:8000 homematch
```

## Tests

```bash
pytest
```
