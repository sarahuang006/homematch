"""Compare vector search vs hybrid search on the eval set.
Usage: PYTHONPATH=src python scripts/run_eval.py
Writes results/eval_report.md and results/eval_results.json"""
import json
from pathlib import Path
from statistics import mean

from openai import OpenAI

from homematch.config import LISTINGS_PATH, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, require_api_key
from homematch.evaluate import EvalQuery, parser_is_correct, score_query
from homematch.index import load_listings
from homematch.search import hybrid_search, vector_search

K = 5
QUERIES_PATH = Path("data/eval_queries.json")
RESULTS_DIR = Path("results")


def avg(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r[key] is not None]
    return mean(values) if values else float("nan")


def main() -> None:
    client = OpenAI(api_key=require_api_key())
    listings = {l.id: l for l in load_listings(LISTINGS_PATH)}
    queries = [EvalQuery.model_validate(q) for q in json.loads(QUERIES_PATH.read_text())]

    rows = []
    for q in queries:
        vec_ids = [r.listing_id for r in vector_search(client, OPENAI_EMBEDDING_MODEL, q.query, K)]
        parsed, hyb = hybrid_search(client, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, q.query, K)
        row = {
            "query": q.query,
            "vector": score_query(vec_ids, listings, q.constraints, K),
            "hybrid": score_query([r.listing_id for r in hyb], listings, q.constraints, K),
            "parser_correct": parser_is_correct(parsed, q.constraints),
            "parsed_filters": parsed.model_dump(exclude_none=True),
        }
        rows.append(row)
        flag = "" if row["parser_correct"] else "   <- parser mismatch"
        print(f"{row['vector']['n_relevant_total']:>3} relevant | {q.query}{flag}")

    summary = {
        mode: {m: avg([r[mode] for r in rows], m) for m in ["constraint_precision", "full_precision", "recall"]}
        for mode in ["vector", "hybrid"]
    }
    parser_acc = mean(r["parser_correct"] for r in rows)

    lines = [
        f"# Evaluation: vector vs hybrid search ({len(rows)} queries, k={K})\n",
        "| Metric | Vector | Hybrid |",
        "|---|---|---|",
    ]
    for m in ["constraint_precision", "full_precision", "recall"]:
        lines.append(f"| {m} | {summary['vector'][m]:.2f} | {summary['hybrid'][m]:.2f} |")
    lines.append(f"\nQuery parser accuracy (exact filter match): {parser_acc:.0%}\n")
    lines += ["## Per query (full_precision)\n", "| Query | Relevant | Vector | Hybrid | Parser OK |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['query']} | {r['vector']['n_relevant_total']} | {r['vector']['full_precision']:.1f} "
            f"| {r['hybrid']['full_precision']:.1f} | {'yes' if r['parser_correct'] else 'no'} |"
        )
    report = "\n".join(lines)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval_report.md").write_text(report)
    (RESULTS_DIR / "eval_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n" + "\n".join(lines[:7]))
    print(f"Full report: {RESULTS_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main()
