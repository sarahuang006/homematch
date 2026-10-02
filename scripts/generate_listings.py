"""Usage: python scripts/generate_listings.py --per-scenario 10
Writes data/listings.jsonl (one JSON listing per line)."""
import argparse
from pathlib import Path

from openai import OpenAI

from homematch.config import OPENAI_MODEL, require_api_key
from homematch.generate import SCENARIOS, generate_batch

OUTPUT = Path("data/listings.jsonl")
BATCH_SIZE = 5  # small requests = shorter JSON = fewer broken responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-scenario", type=int, default=10)
    args = parser.parse_args()

    client = OpenAI(api_key=require_api_key())
    all_listings = []
    for scenario in SCENARIOS:
        count = 0
        while count < args.per_scenario:
            n = min(BATCH_SIZE, args.per_scenario - count)
            batch = generate_batch(client, OPENAI_MODEL, scenario, n, len(all_listings))
            if not batch:  # batch failed even after retries; move on instead of looping forever
                break
            all_listings.extend(batch)
            count += len(batch)
        print(f"{count:>3} listings  <- {scenario}")

    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w") as f:
        for listing in all_listings:
            f.write(listing.model_dump_json() + "\n")
    print(f"Saved {len(all_listings)} listings to {OUTPUT}")


if __name__ == "__main__":
    main()
