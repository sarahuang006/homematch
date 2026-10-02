"""Usage: PYTHONPATH=src python scripts/build_index.py [--if-missing]"""
import argparse

from openai import OpenAI

from homematch.config import LISTINGS_PATH, OPENAI_EMBEDDING_MODEL, require_api_key
from homematch.index import build_index, get_collection, load_listings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true", help="skip if the index already has data")
    args = parser.parse_args()

    if args.if_missing and get_collection().count() > 0:
        print("Index already exists, skipping build")
        return

    client = OpenAI(api_key=require_api_key())
    count = build_index(client, OPENAI_EMBEDDING_MODEL, load_listings(LISTINGS_PATH))
    print(f"Indexed {count} listings into Chroma")


if __name__ == "__main__":
    main()
