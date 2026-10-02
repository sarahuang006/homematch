"""Load settings from environment variables (and a local .env file)."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

LISTINGS_PATH = "data/listings.jsonl"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "listings"


def require_api_key() -> str:
    """Return the API key, or fail with a clear message if it's missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OPENAI_API_KEY
