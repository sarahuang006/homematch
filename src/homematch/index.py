"""Build and load the Chroma vector index."""
import json
from pathlib import Path

import chromadb
from openai import OpenAI

from homematch.config import CHROMA_PATH, COLLECTION_NAME
from homematch.schemas import Listing


def load_listings(path: str) -> list[Listing]:
    with Path(path).open() as f:
        return [Listing.model_validate_json(line) for line in f if line.strip()]


def embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def get_collection(rebuild: bool = False):
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    if rebuild:
        try:
            db.delete_collection(COLLECTION_NAME)
        except Exception:  # collection didn't exist yet
            pass
    return db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_index(client: OpenAI, embedding_model: str, listings: list[Listing]) -> int:
    collection = get_collection(rebuild=True)
    documents = [listing.to_document() for listing in listings]
    collection.add(
        ids=[listing.id for listing in listings],
        documents=documents,
        embeddings=embed(client, embedding_model, documents),
        # metadata = the structured fields we can filter on exactly
        metadatas=[
            {
                "price": l.price,
                "bedrooms": l.bedrooms,
                "bathrooms": l.bathrooms,
                "size_sqft": l.size_sqft,
                "property_type": l.property_type,
                "neighborhood": l.neighborhood,
            }
            for l in listings
        ],
    )
    return collection.count()
