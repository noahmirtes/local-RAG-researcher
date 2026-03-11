# chroma.py

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "local_research_nomic_embed"
DB_PATH = "/Users/noah/REPOS/local-RAG-researcher/chroma_store"

def get_collection():
    client = chromadb.Client(
        Settings(
            persist_directory=DB_PATH,
            is_persistent=True
        )
    )

    return client.get_or_create_collection(COLLECTION_NAME)