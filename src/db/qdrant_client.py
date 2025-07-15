import os

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

qdrant_client =  QdrantClient(
            url=os.getenv("QDRANT_URL"),
          api_key=os.getenv("QDRANT_API_KEY")
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="llamaIndex"
)
