import os

from qdrant_client import QdrantClient


class QdrantWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

    def create_collection(self, session_id: str):
        # Not strictly needed—.add() auto-creates
        if not self.client.collection_exists(session_id):
            self.client.create_collection(session_id)

    def insert_point(self, point_id: str, collection_name: str, payload: dict):
        self.client.add(
            collection_name=collection_name,
            documents=[payload["content"]],
            metadata=[payload],
            ids=[point_id]
        )


# Singleton instance
qdrant = QdrantWrapper()
