import os

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


class QdrantWrapper:
    def __init__(self, host, port):
        self.client = QdrantClient(host=host, port=port)



    def create_collection(self, session_id):
        if not self.client.collection_exists(session_id):
            self.client.create_collection(
                collection_name=session_id,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def insert_point(self, point_id, vector, collection_name, payload):
        self.client.upsert(
            collection_name=collection_name,
            points=[PointStruct(vector=vector, id=point_id, payload=payload)]
        )

qdrant = QdrantWrapper(host="localhost", port=os.getenv("QDRANT_PORT"))


