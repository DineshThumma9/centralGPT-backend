import os

import chromadb
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
from qdrant_client import QdrantClient

# qdrant_client =  AsyncQdrantClient(
#              url=os.getenv("QDRANT_URL"),
#            api_key=os.getenv("QDRANT_API_KEY")
#      )
#



load_dotenv()


client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")

)




def get_vector_store(collection_name):
    collection = client.get_or_create_collection(
        name=collection_name


    )
    vector_store = ChromaVectorStore(

        chroma_collection=collection

    )

    return vector_store


