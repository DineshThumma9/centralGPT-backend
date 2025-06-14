import os
from dotenv import load_dotenv
from fastapi import APIRouter, Body
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    load_index_from_storage, Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# === CONFIG ===
qdrant_dir = "../src/router/qdrant_db"
llama_metadata_dir = "../src/router/storage"
collection_name = "rag"
data_dir = "/src/data"

# === ENV & MODELS ===
load_dotenv()

router = APIRouter()

@router.post("/init-rag")
def init_rag(body:Body(...)):
    Settings.llm = Groq(model="qwen-qwq-32b")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")


# === Qdrant Setup ===
qdrant_client = QdrantClient(path=qdrant_dir)
qdrant_vec = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

# === Index Load/Create ===
docstore_path = os.path.join(llama_metadata_dir, "docstore.json")
if   os.path.exists(docstore_path):
    print("🔁 Loading from saved index")
    storage = StorageContext.from_defaults(persist_dir=llama_metadata_dir, vector_store=qdrant_vec)
    index = load_index_from_storage(storage)
else:
    print("🆕 Creating new index")
    documents = SimpleDirectoryReader(data_dir).load_data()
    storage = StorageContext.from_defaults(vector_store=qdrant_vec)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage, show_progress=True)
    index.storage_context.persist()

# === Chat Engine with Memory ===
query_engine = index.as_query_engine()
from llama_index.storage.chat_store.redis import RedisChatStore
chat_store = RedisChatStore(
    redis_url="redis://localhost:6379",
    ttl=3600 * 24  # 24-hour expiry (optional)
)
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user-session-XYZ",  # use per-user session ID
)

from llama_index.core.chat_engine import CondensePlusContextChatEngine,ContextChatEngine

from llama_index.core.chat_engine import CondensePlusContextChatEngine

# === Chat Engine Setup ===
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",  # use "context" for simpler cases
    memory=memory,
    llm=Settings.llm,
    system_prompt=(
        "You are RAG‑GPT, an honest and reliable assistant. "
        "If you don’t know something, say 'I don’t know'. "
        "Never hallucinate or fabricate information. "
        "Always cite sources or indicate when evidence is insufficient."
    ),
    verbose=True,
)

# === Chat Loop ===
while True:
    try:
        query = input("Query Documents: ").strip()
        if not query:
            continue
        if query == "!exit":
            print("👋 Goodbye!")
            break
        elif query == "!history":
            for msg in memory.get_all():
                print(f"{msg.role.upper()}: {msg.content}")
            continue
        elif query == "!clear":
            memory.reset()
            print("🧠 Memory cleared.")
            continue

        response = chat_engine.chat(query)
        print(f"\n🧠 Response:\n{response}\n")
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        break
