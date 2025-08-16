# src/db/redis_client.py

import os
import redis.asyncio as redis
from llama_index.core import StorageContext
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.redis import RedisVectorStore


REDIS_URL = os.getenv("REDIS_URL")




redis_client = redis.from_url(
    url=REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30
)


chat_store = RedisChatStore(
    redis_url=REDIS_URL,
    aredis_client=redis_client,
    ttl=3600
)




kvstore = RedisKVStore(
    redis_uri=REDIS_URL,
    async_redis_client=redis_client
)



def get_doc_store():
    return RedisDocumentStore(redis_kvstore=kvstore)


def get_index_store(namespace):

      return  RedisIndexStore(
        redis_kvstore=RedisKVStore(
            redis_uri=REDIS_URL,
            async_redis_client=redis_client
        ),
        namespace=namespace
    )










