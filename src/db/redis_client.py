import os

# src/db/redis_client.py
import redis
from llama_index.storage.chat_store.redis import RedisChatStore

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

chat_store = RedisChatStore(
    redis_client=redis_client,
    ttl=3600
)
