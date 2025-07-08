from redis import Redis


import redis
import os

# src/db/redis_client.py
import redis
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=False)

