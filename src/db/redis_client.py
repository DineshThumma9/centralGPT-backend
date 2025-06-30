from redis import Redis


import redis
import os

redis = redis.Redis.from_url(os.getenv("REDIS_URL"))
