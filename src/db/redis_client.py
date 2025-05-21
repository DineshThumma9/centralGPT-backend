from redis import Redis

redis = Redis(host="localhost", port=6379, decode_responses=True)
