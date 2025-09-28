import redis
import json

# Connect to the running Redis container
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def set_kb(kb_id, data):
    """Stores a knowledge base's data in Redis."""
    redis_client.set(kb_id, json.dumps(data))

def get_kb(kb_id):
    """Retrieves a knowledge base's data from Redis."""
    data = redis_client.get(kb_id)
    if data:
        return json.loads(data)
    return None

