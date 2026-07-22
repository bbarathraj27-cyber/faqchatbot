import sys, os
sys.path.insert(0, os.getcwd())
from app.config import settings
import redis

if settings.REDIS_URL:
    client = redis.from_url(settings.REDIS_URL, decode_responses=True)
else:
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        ssl=True,
        decode_responses=True,
    )

client.ping()
print("Connected to Redis.")

deleted = 0
for key in client.scan_iter(match="faq:*"):
    client.delete(key)
    deleted += 1

print(f"Deleted {deleted} cached FAQ answer(s).")
