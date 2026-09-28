import redis.asyncio as redis
from core.config import settings

# Single shared async Redis client
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# TTL constants
INGESTION_STATUS_TTL = 60 * 60 * 24      # 24 hours
CHAT_CACHE_TTL = 60 * 60 * 2             # 2 hours (cache repeated queries)


async def cache_ingestion_status(doc_id: str, status: dict) -> None:
    import json
    await redis_client.setex(
        f"ingest:{doc_id}",
        INGESTION_STATUS_TTL,
        json.dumps(status),
    )


async def get_ingestion_status(doc_id: str) -> dict | None:
    import json
    raw = await redis_client.get(f"ingest:{doc_id}")
    return json.loads(raw) if raw else None


async def cache_chat_response(query_hash: str, response: str) -> None:
    await redis_client.setex(f"chat:{query_hash}", CHAT_CACHE_TTL, response)


async def get_cached_response(query_hash: str) -> str | None:
    return await redis_client.get(f"chat:{query_hash}")
