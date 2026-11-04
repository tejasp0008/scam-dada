"""Redis pub/sub fan-out from a call to its dashboard subscribers.

Events are published as serialised `DomainEvent` JSON on one channel per call. The
dashboard WebSocket subscribes; nothing else does. Live call state is kept in Redis too, so
a dashboard that connects mid-call can render immediately instead of waiting for the next
event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import TypeAdapter
from redis.asyncio import Redis

from app.core.events import DomainEvent
from app.infra.settings import Settings

_EVENT_ADAPTER: TypeAdapter[DomainEvent] = TypeAdapter(DomainEvent)

#: How long a finished call's cached state and event log survive, for a late dashboard.
CALL_STATE_TTL_SECONDS = 3600


def create_redis(settings: Settings) -> Redis:
    """Build a Redis client. Decoding is off: event payloads are handled as bytes."""
    return Redis.from_url(str(settings.redis_url), decode_responses=False)


def channel_for(call_id: str) -> str:
    """Pub/sub channel carrying one call's events."""
    return f"call:{call_id}:events"


def log_key_for(call_id: str) -> str:
    """List key holding one call's events in order, for replay on reconnect."""
    return f"call:{call_id}:log"


class EventBus:
    """Publishes and replays call events."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def publish(self, event: DomainEvent) -> None:
        """Append to the replay log and notify live subscribers."""
        raw = _EVENT_ADAPTER.dump_json(event)
        key = log_key_for(event.call_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, raw)
        pipe.expire(key, CALL_STATE_TTL_SECONDS)
        pipe.publish(channel_for(event.call_id), raw)
        await pipe.execute()

    async def history(self, call_id: str, after_seq: int = -1) -> list[DomainEvent]:
        """Return logged events with `seq` greater than `after_seq`."""
        raw_events = await self._redis.lrange(log_key_for(call_id), 0, -1)
        events = [_EVENT_ADAPTER.validate_json(raw) for raw in raw_events]
        return [event for event in events if event.seq > after_seq]

    async def subscribe(self, call_id: str) -> AsyncIterator[DomainEvent]:
        """Yield events published for `call_id` until the caller stops consuming."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel_for(call_id))
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                yield _EVENT_ADAPTER.validate_json(message["data"])
        finally:
            await pubsub.unsubscribe(channel_for(call_id))
            # redis-py ships no annotation for PubSub.aclose.
            await pubsub.aclose()  # type: ignore[no-untyped-call]
