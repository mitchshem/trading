"""
Server-Sent Events (SSE) broadcaster.

Async fan-out event bus: backend components publish events,
connected frontend clients receive them via GET /events.

Thread-safe: publish() can be called from sync code (live_trading_loop,
monitoring), events are delivered to async subscriber queues.

Usage:
    from sse_broadcaster import broadcaster

    # Publish from anywhere (sync or async):
    broadcaster.publish("equity_update", {"equity": 100500.0, "daily_pnl": 500.0})

    # Subscribe in an async endpoint:
    queue = broadcaster.subscribe()
    try:
        event = await asyncio.wait_for(queue.get(), timeout=15.0)
    finally:
        broadcaster.unsubscribe(queue)
"""

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SSEBroadcaster:
    """
    Async fan-out event bus for Server-Sent Events.

    Subscribers get an asyncio.Queue. Published events are
    delivered to all subscriber queues.
    """

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self) -> asyncio.Queue:
        """Create and register a new subscriber queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(queue)
        logger.debug(f"SSE subscriber added (total: {len(self._subscribers)})")
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass
        logger.debug(f"SSE subscriber removed (total: {len(self._subscribers)})")

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, event_type: str, payload: dict):
        """
        Publish an event to all subscribers.

        Thread-safe: can be called from sync code. Uses
        call_soon_threadsafe to schedule queue puts on the
        event loop.
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with self._lock:
            subscribers = list(self._subscribers)

        if not subscribers:
            return

        # Try to get the running event loop
        loop = self._get_loop()

        for queue in subscribers:
            try:
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(self._put_nowait, queue, event)
                else:
                    # Direct put (already in the event loop thread)
                    self._put_nowait(queue, event)
            except Exception as e:
                logger.debug(f"Failed to publish to subscriber: {e}")

    async def publish_async(self, event_type: str, payload: dict):
        """
        Async version of publish. Use when already in an async context.
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with self._lock:
            subscribers = list(self._subscribers)

        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event to make room
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def _put_nowait(self, queue: asyncio.Queue, event: dict):
        """Put an event into a queue, dropping oldest if full."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the running event loop (cached)."""
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        try:
            self._loop = asyncio.get_running_loop()
            return self._loop
        except RuntimeError:
            return None

    def format_sse(self, event: dict) -> str:
        """Format an event dict as an SSE text message."""
        event_type = event.get("type", "message")
        data = json.dumps(event)
        return f"event: {event_type}\ndata: {data}\n\n"

    def heartbeat_event(self) -> dict:
        """Create a heartbeat event."""
        return {
            "type": "heartbeat",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {},
        }


# Module-level singleton
broadcaster = SSEBroadcaster()
