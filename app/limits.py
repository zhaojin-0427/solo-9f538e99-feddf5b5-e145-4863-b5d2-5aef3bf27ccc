"""ASGI middleware that caps request body size for uploads."""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

Scope = dict
Message = dict
Send = Callable[[Message], Awaitable[None]]

# Seconds spent discarding an oversized body before answering with 413 anyway.
DRAIN_SECONDS = 2.0


class BodyLimitMiddleware:
    """Reject request bodies larger than the configured limit before parsing.

    Starlette spools multipart files to disk, so a chunked upload without a
    ``Content-Length`` header would otherwise be written out unbounded.  The
    body is therefore drained (up to the limit) here and replayed to the
    application; anything larger gets a ``413`` without reaching the multipart
    parser or the disk.

    The effective limit comes from ``app.state.body_limit_bytes`` once the
    application has started, otherwise from ``default_max_bytes``.
    """

    def __init__(
        self,
        app,
        *,
        default_max_bytes: int = 8 * 1024 * 1024,
        methods: tuple[str, ...] = ("POST", "PUT", "PATCH"),
    ):
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.methods = methods

    def _limit(self, scope: Scope) -> int:
        app = scope.get("app")
        limit = getattr(getattr(app, "state", None), "body_limit_bytes", None)
        return int(limit) if limit else self.default_max_bytes

    async def __call__(self, scope: Scope, receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in self.methods:
            await self.app(scope, receive, send)
            return

        limit = self._limit(scope)
        # Draining an oversized body is bounded so a hostile client cannot make
        # the server read forever just to deliver the error.
        drain_cap = 8 * limit
        headers = {key.decode("latin-1").lower(): value for key, value in scope["headers"]}
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    # Read (and discard) the rest so the client can actually
                    # read the 413 instead of seeing a connection reset.
                    await self._drain(receive, drain_cap)
                    await self._reject(send, limit)
                    return
            except ValueError:
                pass

        messages: list[Message] = []
        received = 0
        complete = False
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                complete = True
                break
            received += len(message.get("body", b""))
            if received > limit:
                break
            if not message.get("more_body", False):
                complete = True
                break

        if received > limit:
            if not complete:
                # Read (and discard) the rest so the client can actually see
                # the 413 instead of a connection reset.  Bounded both in size
                # and time so a hostile or stalled client cannot pin a worker.
                await self._drain(receive, drain_cap)
            await self._reject(send, limit)
            return

        index = 0

        async def replay() -> Message:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _drain(receive, budget: int) -> None:
        """Discard the rest of an oversized body, bounded in size and time.

        If the client stalls or disconnects, give up quickly: the 413 is still
        sent and ``Connection: close`` lets the peer notice.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + DRAIN_SECONDS
        while budget >= 0:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=min(remaining, 0.5))
            except Exception:
                return
            if message["type"] == "http.disconnect":
                return
            budget -= len(message.get("body", b""))
            if not message.get("more_body", False):
                return

    async def _reject(self, send: Send, limit: int) -> None:
        payload = json.dumps(
            {
                "detail": {
                    "code": "request_too_large",
                    "message": f"request body must be at most {limit} bytes",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload, "more_body": False})
