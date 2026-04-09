"""Verify that the gateway forwards X-Skvaider-Request-ID to the inference backend."""

from typing import Callable
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import Request

from skvaider.proxy.backends import DummyBackend, SkvaiderBackend
from skvaider.routers.openai import OpenAIProxy


async def test_request_id_forwarded_to_dummy_backend(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory: Callable[..., Request],
) -> None:
    req = mock_request_factory(stream=False)
    await proxy.proxy(req, "/v1/chat/completions")

    assert dummy_backend.last_request_id == req.state.request_id


async def test_request_id_forwarded_to_dummy_backend_streaming(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory: Callable[..., Request],
) -> None:
    req = mock_request_factory(stream=True)
    response = await proxy.proxy(req, "/v1/chat/completions")
    # consume the stream so the generator runs
    async for _ in response.body_iterator:  # type: ignore[union-attr]
        pass

    assert dummy_backend.last_request_id == req.state.request_id


async def test_skvaider_backend_sends_request_id_header() -> None:
    backend = SkvaiderBackend("http://inference:8001")
    backend.models = {}

    captured: dict[str, str] = {}

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(dict(kwargs.get("headers", {})))  # type: ignore[arg-type]
        return httpx.Response(200, json={"id": "cmpl-1", "choices": []})

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=fake_post,
    ):
        await backend.post(
            "/v1/chat/completions",
            {"model": "gemma"},
            request_id="abc12345",
        )

    assert captured.get("X-Skvaider-Request-ID") == "abc12345"


async def test_skvaider_backend_post_stream_sends_request_id_header() -> None:
    backend = SkvaiderBackend("http://inference:8001")

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)

    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = transport
        original_init(self, **kwargs)  # type: ignore[misc]

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        async for _ in backend.post_stream(
            "/v1/chat/completions",
            {"model": "gemma"},
            request_id="abc12345",
        ):
            pass

    assert len(captured) == 1
    assert captured[0].headers.get("x-skvaider-request-id") == "abc12345"
