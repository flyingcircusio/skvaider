"""In-process dummy LLM engine for testing.

Runs a lightweight HTTP server (aiohttp) on the model's configured port,
responding to OpenAI-compatible endpoints with deterministic dummy data.
No external binaries required.

Control API (for tests to configure behavior via HTTP):
  POST /__control/set_response  {"path": str, "status": int, "body": dict}
  GET  /__control/last_request  -> {"path": str, "body": dict}
  POST /__control/reset
"""

import asyncio
import json
import time
from typing import Any

import aiohttp
import aiohttp.web
import structlog

from skvaider.inference.config import DummyModelConfig
from skvaider.inference.model import Model

log = structlog.get_logger()


def _chat_response(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Build a non-streaming chat completion response."""
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 16)
    prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages)
    completion_tokens = min(max_tokens, 10)

    return {
        "id": "chatcmpl-dummy-001",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a dummy response for testing purposes.",
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _completion_response(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Build a non-streaming completions response."""
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 16)
    prompt_tokens = len(prompt)
    completion_tokens = min(max_tokens, 10)

    return {
        "id": "cmpl-dummy-001",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": "This is a dummy completion for testing purposes.",
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _embedding_response(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Build an embeddings response with deterministic vectors."""
    model_dims: dict[str, int] = {
        "embeddinggemma": 512,
        "nomic-embed-text-v1.5": 768,
        "all-MiniLM-L6-v2": 384,
    }
    dims = model_dims.get(model_id, 256)

    input_texts: list[str] = body.get("input", [])
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    prompt_tokens = sum(len(t) for t in input_texts)

    import math

    data: list[dict[str, Any]] = []
    for i in range(len(input_texts)):
        vector: list[float] = [
            round(math.sin((i + 1) * (j + 1)) * 0.5 + 0.5, 6)
            for j in range(dims)
        ]
        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": vector,
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


class DummyModel(Model):
    """In-process dummy LLM that serves an HTTP server on its configured port.

    Handles /health, /v1/chat/completions, /v1/completions, /v1/embeddings
    with OpenAI-compatible responses. No external binaries required.

    Control endpoints (for tests, both unit and E2E):
      POST /__control/set_response  {"path": str, "status": int, "body": dict}
      GET  /__control/last_request  -> {"path": str, "body": dict}
      POST /__control/reset
    """

    _engine = "dummy"
    health_check_interval = 1.0
    health_check_timeout = 5.0

    async def _check_health(self) -> dict[str, str]:
        return {}

    async def _monitor_health(self) -> None:
        self.health_status = "healthy"
        self._notify_status_changed()
        while True:
            await asyncio.sleep(1.0)

    async def _check_completion_health(self) -> dict[str, str]:
        return {}

    async def _check_embedding_health(self) -> dict[str, str]:
        return {}

    async def _check_completion_streaming_tool_call_health(
        self,
    ) -> dict[str, str]:
        return {}

    _app: aiohttp.web.Application | None = None
    _runner: aiohttp.web.AppRunner | None = None
    _site: aiohttp.web.TCPSite | None = None

    # Control state — modified via /__control/ endpoints
    _ctrl_override_path: str | None = None
    _ctrl_override: tuple[int, dict[str, Any]] | None = None
    ctrl_last_request_path: str | None = None
    ctrl_last_request_body: dict[str, Any] | None = None

    def __init__(
        self,
        config: DummyModelConfig,
        on_unexpected_exit: Any,
    ):
        super().__init__(config, on_unexpected_exit)
        self._config = config

    @property
    def slug(self) -> str:
        return self.config.id

    async def download(self) -> None:
        pass

    async def start(self) -> None:
        """Start the in-process HTTP server and set the endpoint."""
        log.info("Starting dummy model", model=self.config.id)
        assert self.process is None
        assert self.process_status == "stopped"

        self.process_status = "starting"

        model_id = self.config.id

        async def health_handler(
            _request: aiohttp.web.Request,
        ) -> aiohttp.web.Response:
            return aiohttp.web.json_response({"status": "ok"})

        # ---- Control API handlers ----

        async def control_set_response(
            request: aiohttp.web.Request,
        ) -> aiohttp.web.Response:
            body = await request.json()
            path: str = body["path"]
            status: int = body.get("status", 200)
            resp_body: dict[str, Any] = body.get("body", {})
            self._ctrl_override_path = path
            self._ctrl_override = (status, resp_body)
            return aiohttp.web.json_response({"ok": True})

        async def control_last_request(
            _request: aiohttp.web.Request,
        ) -> aiohttp.web.Response:
            return aiohttp.web.json_response(
                {
                    "path": self.ctrl_last_request_path,
                    "body": self.ctrl_last_request_body,
                }
            )

        async def control_reset(
            _request: aiohttp.web.Request,
        ) -> aiohttp.web.Response:
            self._ctrl_override_path = None
            self._ctrl_override = None
            self.ctrl_last_request_path = None
            self.ctrl_last_request_body = None
            return aiohttp.web.json_response({"ok": True})

        # ---- OpenAI-compatible handlers ----

        def _consume_override(
            path: str, request_body: dict[str, Any]
        ) -> aiohttp.web.Response | None:
            self.ctrl_last_request_path = path
            self.ctrl_last_request_body = request_body
            if (
                self._ctrl_override_path == path
                and self._ctrl_override is not None
            ):
                status, resp_body = self._ctrl_override
                self._ctrl_override_path = None
                self._ctrl_override = None
                return aiohttp.web.json_response(resp_body, status=status)
            return None

        async def chat_handler(
            request: aiohttp.web.Request,
        ) -> aiohttp.web.StreamResponse | aiohttp.web.Response:
            body = await request.json()
            stream = body.get("stream", False)

            if stream:
                return await self._stream_chat(model_id, body, request)

            override = _consume_override("/v1/chat/completions", body)
            if override is not None:
                return override
            return aiohttp.web.json_response(_chat_response(model_id, body))

        async def completions_handler(
            request: aiohttp.web.Request,
        ) -> aiohttp.web.StreamResponse | aiohttp.web.Response:
            body = await request.json()
            stream = body.get("stream", False)

            if stream:
                return await self._stream_completion(model_id, body, request)

            override = _consume_override("/v1/completions", body)
            if override is not None:
                return override
            return aiohttp.web.json_response(
                _completion_response(model_id, body)
            )

        async def embeddings_handler(
            request: aiohttp.web.Request,
        ) -> aiohttp.web.Response:
            body = await request.json()
            override = _consume_override("/v1/embeddings", body)
            if override is not None:
                return override
            return aiohttp.web.json_response(
                _embedding_response(model_id, body)
            )

        app = aiohttp.web.Application()
        app.router.add_get("/health", health_handler)
        app.router.add_post("/v1/chat/completions", chat_handler)
        app.router.add_post("/v1/completions", completions_handler)
        app.router.add_post("/v1/embeddings", embeddings_handler)
        app.router.add_post("/__control/set_response", control_set_response)
        app.router.add_get("/__control/last_request", control_last_request)
        app.router.add_post("/__control/reset", control_reset)

        self._app = app

        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        site = aiohttp.web.TCPSite(runner, self._host, self.config.port)
        self._site = site
        await site.start()

        self.endpoint = f"http://{self._host}:{self.config.port}"
        self.process_status = "running"
        self.health_status = "healthy"
        self._notify_status_changed()

        self._tasks.create(self._monitor_health)

        log.info(
            "Dummy model started",
            model=self.config.id,
            endpoint=self.endpoint,
        )

    async def _stream_chat(
        self, model_id: str, body: dict[str, Any], request: aiohttp.web.Request
    ) -> aiohttp.web.StreamResponse:
        response = _chat_response(model_id, body)
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        response_obj = aiohttp.web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response_obj.prepare(request)

        words = content.split()
        for i, word in enumerate(words):
            delta = {
                "role": "assistant",
                "content": word + (" " if i < len(words) - 1 else ""),
            }
            chunk = {
                "id": "chatcmpl-dummy-001",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            await response_obj.write(f"data: {json.dumps(chunk)}\n\n".encode())

        final: dict[str, Any] = {
            "id": "chatcmpl-dummy-001",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        await response_obj.write(f"data: {json.dumps(final)}\n\n".encode())
        await response_obj.write(b"data: [DONE]\n\n")

        return response_obj

    async def _stream_completion(
        self, model_id: str, body: dict[str, Any], request: aiohttp.web.Request
    ) -> aiohttp.web.StreamResponse:
        response = _completion_response(model_id, body)
        text = response["choices"][0]["text"]
        usage = response.get("usage", {})

        response_obj = aiohttp.web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response_obj.prepare(request)

        words = text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": "cmpl-dummy-001",
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": word + (" " if i < len(words) - 1 else ""),
                        "finish_reason": None,
                    }
                ],
            }
            await response_obj.write(f"data: {json.dumps(chunk)}\n\n".encode())

        final = {
            "id": "cmpl-dummy-001",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        await response_obj.write(f"data: {json.dumps(final)}\n\n".encode())
        await response_obj.write(b"data: [DONE]\n\n")

        return response_obj

    async def terminate(self) -> None:
        log.info("Terminating dummy model", model=self.config.id)
        self.process_status = "stopping"

        self._tasks.terminate()

        if self._site:
            await self._site.stop()
            self._site = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._app = None
        self.endpoint = None
        self.process_status = "stopped"
        self.health_status = ""
        self._notify_status_changed()

        log.info("Dummy model terminated", model=self.config.id)

    async def _wait_for_startup(self) -> None:
        pass

    async def _monitor_process(self) -> None:
        pass
