"""Open-AI compatible API based on Ollama.

This uses Ollama-internal APIs for better load-balancing but exposes a pure OpenAI-compatible API.

"""

import asyncio
import contextlib
import json
import time
from typing import Any, Dict, Generic, Optional, TypeVar

import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()

T = TypeVar("T")

log = structlog.stdlib.get_logger()


def log_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        log.exception("Exception raised by task = %r", task)


def create_logged_task(aw):
    t = asyncio.create_task(aw)
    t.add_done_callback(log_task_exception)
    return t


class AIModel(BaseModel):
    """Model object per backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str

    backend: "Backend" = Field(exclude=True)
    last_used: float = Field(default=0, exclude=True)
    in_progress: int = Field(default=0, exclude=True)
    limit: int = Field(default=5, exclude=True)
    idle: asyncio.Event = Field(default=True, exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: int = Field(default=0, exclude=True)
    log: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.idle = asyncio.Event()
        self.idle.set()

        self.log = log.bind(model=self.id, backend=self.backend.url)

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            yield
        finally:
            self.in_progress -= 1
            self.log.debug("done", in_progress=self.in_progress)
            if not self.in_progress:
                self.log.debug("idling")
                self.idle.set()

    async def wait(self):
        await self.idle.wait()
        return self


class ModelConfig:
    """Configuration for model-specific options"""

    # map model names (including or excluding tags) to dicts containing model-specific settings
    config: Dict[str, Dict[str, Any]]

    def __init__(self, config):
        self.config = config

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get custom options for a specific model"""
        for candidate in [model_id, model_id.split(":")[0], "__default__"]:
            if candidate in self.config:
                return self.config[candidate]
        return {}


class Backend:
    """Connection to a single backend."""

    url: str

    health_interval: int = 15
    healthy: bool = None
    unhealthy_reason: str = ""
    models: dict[str, AIModel]
    model_config: ModelConfig

    def __init__(self, url, model_config):
        self.url = url
        self.models = {}
        self.model_config = model_config
        self.log = structlog.stdlib.get_logger().bind(backend=self.url)

    @property
    def memory_usage(self):
        return sum([v.memory_usage for v in self.models.values()])

    async def post(self, path: str, data: dict):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(self.url + path, json=data, timeout=120)
            return r.json()

    async def update_model_load_status(self):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(self.url + "/api/ps")
            model_status = {}
            for entry in r.json()["models"]:
                model_status[entry["name"]] = entry

        for model_id, model_obj in self.models.items():
            if model_data := model_status.get(model_obj.id):
                model_obj.is_loaded = True
                model_obj.memory_usage = model_data["size_vram"]
            else:
                model_obj.is_loaded = False
                model_obj.memory_usage = 0

    async def monitor_health_and_update_models(self, pool):
        self.log.debug("starting monitor")
        while True:
            try:
                self.log.debug("probing backend")
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(self.url + "/v1/models")
                    known_models = r.json()["data"] or ()
                self.log.debug("updating backends")
                current_models = self.models
                updated_models = {}
                for model in known_models:
                    if model["id"] not in known_models:
                        model_obj = AIModel(
                            id=model["id"],
                            created=model["created"],
                            owned_by=model["owned_by"],
                            backend=self,
                        )
                    else:
                        model_obj = current_models.get(
                            model["id"],
                        )
                        model_obj.created = model["created"]
                        model_obj.owned_by = model["owned_by"]

                    updated_models[model_obj.id] = model_obj

                self.models = updated_models

                await self.update_model_load_status()

                pool.update_model_maps()

            except Exception as e:
                if self.healthy or self.healthy is None:
                    self.log.error("marking as unhealthy", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)
                # Reset our model knowledge, drop statistics
                self.models = {}
            else:
                if not self.healthy:
                    self.log.info("marking as healthy")
                self.healthy = True
                self.unhealthy_reason = ""

            await asyncio.sleep(self.health_interval)

    # XXX fail over to next ?
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_client.py", line 1730, in _send_single_request
    #     response = await transport.handle_async_request(request)
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 393, in handle_async_request
    #     with map_httpcore_exceptions():
    #   File "/Users/ctheune/.nix-profile/lib/python3.11/contextlib.py", line 158, in __exit__
    #     self.gen.throw(typ, value, traceback)
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    #     raise mapped_exc(message) from exc
    # httpx.ConnectError: All connection attempts failed


class ProxyRequest:
    backend_available: asyncio.Event
    model: AIModel = None

    def __init__(self):
        self.backend_available = asyncio.Event()


class Pool:
    backends: list["Backend"]
    health_check_tasks: list[asyncio.Task]
    queues: dict[str, asyncio.Queue]  # one queue per model

    def __init__(self):
        self.backends = []
        self.health_check_tasks = []
        self.queues = {}
        self.models = {}
        self.queue_tasks = {}

    def add_backend(self, backend):
        self.backends.append(backend)
        self.health_check_tasks.append(
            create_logged_task(backend.monitor_health_and_update_models(self))
        )

    def update_model_maps(self):
        # XXX the same model must not be owned by different organizations!
        # This requires a bit more thought how to handle consistency if
        # backends answer with conflicting/differing model data.
        self.models.clear()
        for backend in self.backends:
            self.models.update(backend.models)

        # Add new models
        for model_id in self.models:
            if model_id in self.queues:
                continue
            self.queues[model_id] = asyncio.Queue()
            self.queue_tasks[model_id] = create_logged_task(
                self.assign_backends(model_id)
            )

        # Remove outdated model queues and tasks
        for model_id, task in self.queue_tasks.items():
            if model_id in self.models:
                continue
            task.cancel()
            del self.queue_tasks[model_id]

        for model_id in self.queues:
            if model_id in self.models:
                continue
            del self.queues[model_id]

    async def assign_backends(self, model_id: str):
        """Continuously assign requests to backends.

        Perform batching and model distribution and warmup.

        """
        while True:
            log.debug("waiting for request", model=model_id)
            queue = self.queues[model_id]
            request_batch = [await queue.get()]
            log.debug("got request", model=model_id)

            # Now, are there any backends with the model loaded and are they available?
            while not (
                model_backends := [
                    b for b in self.backends if model_id in b.models
                ]
            ):
                log.warning("no backends with model available", model=model_id)
                await asyncio.sleep(1)

            loaded_backends = [
                b for b in model_backends if b.models[model_id].is_loaded
            ]
            idle_backends = [
                b for b in loaded_backends if b.models[model_id].idle.is_set()
            ]
            not_loaded_backends = [
                b for b in model_backends if not b.models[model_id].is_loaded
            ]

            if (
                not idle_backends
                and len(loaded_backends) < 2
                and not_loaded_backends
            ):  # At most 2 instances per model
                # Load the model on a host with as little used memory as possible
                # if we have spare hosts.
                not_loaded_backends.sort(key=lambda b: b.memory_usage)
                new_backend = not_loaded_backends[0]
                idle_backends.insert(0, new_backend)

            if not idle_backends:
                # Need to wait for an idle backend
                log.debug("waiting for idle backends", model=model_id)
                backends_to_wait_for = [
                    create_logged_task(b.models[model_id].wait())
                    for b in model_backends
                ]
                idle_backends, _ = await asyncio.wait(
                    backends_to_wait_for,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # the above is a set, we want a list
                idle_backends = [b for b in idle_backends]
            backend = idle_backends[0]
            model = backend.models[model_id]
            log.debug("got idle backend", backend=backend.url, model=model_id)

            log.debug("gathering more batchable requests", model=model_id)
            # Prime the model
            # Wait up to 0.1s for up to N requests
            more_request_tasks = await asyncio.gather(
                *[
                    asyncio.wait_for(queue.get(), 0.001)
                    for _ in range(model.limit - 1)
                ],
                return_exceptions=True,
            )
            request_batch.extend(
                [t for t in more_request_tasks if not isinstance(t, Exception)]
            )
            for request in request_batch:
                log.debug(
                    "assigning request to backend",
                    model=model_id,
                    backend=backend.url,
                )
                model.in_progress += 1
                request.model = model
                request.backend_available.set()
            model.idle.clear()

    def close(self):
        for task in self.health_check_tasks:
            task.cancel()
        for task in self.queue_tasks.values():
            task.cancel()

    # def choose_backend(self, model_id: str):
    #     """Return a list of all healthy connections sorted by least number of
    #     current connections.
    #     """
    #     healthy = filter(lambda b: b.healthy, self.backends)
    #     with_model = filter(lambda b: model_ in b.models, healthy)
    #     available_models = sorted(with_model, key=lambda x: x.connections)
    #     if not available_models:
    #         raise HTTPException(
    #             400,
    #             f"The model: `{model_id}` does not exist",
    #         )

    #     ranked_models = sorted(
    #         with_model,
    #         key=lambda x: (
    #             x.models[model_id].is_loaded,
    #             x.models[model_id].open_slots,
    #         ),
    #     )

    #     # backend = self.choose_backend(model_id)
    #     # model = backend.models[model_id]
    #     # model.last_used = time.time()
    #     # try:
    #     #     yield backend
    #     # finally:
    #     #     model.in_progress
    #     #     model.connections -= 1

    #     # - consider actual ram usage on backends before asking a server
    #     #   to load a fresh model

    #     # Better decision for later
    #     # - perform batching on our side and then unblock a number of requests at the same time
    #     # - do not select a backend until it has capacity for us

    #     return ranked[_models0]

    @contextlib.asynccontextmanager
    async def use(self, model_id: str):
        request = ProxyRequest()
        assert model_id in self.queues
        log.debug("queuing request", model=model_id)
        queue = self.queues[model_id]
        await queue.put(request)
        log.debug("waiting for backend to become available", model=model_id)
        await request.backend_available.wait()
        log.debug(
            "got backend", backend=request.model.backend.url, model=model_id
        )
        async with request.model.use():
            yield request.model.backend


class OpenAIProxy:
    """Proxy that uses translators to convert between OpenAI and Ollama formats."""

    def __init__(
        self,
        services: svcs.fastapi.DepContainer,
        translator: "Translator",
        ollama_endpoint: str,
    ):
        self.services = services
        self.pool = services.get(Pool)
        self.translator = translator
        self.ollama_endpoint = ollama_endpoint

    async def proxy(self, request: Request, allow_stream=True):
        """Proxy a request to the backend using the translator."""
        request_data = await request.json()
        request_data["store"] = False
        request.state.model = request_data["model"]
        request.state.stream = allow_stream and request_data.get(
            "stream", False
        )

        if request.state.model not in self.pool.queues:
            raise HTTPException(
                400,
                f"The model `{request.state.model}` is currently not available.",
            )

        async with self.pool.use(request.state.model) as backend:
            # Translate OpenAI request to Ollama format
            ollama_request = self.translator.translate_request(request_data)

            # Add model configuration options
            model_id = request_data["model"]
            model_options = backend.model_config.get(model_id)
            if model_options:
                if "options" in ollama_request:
                    # Merge with existing options, model config takes priority
                    ollama_request["options"].update(model_options)
                else:
                    ollama_request["options"] = model_options

            if request.state.stream:
                return await self._proxy_stream(
                    backend, ollama_request, request_data
                )
            else:
                return await self._proxy_non_stream(
                    backend, ollama_request, request_data
                )

    async def _proxy_non_stream(
        self, backend, ollama_request, original_request
    ):
        """Handle non-streaming requests."""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                r = await client.post(
                    backend.url + self.ollama_endpoint,
                    json=ollama_request,
                    timeout=120,
                )
                r.raise_for_status()
                ollama_response = r.json()

                # Translate Ollama response back to OpenAI format
                return self.translator.translate_response(
                    ollama_response, original_request
                )
            except httpx.HTTPStatusError as e:
                # Convert backend errors to OpenAI-compatible error responses
                error_message = "Backend error"
                error_type = "api_error"

                try:
                    # Extract error message from JSON response
                    error_data = e.response.json()
                    error_message = error_data.get("error", e.response.text)
                except Exception:
                    # Fallback to raw response text if JSON parsing fails
                    error_message = e.response.text

                # Handle specific error cases
                if e.response.status_code == 400:
                    error_type = "invalid_request_error"
                elif e.response.status_code == 500:
                    error_type = "api_error"
                    # Check if it's related to unsupported multimodal content based on backend error message
                    if (
                        "missing data required for image input"
                        in error_message.lower()
                    ):
                        error_message = (
                            "Model does not support multimodal image inputs"
                        )

                raise HTTPException(
                    status_code=400 if e.response.status_code == 400 else 500,
                    detail={
                        "error": {
                            "message": error_message,
                            "type": error_type,
                            "code": None,
                        }
                    },
                )

    async def _proxy_stream(self, backend, ollama_request, original_request):
        """Handle streaming requests."""

        async def stream():
            async with httpx.AsyncClient(follow_redirects=True) as client:
                try:
                    async with client.stream(
                        "POST",
                        backend.url + self.ollama_endpoint,
                        json=ollama_request,
                        timeout=120,
                    ) as response:
                        response.raise_for_status()
                        request_id = (
                            f"chatcmpl-{int(time.time())}"
                            if self.ollama_endpoint == "/api/chat"
                            else f"cmpl-{int(time.time())}"
                        )

                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    openai_chunk = self.translator.translate_response_chunk(
                                        chunk, request_id
                                    )
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue

                        yield "data: [DONE]\n\n"

                except httpx.HTTPStatusError as e:
                    # Convert backend errors to OpenAI-compatible error responses for streaming
                    error_message = "Backend error"
                    error_type = "api_error"

                    try:
                        # Extract error message from JSON response
                        error_data = e.response.json()
                        error_message = error_data.get("error", e.response.text)
                    except Exception:
                        # Fallback to raw response text if JSON parsing fails
                        error_message = e.response.text

                    # Handle specific error cases
                    if e.response.status_code == 400:
                        error_type = "invalid_request_error"
                    elif e.response.status_code == 500:
                        error_type = "api_error"
                        # Check if it's related to unsupported multimodal content based on backend error message
                        if (
                            "missing data required for image input"
                            in error_message.lower()
                        ):
                            error_message = (
                                "Model does not support multimodal image inputs"
                            )

                    # Yield error in streaming format
                    error_chunk = {
                        "error": {
                            "message": error_message,
                            "type": error_type,
                            "code": None,
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )


class Translator:
    """Base class for translating between OpenAI and Ollama formats."""

    def translate_request(self, request_data: dict) -> dict:
        """Translate OpenAI request format to Ollama format."""
        raise NotImplementedError

    def translate_response(
        self, ollama_response: dict, original_request: dict
    ) -> dict:
        """Translate Ollama response format to OpenAI format."""
        raise NotImplementedError

    def translate_response_chunk(
        self, ollama_chunk: dict, request_id: str
    ) -> dict:
        """Translate Ollama streaming chunk format to OpenAI format."""
        raise NotImplementedError

    def _map_openai_options_to_ollama(self, request_data: dict) -> dict:
        """Map OpenAI parameters to Ollama options format."""
        options = {}

        # Temperature
        if "temperature" in request_data:
            options["temperature"] = request_data["temperature"]

        # Max tokens (mapped to num_predict in Ollama)
        if "max_tokens" in request_data:
            options["num_predict"] = request_data["max_tokens"]

        # Top-p sampling
        if "top_p" in request_data:
            options["top_p"] = request_data["top_p"]

        # Frequency penalty
        if "frequency_penalty" in request_data:
            options["frequency_penalty"] = request_data["frequency_penalty"]

        # Presence penalty
        if "presence_penalty" in request_data:
            options["presence_penalty"] = request_data["presence_penalty"]

        # Stop sequences
        if "stop" in request_data:
            options["stop"] = request_data["stop"]

        # Random seed
        if "seed" in request_data:
            options["seed"] = request_data["seed"]

        return options

    def _handle_response_format(self, request_data: dict) -> Optional[str]:
        """Handle OpenAI response format and return Ollama format value."""
        response_format = request_data.get("response_format")
        if response_format and response_format.get("type") == "json_object":
            return "json"
        return None

    def _calculate_usage(self, ollama_response: dict) -> dict:
        """Calculate usage statistics from Ollama response."""
        prompt_tokens = ollama_response.get("prompt_eval_count", 0)
        completion_tokens = ollama_response.get("eval_count", 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _generate_request_id(self, prefix: str) -> str:
        """Generate a request ID with the given prefix."""
        return f"{prefix}-{int(time.time())}"


class ChatCompletionTranslator(Translator):
    """Translator for chat completion endpoints."""

    def translate_request(self, request_data: dict) -> dict:
        """Translate OpenAI chat completion request to Ollama format."""
        messages = request_data.get("messages", [])

        # Process messages to handle images for multimodal models
        processed_messages = []

        for message in messages:
            processed_message = {"role": message.get("role", "user")}
            message_images = []

            content = message.get("content")
            if isinstance(content, list):
                # Handle array of content parts (text + images)
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image/"):
                            # Extract base64 data from data URL
                            base64_data = (
                                image_url.split(",", 1)[1]
                                if "," in image_url
                                else ""
                            )
                            if base64_data:
                                message_images.append(base64_data)

                processed_message["content"] = " ".join(text_parts)
            else:
                # Handle simple string content
                processed_message["content"] = content or ""

            # Add images to the message if any were found
            if message_images:
                processed_message["images"] = message_images

            processed_messages.append(processed_message)

        ollama_data = {
            "model": request_data.get("model"),
            "messages": processed_messages,
            "stream": request_data.get("stream", False),
        }

        # Handle response format (JSON mode)
        format_value = self._handle_response_format(request_data)
        if format_value:
            ollama_data["format"] = format_value

        # Handle tools (function calling)
        if "tools" in request_data:
            ollama_data["tools"] = request_data["tools"]

        # Map OpenAI parameters to Ollama options
        options = self._map_openai_options_to_ollama(request_data)
        if options:
            ollama_data["options"] = options

        return ollama_data

    def translate_response(
        self, ollama_response: dict, original_request: dict
    ) -> dict:
        """Translate Ollama chat response to OpenAI format."""
        message = ollama_response.get("message", {})
        tool_calls = message.get("tool_calls")

        return {
            "id": self._generate_request_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original_request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                    "finish_reason": (
                        "tool_calls"
                        if tool_calls
                        else ("stop" if ollama_response.get("done") else None)
                    ),
                }
            ],
            "usage": self._calculate_usage(ollama_response),
        }

    def translate_response_chunk(
        self, ollama_chunk: dict, request_id: str
    ) -> dict:
        """Translate Ollama streaming chunk to OpenAI format."""
        delta = {}
        message = ollama_chunk.get("message", {})

        # Only include role in first chunk (when role is present)
        if message.get("role"):
            delta["role"] = message["role"]

        # Include content if present
        content = message.get("content", "")
        if content:
            delta["content"] = content

        # Include tool calls if present
        tool_calls = message.get("tool_calls")
        if tool_calls:
            delta["tool_calls"] = tool_calls

        # Determine finish reason
        finish_reason = None
        if ollama_chunk.get("done"):
            if tool_calls:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": ollama_chunk.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }


class CompletionTranslator(Translator):
    """Translator for text completion endpoints."""

    def translate_request(self, request_data: dict) -> dict:
        """Translate OpenAI completion request to Ollama format."""
        ollama_data = {
            "model": request_data.get("model"),
            "prompt": request_data.get("prompt", ""),
            "stream": request_data.get("stream", False),
        }

        # Handle suffix parameter
        if "suffix" in request_data:
            ollama_data["suffix"] = request_data["suffix"]

        # Handle response format (JSON mode)
        format_value = self._handle_response_format(request_data)
        if format_value:
            ollama_data["format"] = format_value

        # Map OpenAI parameters to Ollama options
        options = self._map_openai_options_to_ollama(request_data)
        if options:
            ollama_data["options"] = options

        return ollama_data

    def translate_response(
        self, ollama_response: dict, original_request: dict
    ) -> dict:
        """Translate Ollama completion response to OpenAI format."""
        return {
            "id": self._generate_request_id("cmpl"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": original_request["model"],
            "choices": [
                {
                    "index": 0,
                    "text": ollama_response.get("response", ""),
                    "finish_reason": (
                        "stop" if ollama_response.get("done") else None
                    ),
                }
            ],
            "usage": self._calculate_usage(ollama_response),
        }

    def translate_response_chunk(
        self, ollama_chunk: dict, request_id: str
    ) -> dict:
        """Translate Ollama streaming completion chunk to OpenAI format."""
        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": ollama_chunk.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "text": ollama_chunk.get("response", ""),
                    "finish_reason": (
                        "stop" if ollama_chunk.get("done") else None
                    ),
                }
            ],
        }


class EmbeddingTranslator(Translator):
    """Translator for embedding endpoints."""

    def translate_request(self, request_data: dict) -> dict:
        """Translate OpenAI embeddings request to Ollama format."""
        # Handle both single input and array of inputs
        input_data = request_data.get("input", "")

        # Ollama's /api/embed endpoint supports array of inputs in the "input" field
        ollama_data = {
            "model": request_data.get("model"),
            "input": input_data,  # Keep original format - Ollama handles both string and array
        }

        return ollama_data

    def translate_response(
        self, ollama_response: dict, original_request: dict
    ) -> dict:
        """Translate Ollama embeddings response to OpenAI format."""
        # Handle both single embedding and array of embeddings
        embeddings = ollama_response.get("embeddings", [])
        if not embeddings:
            # Fallback to single embedding format
            single_embedding = ollama_response.get("embedding", [])
            embeddings = [single_embedding] if single_embedding else []

        data = []
        for i, embedding in enumerate(embeddings):
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": original_request["model"],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "total_tokens": ollama_response.get("prompt_eval_count", 0),
            },
        }

    def translate_response_chunk(
        self, ollama_chunk: dict, request_id: str
    ) -> dict:
        """Embeddings don't support streaming, so this should not be called."""
        raise NotImplementedError("Embeddings do not support streaming")


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]


@router.get("/v1/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> ListResponse[AIModel]:
    pool = services.get(Pool)
    return ListResponse[AIModel](data=pool.models.values())


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, services: svcs.fastapi.DepContainer
) -> AIModel:
    pool = services.get(Pool)
    return pool.models[model_id]


@router.post("/v1/chat/completions")
async def chat_completions(
    r: Request, services: svcs.fastapi.DepContainer
) -> Any:
    translator = ChatCompletionTranslator()
    proxy = OpenAIProxy(services, translator, "/api/chat")
    return await proxy.proxy(r)


@router.post("/v1/completions")
async def completions(r: Request, services: svcs.fastapi.DepContainer) -> Any:
    translator = CompletionTranslator()
    proxy = OpenAIProxy(services, translator, "/api/generate")
    return await proxy.proxy(r)


@router.post("/v1/embeddings")
async def embeddings(r: Request, services: svcs.fastapi.DepContainer) -> Any:
    translator = EmbeddingTranslator()
    proxy = OpenAIProxy(services, translator, "/api/embed")
    return await proxy.proxy(r, allow_stream=False)
