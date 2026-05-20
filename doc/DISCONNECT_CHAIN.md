# Client Disconnect Chain Analysis

**Goal**: Verify that when a client disconnects mid-request, the cancellation properly propagates through the entire chain and frees resources at each layer.

## Architecture

```
Client → Gateway (proxy) → Inference Server → vLLM subprocess
```

Each hop is an HTTP connection. Streaming responses use SSE.

---

## Layer 1: Client → Gateway

**File**: `src/skvaider/routers/openai.py`

### Non-streaming requests
- `OpenAIProxy._execute_with_retry()` → `await backend.post(endpoint, data, ...)`
- `backend.post()` creates a one-shot `httpx.AsyncClient` and awaits the response
- If client disconnects, FastAPI/Starlette cancels the handler coroutine
- The `async with httpx.AsyncClient` context manager exits on cancellation → **connection closed** ✅

### Streaming requests
- `OpenAIProxy._execute_stream_with_retry()` → `stream_aws = backend.post_stream(...)`
- Returns `StreamingResponse(stream(first_chunk, stream_aws, context, ...), ...)`
- The inner `stream()` generator yields chunks from `stream_aws`
- `finally` block calls `await context.__aexit__(None, None, None)` (releases semaphore)

**Problem**: `stream_aws` (the async generator from `post_stream`) is **not explicitly closed** in the `finally` block. When Starlette detects a client disconnect, it calls `aclose()` on the response body generator. The gateway's `stream()` generator receives `GeneratorExit`, enters `finally`, but `stream_aws` is abandoned — its own cleanup (closing the HTTP connection to the inference server) depends on Python's async generator finalization, which is **not guaranteed to be awaited**.

**⚠️ GAP**: `stream_aws.aclose()` is never called. The HTTP connection to the inference server may stay open until the 120s timeout.

---

## Layer 2: Gateway → Inference Server

**File**: `src/skvaider/proxy/backends.py` — `SkvaiderBackend.post_stream()`

```python
async with httpx.AsyncClient(follow_redirects=True) as client:
    async with client.stream("POST", url, json=data, ...) as response:
        async for event in httpx_sse.EventSource(response).aiter_sse():
            yield f"data: {event.data}\n\n"
```

- Uses `client.stream()` as an async context manager
- If the async generator is properly `aclose()`d, the context manager exits → **connection closed** ✅
- If abandoned (see Layer 1 gap), the `async with` blocks stay entered until GC finalizes the generator, which may not await cleanup

**⚠️ Depends on Layer 1**: If the gateway doesn't close `stream_aws`, this connection hangs.

---

## Layer 3: Inference Server → vLLM

**File**: `src/skvaider/inference/routers/models.py` — `proxy_request()`

```python
upstream = await client.send(req, stream=True)

async def stream() -> AsyncGenerator[bytes]:
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()
        await client.aclose()
        await model.lock.user_release()
```

- Opens HTTP connection to vLLM subprocess
- Streams response body back to the gateway
- `finally` block properly closes upstream connection and releases model lock

**✅ Correct**: The `finally` block closes all resources. If Starlette calls `aclose()` on this generator (triggered by the gateway disconnecting), the vLLM connection is properly closed.

### What happens in vLLM? — **TESTED**
- vLLM runs as a separate process with its own HTTP server
- When the HTTP connection is closed, vLLM detects the broken pipe but **continues generating**
- vLLM has an `abort` endpoint (`POST /v1/chat/completions/abort`) but **nothing calls it**

**Test results** (ike00, gpt-oss:20b, 2026-05-20 13:20):

| Test | Client Action | Server Duration | Verdict |
|------|---------------|-----------------|--------|
| Streaming, disconnect after 5 chunks | aclose() at 0.04s | 0.103s | ⚠️ HTTP closed fast, vLLM likely continued |
| Non-streaming, cancel after 0.5s | task.cancel() at 0.5s | **7.758s** | ❌ vLLM generated full response |

**Root cause**: `upstream.aclose()` closes the HTTP response, but vLLM continues the generation task. For a 2000-token request this wasted ~7s of GPU time; longer generations waste proportionally more.

---

## Summary

| Layer | Disconnect Handling | Status |
|-------|-------------------|--------|
| Client → Gateway (non-stream) | `async with` exits on cancel | ✅ |
| Client → Gateway (stream) | `stream_aws` not closed in `finally` | ⚠️ GAP |
| Gateway → Inference | Depends on Layer 1 closing `stream_aws` | ⚠️ GAP |
| Inference → vLLM | `finally` closes connection properly | ✅ HTTP only |
| vLLM internal | No abort signal sent | ❌ CONFIRMED waste |

## Recommendations

1. **Gateway `stream()` finally block**: Add `stream_aws.aclose()` to explicitly close the backend connection
2. **vLLM abort**: Consider sending `POST /v1/chat/completions/abort` with the request ID when the client disconnects (requires tracking request IDs through the chain)
