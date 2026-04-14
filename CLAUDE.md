# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Enter development shell (requires Nix + devenv)
devenv shell

# Run all tests
run-tests

# Run a single test file
uv run pytest src/skvaider/inference/tests/test_manager.py -vv

# Run a specific test
uv run pytest src/skvaider/inference/tests/test_manager.py::test_manager_start_model -vv

# Start all services in the background (terminal stays free)
devenv up -d

# Stop background services
devenv down

# Type checking, linting, formatting, etc. all in one:
pre-commit run -a

```

## Architecture

Skvaider is an OpenAI-compatible API proxy with two parts.


### The OpenAI compatible gateway facing application clients (`skvaider:app_factory()`)

Routes requests to inference backends with load balancing, authentication, health checks and resource management.

- **Entry point**: `src/skvaider/__init__.py`
- **Config file**: `config.toml`
- **Port**: 8000

Key components:
- `proxy/pool.py` - Request queue and backend load balancing
- `proxy/backends.py` - Backend interface (SkvaiderBackend)
- `routers/openai.py` - OpenAI-compatible endpoints (`/openai/v1/...`)
- `auth.py` - Token authentication via aramaki

### Inference server (`skvaider.inference:app_factory()`)

Runs local LLMs via llama-server subprocesses.

- **Entry point**: `src/skvaider/inference/__init__.py`
- **Config file**: `config-inference-{1,2}.toml` (via `SKVAIDER_CONFIG_FILE` env var)
- **Ports**: 8001, 8002

Key components:
- `inference/manager.py` - Model lifecycle (download, start, health check, terminate)
- `inference/routers/models.py` - Model management endpoints (`/models/{name}/load`, `/models/{name}/proxy/{path}`)
- `inference/routers/manager.py` - Health and VRAM usage endpoints

### Aramaki (`src/aramaki/`)

WebSocket-based distributed state management for authentication tokens.

Aramaki is intended to be split off later into a separate package. It is extremely important that no references (imports) from aramaki (`src/aramaki`) to the skvaider code base (`src/skvaider`) are
introduced under any circumstances.

- `manager.py` - WebSocket connections and subscriptions
- `collection.py` - Collection protocol and replication
- `db.py` - SQLite persistence

## Request Flow

1. Client → Proxy (`/openai/v1/chat/completions`)
2. Proxy authenticates via aramaki tokens
3. Pool assigns request to least-loaded backend but batches requests that are incoming at the same time.
4. Backend proxies to inference server (`/models/{model}/proxy/v1/chat/completions`)
5. Proxy starts models as needed (llama-server subprocess). At least one reserved model instance should always be available. Additional models are stopped and started as needed.
6. Response streams back through the chain

## Model Status System

Models track two status dimensions (inspired by Ceph):
- `process_status`: stopped → starting → running → stopping
- `health_status`: "" → healthy/unhealthy

Combined into `status` set with "active" (running+healthy) or "inactive".

## Configuration

Pydantic models in `config.py` files. Key patterns:
- Model files: URL + SHA256 hash for verification
- Logging: structlog with IP anonymization

## Code Style

- "-> None"  is not needed on `__init__` methods
- if filtering through lists in a compound statement, prefer to use the `guardian` pattern to avoid long indentations.

  Good:

   ```
    for x in mylist:
      if not condition(x):
        continue
      ... do the happy path work ...
  ```

  Bad:

  ```
    for x in mylist:
      if condition(x):
        ... do the happy path work ...
  ```

- do not add superfluous comments to code that is already there. when making comments to
  new code you generate then do not make the comment if its basically exactly what the
  code already reads like or is sensibly obvious. stick to higher order "why" comments
  instead of superfluous comments

  bad examples:

  ```
    # do the foo bar thing
    do_foo_bar()

    # Get per-process VRAM usage from --showpids
    await self._update_per_model_vram_rocm()

    # Get total VRAM from --showmeminfo
    proc = await asyncio.create_subprocess_exec(
        "rocm-smi",
        "--json",
        "--showmeminfo",
        "all",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
  ```

- do not make overly aggressive use of _ (underscore) methods or attributes. this is python, not java.

- if you log an exception, use the log.exception() function to ensure we see a proper traceback

- basedpyright strict mode
- black + isort (line length 80)
- ruff (ignoring E501, F401)
