import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import skvaider.proxy.backends
import skvaider.proxy.pool
from skvaider import app_factory

hasher = PasswordHasher()


class DummyTokens:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def keys(self):
        return self.data.keys()


DUMMY_TOKENS = DummyTokens()


@pytest.fixture
def services():
    reg = svcs.Registry()
    reg.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)
    with svcs.Container(reg) as container:
        yield container


@svcs.fastapi.lifespan
async def test_lifespan(app: FastAPI, registry: svcs.Registry):
    pool = skvaider.proxy.pool.Pool()
    model_config = skvaider.proxy.backends.ModelConfig(
        {
            "TinyMistral-248M-v2-Instruct": {"num_ctx": 2048},
            "TinyMistral-248M-v2-Instruct-Embed": {"num_ctx": 2048},
        }
    )

    # Ensure model exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_name = "TinyMistral-248M-v2-Instruct"
    model_name_embed = "TinyMistral-248M-v2-Instruct-Embed"

    filename = "TinyMistral-248M-v2-Instruct.Q2_K.gguf"
    filename_embed = "TinyMistral-248M-v2-Instruct-Embed.Q2_K.gguf"

    model_path = models_dir / filename
    model_path_embed = models_dir / filename_embed

    json_path = models_dir / f"{filename}.json"
    json_path_embed = models_dir / f"{filename_embed}.json"

    if not model_path.exists():
        url = "https://huggingface.co/M4-ai/TinyMistral-248M-v2-Instruct-GGUF/resolve/main/TinyMistral-248M-v2-Instruct.Q2_K.gguf?download=true"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                with open(model_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

    if not model_path_embed.exists():
        os.symlink(filename, model_path_embed)

    with open(json_path, "w") as f:
        json.dump(
            {
                "name": model_name,
                "context_size": 2048,
                "cmd_args": [],
            },
            f,
        )

    with open(json_path_embed, "w") as f:
        json.dump(
            {
                "name": model_name_embed,
                "context_size": 2048,
                "cmd_args": ["--embedding", "--pooling", "mean"],
            },
            f,
        )

    import socket
    import subprocess
    import sys
    import time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]

    # Start inference server
    proc = subprocess.Popen(
        [sys.executable, "-m", "skvaider.inference.main"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = f"http://localhost:{port}"
    start = time.time()
    while time.time() - start < 30:  # Increased timeout to 30s
        if proc.poll() is not None:
            # Process exited prematurely
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"Inference server exited prematurely with code {proc.returncode}.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
            )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    break
        except Exception:
            await asyncio.sleep(0.1)
    else:
        proc.terminate()
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Inference server failed to start within timeout.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
        )

    pool.add_backend(skvaider.proxy.backends.SkvaiderBackend(url, model_config))

    if ollama_host := os.environ.get("OLLAMA_HOST"):
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        pool.add_backend(
            skvaider.proxy.backends.OllamaBackend(ollama_host, model_config)
        )

    registry.register_value(skvaider.proxy.pool.Pool, pool)
    registry.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)

    # Wait for backends to become healthy
    timeout = 2 * pool.backends[0].health_interval
    start = time.time()
    while time.time() - start < timeout:
        if all(b.healthy for b in pool.backends):
            break
        await asyncio.sleep(0.1)

    yield {}
    pool.close()
    proc.terminate()
    proc.wait()


@pytest.fixture
def token_db():
    DUMMY_TOKENS.data.clear()
    yield DUMMY_TOKENS


@pytest.fixture
async def auth_token(token_db):
    """Return a valid auth token."""
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    )
    yield auth_token.decode("ascii")


@pytest.fixture
async def auth_header(client, auth_token):
    """Inject a valid auth header into all client requests."""
    header = {"Authorization": f"Bearer {auth_token}"}
    client.headers.update(header)
    yield


@pytest.fixture
def client():
    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client
