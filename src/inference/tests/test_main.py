import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from inference.main import app

client = TestClient(app)


@pytest.fixture
def clean_models_dir():
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    yield
    if os.path.exists("models"):
        shutil.rmtree("models")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Use a long timeout for this test
@pytest.mark.timeout(1200)
def test_integration_real_download_and_inference():
    # Setup
    if os.path.exists("models"):
        shutil.rmtree("models")

    # Client timeout needs to be long enough for download
    client = TestClient(app)

    model_url = "https://huggingface.co/M4-ai/TinyMistral-248M-v2-Instruct-GGUF/resolve/main/TinyMistral-248M-v2-Instruct.Q2_K.gguf?download=true"
    filename = "TinyMistral-248M-v2-Instruct.Q2_K.gguf"
    model_name = "TinyMistral-248M-v2-Instruct"

    print("\n[Integration] 1. Downloading model... this may take a while.")
    response = client.post(
        "/download",
        json={
            "url": model_url,
            "filename": filename,
            "metadata": {
                "name": model_name,
                "context_size": 2048,
                "cmd_args": ["-ngl", "0"],  # Force CPU
            },
        },
    )

    if response.status_code != 200:
        print(f"Download failed: {response.text}")

    assert response.status_code == 200
    assert response.json()["status"] == "downloaded"
    assert os.path.exists(f"models/{filename}")
    print("[Integration] Download complete.")

    # 2. Inference
    print("[Integration] 2. Running inference...")
    # We don't request streaming, so we expect a full JSON response
    response = client.post("/load", json={"model": model_name})

    if response.status_code != 200:
        print(f"Load failed: {response.text}")

    assert response.status_code == 200
    port = response.json()["port"]

    # Now connect directly to the model
    import httpx

    response = httpx.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with just the number.",
                }
            ],
        },
        timeout=120,
    )

    if response.status_code != 200:
        print(f"Inference failed: {response.text}")

    assert response.status_code == 200

    content = response.text
    print(f"[Integration] Response: {content}")

    # Check for expected answer
    assert "4" in content or "four" in content.lower()

    # Cleanup
    if os.path.exists("models"):
        shutil.rmtree("models")
