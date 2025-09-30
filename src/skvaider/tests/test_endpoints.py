#!/usr/bin/env python3
"""
Simple test script to verify the OpenAI-compatible endpoints work correctly.
"""
import httpx


def test_list_models(client, auth_header):
    response = client.get("http://localhost:8000/openai/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) >= 1
    assert "gemma3:1b" in [m["id"] for m in data]


def test_get_model(client, auth_header):
    response = client.get("http://localhost:8000/openai/v1/models/gemma3:1b")
    assert response.status_code == 200
    model = response.json()
    assert set(model) == {"created", "id", "object", "owned_by"}
    assert model["id"] == "gemma3:1b"
    assert model["object"] == "model"
    assert model["owned_by"] == "library"


def test_completions_with_non_existing_model(client, auth_header):
    payload = {
        "model": "non-existing",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 50,
    }
    response = client.post(
        "http://localhost:8000/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400


def test_chat_completions_non_streaming(client, auth_header):
    payload = {
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 50,
    }
    response = client.post(
        "http://localhost:8000/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text


def test_chat_completions_streaming(client, auth_header):
    payload = {
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
        "max_tokens": 20,
    }
    with client.stream(
        "POST",
        "http://localhost:8000/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    ) as response:
        assert response.status_code == 200, response.read()
        assert (
            response.headers["content-type"]
            == "text/event-stream; charset=utf-8"
        )
        # Just confirm we can read some chunks
        chunk_count = 0
        for chunk in response.iter_text():
            if chunk.strip():
                chunk_count += 1
                if chunk_count <= 3:  # Only print first few chunks
                    print(f"Chunk {chunk_count}: {chunk[:50]}...")
                if chunk_count >= 10:
                    break


def test_completions_non_streaming(client, auth_header):
    payload = {
        "model": "gemma3:1b",
        "prompt": "The capital of France is",
        "stream": False,
        "max_tokens": 10,
    }
    response = client.post(
        "http://localhost:8000/openai/v1/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"


def test_unload_load_all_backends(client, auth_header, ollama_backend_urls):
    """Test unloading all models from all backends, then loading via requests, and verifying all models are correctly loaded"""

    # Unload all models from all backends
    with httpx.Client() as ollama_client:
        for backend_url in ollama_backend_urls:
            ps_response = ollama_client.get(f"{backend_url}/api/ps")
            if ps_response.status_code == 200:
                ps_data = ps_response.json()
                loaded_models = ps_data.get("models", [])

                for model_entry in loaded_models:
                    model_name = model_entry.get("model") or model_entry.get(
                        "name"
                    )
                    if model_name:
                        unload_response = ollama_client.post(
                            f"{backend_url}/api/generate",
                            json={"model": model_name, "keep_alive": 0},
                        )
                        assert (
                            unload_response.status_code == 200
                        ), f"Failed to unload {model_name} from {backend_url}: {unload_response.text}"
                        unload_data = unload_response.json()
                        assert (
                            unload_data.get("done_reason") == "unload"
                        ), f"Model {model_name} was not properly unloaded"

    # Verify all models are unloaded
    with httpx.Client() as ollama_client:
        for backend_url in ollama_backend_urls:
            ps_response = ollama_client.get(f"{backend_url}/api/ps")
            assert (
                ps_response.status_code == 200
            ), f"Failed to get status from {backend_url}"
            ps_data = ps_response.json()
            loaded_models = ps_data.get("models", [])
            assert (
                len(loaded_models) == 0
            ), f"Backend {backend_url} still has loaded models: {loaded_models}"

    # Load models using requests through the API
    test_models = ["gemma3:1b"]

    for model_name in test_models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 10,
        }
        response = client.post(
            "http://localhost:8000/openai/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
        )
        assert (
            response.status_code == 200
        ), f"Failed to make request with model {model_name}: {response.text}"

    # Collect all loaded models from all backends
    all_loaded_models = {}

    with httpx.Client() as ollama_client:
        for backend_url in ollama_backend_urls:
            ps_response = ollama_client.get(f"{backend_url}/api/ps")
            assert (
                ps_response.status_code == 200
            ), f"Failed to get status from {backend_url}"
            ps_data = ps_response.json()
            loaded_models = ps_data.get("models", [])

            for entry in loaded_models:
                model_name = entry.get("model") or entry.get("name")
                if model_name:
                    all_loaded_models[model_name] = (backend_url, entry)

    # Verify each test model is loaded on at least one backend with correct configuration
    for model_name in test_models:
        assert (
            model_name in all_loaded_models
        ), f"Model {model_name} is not loaded on any backend! Available models: {list(all_loaded_models.keys())}"

        backend_url, model_entry = all_loaded_models[model_name]

        # Verify that custom context length is applied
        if model_name.startswith("gemma3"):
            expected_ctx = 3072
            actual_ctx = model_entry.get("context_length")
            assert (
                actual_ctx == expected_ctx
            ), f"Model {model_name} on {backend_url} has context_length {actual_ctx}, expected {expected_ctx}"
