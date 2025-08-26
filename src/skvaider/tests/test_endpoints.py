#!/usr/bin/env python3
"""
Simple test script to verify the OpenAI-compatible endpoints work correctly.
"""


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
