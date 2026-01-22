#!/usr/bin/env python3
"""
Simple test script to verify the OpenAI-compatible endpoints work correctly.
"""


def test_list_models(client, auth_header, llm_model_name):
    response = client.get("/openai/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) >= 1
    assert llm_model_name in [m["id"] for m in data]


def test_get_model(client, auth_header, llm_model_name):
    response = client.get(f"/openai/v1/models/{llm_model_name}")
    assert response.status_code == 200
    model = response.json()
    assert set(model) == {"created", "id", "object", "owned_by"}
    assert model["id"] == llm_model_name
    assert model["object"] == "model"
    # assert model["owned_by"] == "skvaider"


def test_completions_with_non_existing_model(client, auth_header):
    payload = {
        "model": "non-existing",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 50,
    }
    response = client.post(
        "/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400


def test_chat_completions_non_streaming(client, auth_header, llm_model_name):
    payload = {
        "model": llm_model_name,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 50,
    }
    response = client.post(
        "/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text


def test_chat_completions_streaming(client, auth_header, llm_model_name):
    payload = {
        "model": llm_model_name,
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
        "max_tokens": 20,
    }
    with client.stream(
        "POST",
        "/openai/v1/chat/completions",
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


def test_multiple_streaming_requests(client, auth_header, llm_model_name):
    """
    Test multiple concurrent streaming requests to verify pool management.

    Each request asks the model to count from 1 to 5, and we stagger the start times slightly. (0.2s apart, because batching waits up to 0.1s)
    5 requests are made in total, and we verify that all complete successfully.
    """
    import threading
    import time

    results = []
    lock = threading.Lock()

    def make_request(index):
        payload = {
            "model": llm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"Count from 1 to 5, request {index}",
                }
            ],
            "stream": True,
            "max_tokens": 50,
        }
        with client.stream(
            "POST",
            "/openai/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status_code != 200:
                with lock:
                    results.append(
                        (index, False, f"Status code: {response.status_code}")
                    )
                return
            # Read the streamed response
            content = ""
            for chunk in response.iter_text():
                if chunk.strip():
                    content += chunk
            with lock:
                results.append((index, True, content))

    threads = []
    for i in range(5):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(0.2)  # Stagger the start times slightly

    for t in threads:
        t.join()

    for index, success, content in results:
        assert success, f"Request {index} failed: {content}"


def test_completions_non_streaming(client, auth_header, llm_model_name):
    payload = {
        "model": llm_model_name,
        "prompt": "The capital of France is",
        "stream": False,
        "max_tokens": 10,
    }
    response = client.post(
        "/openai/v1/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"


def test_model_context_limit_applied(client, auth_header, llm_model_name):
    """Test that custom context limits are applied when loading models"""
    import time

    # First, make a chat completion request to ensure TinyMistral-248M-v2-Instruct is loaded with custom options
    payload = {
        "model": llm_model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "max_tokens": 10,
    }
    response = client.post(
        "/openai/v1/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text

    # Give the model a moment to fully load
    time.sleep(2)

    # TODO: Verify context size in logs or via some other mechanism
    # For now, we just verify the request succeeded
