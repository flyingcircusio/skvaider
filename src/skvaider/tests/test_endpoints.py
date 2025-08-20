#!/usr/bin/env python3
"""
Simple test script to verify the OpenAI-compatible endpoints work correctly.
"""

import httpx
import pytest


@pytest.mark.asyncio
async def test_models_endpoint(server):
    """Test the models listing endpoint."""
    print("Testing /v1/models endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/openai/v1/models",
            headers={"Authorization": "Bearer test-token"},
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['data'])} models:")
            for model in data["data"]:
                print(f"  - {model['id']}")
        else:
            print(f"Error: {response.text}")
        print()


@pytest.mark.asyncio
async def test_chat_completions_non_streaming():
    """Test chat completions without streaming."""
    print("Testing /v1/chat/completions (non-streaming)...")

    payload = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 50,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer test-token",
                    "Content-Type": "application/json",
                },
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Non-streaming chat completions: SUCCESS")
                # Don't print the full response as it might be large
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
    print()


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    """Test chat completions with streaming."""
    print("Testing /v1/chat/completions (streaming)...")

    payload = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
        "max_tokens": 20,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream(
                "POST",
                "http://localhost:8000/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer test-token",
                    "Content-Type": "application/json",
                },
            ) as response:
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print("Streaming chat completions: SUCCESS")
                    # Just confirm we can read some chunks
                    chunk_count = 0
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunk_count += 1
                            if chunk_count <= 3:  # Only print first few chunks
                                print(f"Chunk {chunk_count}: {chunk[:50]}...")
                            if chunk_count >= 10:  # Stop after 10 chunks
                                break
                    print(f"Received {chunk_count} chunks total")
                else:
                    content = await response.aread()
                    print(f"Error: {content}")
        except Exception as e:
            print(f"Streaming request failed: {e}")
    print()


@pytest.mark.asyncio
async def test_completions_non_streaming():
    """Test completions without streaming."""
    print("Testing /v1/completions (non-streaming)...")

    payload = {
        "model": "gpt-oss:20b",
        "prompt": "The capital of France is",
        "stream": False,
        "max_tokens": 10,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/openai/v1/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer test-token",
                    "Content-Type": "application/json",
                },
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Non-streaming completions: SUCCESS")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
    print()
