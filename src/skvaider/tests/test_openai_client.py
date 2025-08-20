#!/usr/bin/env python3
"""
Test script to verify OpenAI client compatibility with our API gateway.
"""

import pytest
from openai import AsyncOpenAI, OpenAI


def test_openai_client_sync():
    """Test using the synchronous OpenAI client."""
    print("=== Testing with OpenAI sync client ===\n")

    # Initialize client pointing to our local server
    client = OpenAI(
        base_url="http://localhost:8000/openai/v1", api_key="test-token"
    )

    # Test 1: List models
    print("1. Testing list models...")
    try:
        models = client.models.list()
        print(f"✅ Found {len(models.data)} models:")
        for model in models.data:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    print()

    # Test 2: Chat completions (non-streaming) - Standard model
    print("2. Testing chat completions (non-streaming) - Standard model...")
    try:
        response = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=50,  # More generous token budget
        )
        print("✅ Chat completion successful!")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Error with chat completion: {e}")
    print()

    # Test 2b: Chat completions with reasoning model (needs more tokens)
    print("2b. Testing chat completions - Reasoning model (gpt-oss:20b)...")
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Explain your reasoning briefly.",
                }
            ],
            max_tokens=200,  # Much larger budget for reasoning models
        )
        print("✅ Reasoning model completion successful!")
        # Check both content and reasoning fields
        content = response.choices[0].message.content or ""
        reasoning = getattr(response.choices[0].message, "reasoning", "") or ""
        print(
            f"   Content: {content[:100]}{'...' if len(content) > 100 else ''}"
        )
        if reasoning:
            print(
                f"   Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}"
            )
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Error with reasoning model: {e}")
    print()

    # Test 3: Chat completions (streaming)
    print("3. Testing chat completions (streaming)...")
    try:
        stream = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            max_tokens=100,  # More generous for streaming
            stream=True,
        )
        print("✅ Streaming chat completion successful!")
        chunk_count = 0
        full_content = ""
        for chunk in stream:
            chunk_count += 1
            content = (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta.content
                else ""
            )
            full_content += content
            if chunk_count <= 5:  # Show first 5 chunks
                print(f"   Chunk {chunk_count}: '{content}'")
            if chunk_count >= 20:  # Stop after reasonable number
                break
        print(f"   Full response: '{full_content}'")
        print(f"   Total chunks received: {chunk_count}")
    except Exception as e:
        print(f"❌ Error with streaming chat completion: {e}")
    print()

    # Test 4: Text completions (non-streaming)
    print("4. Testing text completions (non-streaming)...")
    try:
        response = client.completions.create(
            model="llama3.1:8b",
            prompt="The capital of France is",
            max_tokens=50,  # More generous token budget
        )
        print("✅ Text completion successful!")
        print(f"   Response: '{response.choices[0].text}'")
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Error with text completion: {e}")
    print()


@pytest.mark.asyncio
async def test_openai_client_async():
    """Test using the asynchronous OpenAI client."""
    print("=== Testing with OpenAI async client ===\n")

    # Initialize async client pointing to our local server
    client = AsyncOpenAI(
        base_url="http://localhost:8000/openai/v1", api_key="test-token"
    )

    # Test 1: List models
    print("1. Testing async list models...")
    try:
        models = await client.models.list()
        print(f"✅ Found {len(models.data)} models:")
        for model in models.data:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    print()

    # Test 2: Chat completions (non-streaming)
    print("2. Testing async chat completions (non-streaming)...")
    try:
        response = await client.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Say goodbye in one word"}],
            max_tokens=50,  # More generous token budget
        )
        print("✅ Async chat completion successful!")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Error with async chat completion: {e}")
    print()

    # Test 3: Async streaming chat completions
    print("3. Testing async chat completions (streaming)...")
    try:
        stream = await client.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "List 3 colors"}],
            max_tokens=100,  # More generous for streaming
            stream=True,
        )
        print("✅ Async streaming chat completion successful!")
        chunk_count = 0
        full_content = ""
        async for chunk in stream:
            chunk_count += 1
            content = (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta.content
                else ""
            )
            full_content += content
            if chunk_count <= 5:  # Show first 5 chunks
                print(f"   Chunk {chunk_count}: '{content}'")
            if chunk_count >= 20:  # Stop after reasonable number
                break
        print(f"   Full response: '{full_content}'")
        print(f"   Total chunks received: {chunk_count}")
    except Exception as e:
        print(f"❌ Error with async streaming chat completion: {e}")
    print()

    # Test 4: Async text completions
    print("4. Testing async text completions (non-streaming)...")
    try:
        response = await client.completions.create(
            model="llama3.1:8b",
            prompt="Python is a",
            max_tokens=50,  # More generous token budget
        )
        print("✅ Async text completion successful!")
        print(f"   Response: '{response.choices[0].text}'")
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Error with async text completion: {e}")
    print()

    await client.close()
