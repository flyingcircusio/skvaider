#!/usr/bin/env python3
"""
Test script to verify OpenAI client compatibility with our API gateway.
"""

import pytest
from openai import OpenAI


@pytest.fixture
def openai_client(client, auth_token):
    yield OpenAI(
        base_url="http://localhost:8000/openai/v1",
        http_client=client,
        api_key=auth_token,
    )


def test_model_list(openai_client):
    models = openai_client.models.list()
    assert len(models.data) >= 1
    assert "TinyMistral-248M-v2-Instruct" in [x.id for x in models.data]


def test_chat_completions(openai_client):
    response = openai_client.chat.completions.create(
        model="TinyMistral-248M-v2-Instruct",
        messages=[{"role": "user", "content": "Say 'hello world'"}],
        max_tokens=50,  # More generous token budget
    )
    assert response.choices[0].message.content
    assert 0 < response.usage.total_tokens < 75


# gpt-oss:20b is just too large to download on every commit ... maybe
# rather use a fake backend for this?
#
# def test_chat_completions_nonstreaming_reasoning(openai_client):
#     response = openai_client.chat.completions.create(
#         model="gpt-oss:20b",
#         messages=[
#             {
#                 "role": "user",
#                 "content": "What is 2+2? Explain your reasoning briefly.",
#             }
#         ],
#         max_tokens=200,  # Much larger budget for reasoning models
#     )
#     content = response.choices[0].message.content
#     assert content
#     reasoning = getattr(response.choices[0].message, "reasoning", "")
#     assert reasoning

#     assert response.usage.total_tokens < 200


def test_chat_completions_streaming(openai_client):
    stream = openai_client.chat.completions.create(
        model="TinyMistral-248M-v2-Instruct",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        max_tokens=100,  # More generous for streaming
        stream=True,
    )
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
        if chunk_count >= 100:  # Stop after reasonable number
            break

    assert full_content


def test_completions(openai_client):
    response = openai_client.completions.create(
        model="TinyMistral-248M-v2-Instruct",
        prompt="The capital of France is",
        max_tokens=50,  # More generous token budget
    )
    assert response.choices[0].text
    assert 0 < response.usage.total_tokens <= 100


def test_completions_streaming(openai_client):
    stream = openai_client.completions.create(
        model="TinyMistral-248M-v2-Instruct",
        prompt="The capital of France is",
        max_tokens=50,  # More generous token budget
        stream=True,
    )
    chunk_count = 0
    full_content = ""
    for chunk in stream:
        chunk_count += 1
        content = chunk.choices[0].text if chunk.choices[0].text else ""
        full_content += content
        if chunk_count <= 5:  # Show first 5 chunks
            print(f"   Chunk {chunk_count}: '{content}'")
        if chunk_count >= 20:  # Stop after reasonable number
            break

    assert full_content
    assert 0 < chunk_count < 100


def test_embeddings(openai_client):
    response = openai_client.embeddings.create(
        input="Test String", model="TinyMistral-248M-v2-Instruct"
    )
    assert response.data is not None
    assert len(response.data[0].embedding) >= 100
