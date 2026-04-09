#!/usr/bin/env python3
"""
Test script to verify OpenAI client compatibility with our API gateway.
"""

from openai import OpenAI


def test_model_list(openai_client: OpenAI, llm_model_name: str):
    models = openai_client.models.list()
    assert len(models.data) >= 1
    assert llm_model_name in [x.id for x in models.data]


def test_chat_completions(openai_client: OpenAI, llm_model_name: str):
    response = openai_client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": "Say 'hello world'"}],
        max_tokens=50,  # More generous token budget
    )
    assert response.choices[0].message.content
    assert response.usage
    assert 0 < response.usage.total_tokens < 100


def test_chat_completions_model_normalization(
    openai_client: OpenAI, llm_model_name: str
):
    mixed_case_model = llm_model_name.upper()
    response = openai_client.chat.completions.create(
        model=mixed_case_model,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    assert response.choices[0].message.content


# def test_chat_completions_nonstreaming_reasoning(openai_client):
#     response = openai_client.chat.completions.create(
#         model="gpt-oss",
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


def test_chat_completions_streaming(openai_client: OpenAI, llm_model_name: str):
    stream = openai_client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        max_tokens=100,  # More generous for streaming
        stream=True,
    )
    chunk_count = 0
    full_content = ""
    for chunk in stream:
        chunk_count += 1
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content or ""
        full_content += content
        if chunk_count >= 100:  # Stop after reasonable number
            break

    assert full_content


def test_completions(openai_client: OpenAI, llm_model_name: str):
    response = openai_client.completions.create(
        model=llm_model_name,
        prompt="The capital of France is",
        max_tokens=50,  # More generous token budget
    )
    assert response.choices[0].text
    assert response.usage
    assert 0 < response.usage.total_tokens <= 100


def test_completions_streaming(openai_client: OpenAI, llm_model_name: str):
    stream = openai_client.completions.create(
        model=llm_model_name,
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


def test_embeddings(openai_client: OpenAI):
    response = openai_client.embeddings.create(
        input="Test String", model="embeddinggemma"
    )
    assert response.data is not None
    assert len(response.data[0].embedding) >= 100
