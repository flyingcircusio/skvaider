from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI


@pytest.fixture
def openai_client(
    client: TestClient, auth_token: str
) -> Generator[OpenAI, None, None]:
    yield OpenAI(
        base_url="http://testserver/openai/v1",
        http_client=client,  # pyright: ignore[reportArgumentType]
        api_key=auth_token,
    )


def test_chat_completions_model_normalization(
    openai_client: OpenAI, llm_model_name: str
):
    """
    Test that model name is normalized to lowercase in chat completions.
    """
    mixed_case_model = llm_model_name.upper()
    response = openai_client.chat.completions.create(
        model=mixed_case_model,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    assert response.choices[0].message.content
