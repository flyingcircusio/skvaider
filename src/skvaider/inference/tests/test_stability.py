import httpx
import pytest

from skvaider.dummy_engine import DummyModel


@pytest.mark.timeout(30)
async def test_dummy_embedding_determinism(embeddinggemma: DummyModel):
    """Dummy engine produces deterministic embeddings."""
    await embeddinggemma.start()

    async with httpx.AsyncClient(timeout=30) as client:
        response1 = await client.post(
            f"{embeddinggemma.endpoint}/v1/embeddings",
            json={"input": "why is the sky blue?"},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        response2 = await client.post(
            f"{embeddinggemma.endpoint}/v1/embeddings",
            json={"input": "why is the sky blue?"},
        )
        assert response2.status_code == 200
        data2 = response2.json()

    # Same input → same deterministic embedding
    assert data1["data"][0]["embedding"] == data2["data"][0]["embedding"]
    assert len(data1["data"][0]["embedding"]) > 0
