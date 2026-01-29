import json
import pathlib

import httpx

from skvaider.inference.manager import Model


async def test_embeddinggemma_output_stability(embeddinggemma: Model):
    await embeddinggemma.start()

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{embeddinggemma.endpoint}/v1/embeddings",
            json={
                "input": "why is the sky blue?",
                "temperature": 0.0,
                "seed": 42,
                "cache_prompt": False,
            },
        )  # inspired by ollama/integration/model_arch_Test_go.
        assert response.status_code == 200

        # uncomment to update the expected output
        # with open(pathlib.Path(__file__).parent / "fixtures" / "embeddinggemma_stability_output.json", "w") as f:
        #     f.write(response.text)

        with open(
            pathlib.Path(__file__).parent
            / "fixtures"
            / "embeddinggemma_stability_output.json",
            "r",
        ) as f:
            expected_response = json.load(f)
        assert response.json() == expected_response
