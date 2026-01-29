import json
import pathlib

import httpx
import pytest

from skvaider.inference.manager import Model


@pytest.mark.timeout(120)
async def test_embeddinggemma_output_stability(embeddinggemma: Model):
    await embeddinggemma.start()

    async with httpx.AsyncClient(timeout=120) as client:
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

        # check data, max 1e-2 difference in each embedding value
        resp_json = response.json()
        for resp_item, exp_item in zip(
            resp_json["data"], expected_response["data"]
        ):
            resp_embedding = resp_item["embedding"]
            exp_embedding = exp_item["embedding"]
            assert len(resp_embedding) == len(exp_embedding)
            for r_val, e_val in zip(resp_embedding, exp_embedding):
                assert abs(r_val - e_val) < 1e-2
        # delete data to compare the rest of the response
        del resp_json["data"]
        del expected_response["data"]
        assert resp_json == expected_response
