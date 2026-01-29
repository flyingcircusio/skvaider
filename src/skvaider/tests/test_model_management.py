# we want to test that models are loaded correctly when new backends are added

import asyncio

import httpx
import svcs

import skvaider.proxy.backends
import skvaider.proxy.pool
from skvaider.conftest import backend_connection_is_up


async def test_backend_model_warmup(
    llm_model_name: str, services: svcs.Container
):
    url = "http://127.0.0.1:8001"

    await backend_connection_is_up(url)

    # call /models/{llm_model_name}/unload to ensure model is not loaded
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{url}/models/{llm_model_name}/unload")
        assert resp.status_code == 200

    # now, load the test client
    pool = skvaider.proxy.pool.Pool()
    pool.add_backend(skvaider.proxy.backends.SkvaiderBackend(url))
    await asyncio.sleep(20)
    # for model known to pool, all should be loaded
    model_backends = [b for b in pool.backends if llm_model_name in b.models]
    models_in_backends = [b.models[llm_model_name] for b in model_backends]
    # it should be loaded in any of the backends
    assert any(m.is_loaded for m in models_in_backends)
