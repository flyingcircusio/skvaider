# we want to test that models are loaded correctly when new backends are added

import asyncio

import httpx

import skvaider.proxy.backends
from skvaider.conftest import wait_for_condition


async def test_backend_model_warmup(llm_model_name, services):
    url = "http://127.0.0.1:8001"

    @wait_for_condition()
    async def backend_connection_is_up():
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/manager/health")
            if resp.status_code == 200:
                return True

    await backend_connection_is_up()

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
