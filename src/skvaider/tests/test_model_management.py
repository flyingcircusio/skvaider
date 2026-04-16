# we want to test that models are loaded correctly when new backends are added

import httpx
import pytest
import svcs

import skvaider.proxy.backends
import skvaider.proxy.pool
from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.conftest import backend_connection_is_up, wait_for_condition


# XXX
@pytest.mark.skip(
    reason="this way of instrumenting has been broken with the map serial protocol"
)
async def test_backend_model_warmup(
    llm_model_name: str, services: svcs.Container
):
    url = "http://127.0.0.1:8001"

    await backend_connection_is_up(url)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.patch(
            f"{url}/manager/manifest", json={"models": [], "serial": ["1", 1]}
        )
        assert resp.status_code == 202

    @wait_for_condition()
    async def wait_for_model_unloaded() -> bool:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{url}/models/{llm_model_name}")
            if resp.status_code != 200:
                return False
            return "active" not in resp.json()["status"]

    await wait_for_model_unloaded()

    model_config = ModelInstanceConfig(
        id=llm_model_name,
        instances=1,
        memory={"ram": parse_size("1.3G")},
        task="chat",
    )
    backend = skvaider.proxy.backends.SkvaiderBackend(url)
    pool = skvaider.proxy.pool.Pool([model_config], [backend])

    @wait_for_condition()
    async def wait_for_model_loaded() -> bool:
        return pool.count_loaded_instances(llm_model_name) > 0

    await wait_for_model_loaded()

    model_backends = [b for b in pool.backends if llm_model_name in b.models]
    models_in_backends = [b.models[llm_model_name] for b in model_backends]
    loaded = [m for m in models_in_backends if m.is_loaded]
    assert loaded
    assert loaded[0].limit == 21
