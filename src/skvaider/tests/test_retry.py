from typing import Callable

import pytest
import svcs
from fastapi import HTTPException

from skvaider.config import ModelInstanceConfig
from skvaider.conftest import registered_model_factory
from skvaider.proxy.backends import DummyBackend
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy
from skvaider.utils import TaskManager


async def test_proxy_retry_all_fail(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory,  # type: ignore[misc]
) -> None:
    dummy_backend.fail_count = 100
    req = mock_request_factory(stream=False)  # type: ignore[misc]

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(req, "/test")  # type: ignore[misc]

    assert exc.value.status_code == 503


async def test_proxy_retry_verifies_backend_switching(
    svcs_registry: svcs.Registry,
    services: svcs.Container,
    task_managers: list[TaskManager],
    mock_request_factory,  # type: ignore[misc]
    dummy_backend_factory: Callable[..., DummyBackend],
) -> None:
    """First backend returns 540, proxy retries on the second and succeeds.

    Routing is deterministic: ``ModelSemaphore.acquire`` picks the candidate
    with the lowest ``in_progress`` count, breaking ties by list order.
    b1 is appended first so it is always tried first.
    """
    b1 = dummy_backend_factory(fail_count=1)
    b2 = dummy_backend_factory()

    config = ModelInstanceConfig(
        id="test-model", instances=2, memory={"ram": 10}, task="chat"
    )
    pool = Pool([config], [b1, b2])
    task_managers.append(pool.tasks)

    for b in [b1, b2]:
        registered_model_factory(
            "test-model", b, ram=10, limit=100, loaded=True
        )

    await pool.rebalance()
    svcs_registry.register_value(  # pyright: ignore[reportUnknownMemberType]
        Pool, pool
    )

    proxy = OpenAIProxy(services)
    req = mock_request_factory(stream=False)  # type: ignore[misc]
    await proxy.proxy(req, "/test")  # type: ignore[misc]

    # b1 was tried first (list order), failed with 540, then b2 succeeded.
    assert b1.call_count == 1, (
        "b1 should have been called exactly once (and failed)"
    )
    assert b2.call_count == 1, "b2 should have handled the retry"
