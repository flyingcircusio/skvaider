from asyncio import TimeoutError

import pytest

from skvaider.conftest import wait_for_condition


async def test_wait_for_condition_assertion_timeout():
    @wait_for_condition(interval=0.1, timeout=0.5)
    async def retry_until_ready():
        assert 2 == 3
        return True

    with pytest.raises(AssertionError):
        await retry_until_ready()


async def test_wait_for_condition_assertion():
    count = 0

    @wait_for_condition(interval=0.1, timeout=0.5)
    async def retry_until_ready():
        nonlocal count
        count += 1
        assert count == 3
        return True

    await retry_until_ready()


async def test_wait_for_condition_simple_retry():
    results = [False, False, True]

    @wait_for_condition()
    async def retry_until_ready():
        return results.pop(0)

    await retry_until_ready()


async def test_wait_for_condition_timeout():
    results = [False, False, False, False, True]

    @wait_for_condition(interval=0.1, timeout=0.3)
    async def retry_until_ready():
        return results.pop(0)

    with pytest.raises(TimeoutError):
        await retry_until_ready()
