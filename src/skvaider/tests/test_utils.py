import asyncio
from asyncio import TimeoutError

import pytest

from skvaider.conftest import wait_for_condition
from skvaider.utils import TaskManager, slugify


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


# slugify


def test_slugify():
    assert slugify("Hello World") == "hello-world"
    assert slugify("café") == "cafe"
    assert slugify("a:b_c/d") == "a-b-c-d"
    # ! and @ are not separators — removed without inserting a hyphen
    assert slugify("a!b@c") == "abc"
    assert slugify("a  b") == "a-b"
    assert slugify("--hello--") == "hello"
    # Only unsafe chars → stripped to empty → falls back to "unnamed"
    assert slugify("!!!") == "unnamed"
    # input_len=16, overflow=8 → text[:4] + text[12:] = "abcd" + "mnop"
    assert slugify("abcdefghijklmnop", max_length=8) == "abcdmnop"


# TaskManager


async def test_task_manager_count():
    tm = TaskManager()
    tm.create(lambda: asyncio.sleep(1000))
    assert tm.count == 1
    tm.terminate()


async def test_task_manager_create_runs_coroutine():
    event = asyncio.Event()
    tm = TaskManager()

    async def setter():
        event.set()

    tm.create(setter)
    await asyncio.wait_for(event.wait(), timeout=1.0)
    assert event.is_set()
    tm.terminate()


async def test_task_manager_unique_task_returns_same_object():
    started = asyncio.Event()
    call_count = 0
    tm = TaskManager()

    async def counted():
        nonlocal call_count
        call_count += 1
        started.set()
        await asyncio.Event().wait()

    task1 = tm.create(counted, id="x")
    task2 = tm.create(counted, id="x")
    assert task1 is task2
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert call_count == 1
    tm.terminate()


async def test_task_manager_unique_task_removed_from_map_on_completion():
    tm = TaskManager()

    async def quick():
        pass

    task = tm.create(quick, id="y")
    await task
    assert "y" not in tm.unique_task_map


async def test_task_manager_done_task_removed_from_list():
    tm = TaskManager()

    async def quick():
        pass

    task = tm.create(quick)
    await task
    assert tm.count == 0


async def test_task_manager_cancel_by_id():
    tm = TaskManager()
    task = tm.create(asyncio.Event().wait, id="z")
    tm.cancel("z")
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    tm.terminate()


async def test_task_manager_terminate_cancels_all():
    tm = TaskManager()
    tm.create(asyncio.sleep, args=(1000,))
    tm.create(asyncio.sleep, args=(1000,))
    assert tm.count == 2
    tm.terminate()
    assert tm.count == 0


async def test_task_manager_poll_calls_function_repeatedly():
    counter = 0
    done = asyncio.Event()
    tm = TaskManager()

    async def increment():
        nonlocal counter
        counter += 1
        if counter >= 2:
            done.set()

    tm.poll(increment, interval=0.001)
    await asyncio.wait_for(done.wait(), timeout=1.0)
    assert counter >= 2
    tm.terminate()
