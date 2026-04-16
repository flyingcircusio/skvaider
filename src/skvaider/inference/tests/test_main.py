from skvaider.conftest import wait_for_condition
from skvaider.inference.conftest import TestAPI
from skvaider.inference.manager import Manager
from skvaider.inference.model import Model
from skvaider.proxy.backends import BackendHealthRequest


async def test_health(test_api: TestAPI, gemma: Model):
    health = await test_api(BackendHealthRequest())
    models = {m.id: m for m in health.models}
    assert models["gemma"].status == {"inactive", "stopped"}

    await gemma.start()

    # the model should be started but health check not completed yet.
    # this might be flaky.
    health = await test_api(BackendHealthRequest())
    models = {m.id: m for m in health.models}
    assert models["gemma"].status == {"inactive", "running"}

    @wait_for_condition()
    async def healthy_model():
        health = await test_api(BackendHealthRequest())
        models = {m.id: m for m in health.models}
        assert models["gemma"].status == {"active", "healthy", "running"}
        return True

    await healthy_model()


async def test_usage_returns_ram_structure(
    test_api: TestAPI, manager: Manager
) -> None:
    health = await test_api(BackendHealthRequest())
    assert "ram" in health.usage
    ram = health.usage["ram"]
    assert set(ram.keys()) == {"total", "used", "free"}
    assert all(isinstance(v, int) for v in ram.values())


async def test_usage_reflects_monitor_values(
    test_api: TestAPI, manager: Manager
) -> None:
    monitor = manager.monitors["ram"]
    monitor.total = 8 * 1024**3
    monitor.used = 2 * 1024**3
    monitor.free = 6 * 1024**3

    health = await test_api(BackendHealthRequest())
    ram = health.usage["ram"]
    assert ram["total"] == 8 * 1024**3
    assert ram["used"] == 2 * 1024**3
    assert ram["free"] == 6 * 1024**3
