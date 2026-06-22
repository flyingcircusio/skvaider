from skvaider.inference.conftest import TestAPI
from skvaider.inference.manager import Manager
from skvaider.inference.model import Model
from skvaider.proxy.backends import BackendHealthRequest


async def test_health(test_api: TestAPI, gemma: Model):
    health = await test_api(BackendHealthRequest())
    models = {m.id: m for m in health.models}
    assert models["gemma"].status == {"inactive", "stopped"}

    await gemma.start()

    # DummyModel becomes healthy immediately; real llama-server would
    # have an intermediate "inactive, running" phase.
    health = await test_api(BackendHealthRequest())
    models = {m.id: m for m in health.models}
    assert models["gemma"].status == {"active", "healthy", "running"}


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
