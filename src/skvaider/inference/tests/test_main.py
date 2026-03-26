from fastapi.testclient import TestClient

from skvaider.conftest import wait_for_condition
from skvaider.inference.manager import Manager
from skvaider.inference.model import Model


async def test_health(client: TestClient, gemma: Model):
    response = client.get("/manager/health")
    assert response.json() == {
        "models": {
            "gemma": {
                "status": [
                    "inactive",
                    "stopped",
                ],
            },
        },
        "status": "ok",
    }
    assert response.status_code == 200

    await gemma.start()

    response = client.get("/manager/health")
    # the model should be started but health check not completed yet.
    # this might be flaky.
    assert response.json() == {
        "models": {
            "gemma": {
                "status": [
                    "inactive",
                    "running",
                ],
            },
        },
        "status": "ok",
    }
    assert response.status_code == 200

    @wait_for_condition()
    async def healthy_model():
        response = client.get("/manager/health")
        # the model should be started but health check not completed yet.
        # this might be flaky.
        assert response.status_code == 200
        assert response.json() == {
            "models": {
                "gemma": {
                    "status": [
                        "active",
                        "healthy",
                        "running",
                    ],
                },
            },
            "status": "ok",
        }
        return True

    await healthy_model()


async def test_usage_returns_ram_structure(
    client: TestClient, manager: Manager
) -> None:
    response = client.get("/manager/usage")
    assert response.status_code == 200
    body = response.json()
    assert "memory" in body
    assert "ram" in body["memory"]
    ram = body["memory"]["ram"]
    assert set(ram.keys()) == {"total", "used", "free"}
    assert all(isinstance(v, int) for v in ram.values())


async def test_usage_reflects_monitor_values(
    client: TestClient, manager: Manager
) -> None:
    monitor = manager.monitors["ram"]
    monitor.total = 8 * 1024**3
    monitor.used = 2 * 1024**3
    monitor.free = 6 * 1024**3

    response = client.get("/manager/usage")
    assert response.status_code == 200
    ram = response.json()["memory"]["ram"]
    assert ram["total"] == 8 * 1024**3
    assert ram["used"] == 2 * 1024**3
    assert ram["free"] == 6 * 1024**3
