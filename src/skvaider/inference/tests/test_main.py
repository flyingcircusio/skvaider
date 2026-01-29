from fastapi.testclient import TestClient

from skvaider.conftest import wait_for_condition
from skvaider.inference.manager import Model


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
