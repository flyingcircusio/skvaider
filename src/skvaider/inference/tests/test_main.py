from fastapi.testclient import TestClient

from skvaider.inference.manager import Model


async def test_health(client: TestClient, gemma: Model):
    response = client.get("/manager/health")
    assert response.status_code == 200
    assert response.json() == {
        "models": {
            "gemma": {
                "healthy": False,
                "status": "stopped",
            },
        },
        "status": "ok",
    }

    await gemma.start()

    response = client.get("/manager/health")
    # the model is started but not health checked, yet.
    assert response.status_code == 503
    assert response.json() == {
        "models": {
            "gemma": {
                "healthy": False,
                "status": "running",
            },
        },
        "status": "error",
    }
