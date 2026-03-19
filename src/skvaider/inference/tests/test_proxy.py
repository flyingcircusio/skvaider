from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from skvaider.inference.manager import Manager, Model


async def test_proxy_returns_540_when_model_unavailable(
    client: TestClient, manager: Manager
):
    """
    Test that proxy endpoint returns 540 status code when model is unavailable.

    This happens when use_model returns None, which occurs when the model
    is not in an 'active' state.
    """
    # Mock use_model to return None, simulating an unavailable model
    with patch.object(
        manager, "use_model", new_callable=AsyncMock
    ) as mock_use_model:
        mock_use_model.return_value = None

        response = client.get("/models/gemma/proxy/health")
        assert response.status_code == 540
        assert response.json() == {"detail": "Model unavailable"}
        mock_use_model.assert_called_once_with("gemma")


async def test_proxy_returns_540_when_model_inactive(
    client: TestClient, gemma: Model
):
    """
    Test that proxy endpoint returns 540 when model exists but is not active.

    This is a scenario where the model is in the manager but
    not yet started or has been stopped.
    """
    # Ensure the model is not started (should be in inactive state by default)
    assert "active" not in gemma.status

    response = client.get("/models/gemma/proxy/health")
    assert response.status_code == 540
    assert response.json() == {"detail": "Model unavailable"}
