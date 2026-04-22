from pathlib import Path

from fastapi.testclient import TestClient


def test_no_debug_by_default(
    tmp_path: Path, client: TestClient, auth_header: None, llm_model_name: str
):
    response = client.post(
        "/openai/v1/completions",
        json={"model": "gemma"},
    )
    request_id = response.headers["x-skvaider-request-id"]
    assert not (tmp_path / "debug" / f"{request_id}.request").exists()
    assert not (tmp_path / "debug" / f"{request_id}.response").exists()


def test_debug_via_header(
    tmp_path: Path, client: TestClient, auth_header: None, llm_model_name: str
):
    response = client.post(
        "/openai/v1/completions",
        headers={"x-skvaider-debug": "yes"},
        json={"model": "gemma"},
    )
    request_id = response.headers["x-skvaider-request-id"]
    assert (tmp_path / "debug" / f"{request_id}.request").exists()
    assert (tmp_path / "debug" / f"{request_id}.response").exists()


def test_no_debug_if_unauthenticated(
    tmp_path: Path, client: TestClient, llm_model_name: str
):
    response = client.post(
        "/openai/v1/completions",
        headers={"x-skvaider-debug": "yes"},
        json={"model": "gemma"},
    )
    assert response.status_code == 403
    request_id = response.headers["x-skvaider-request-id"]
    assert not (tmp_path / "debug" / f"{request_id}.request").exists()
    assert not (tmp_path / "debug" / f"{request_id}.response").exists()
    assert not (tmp_path / "debug" / f"{request_id}.response").exists()
