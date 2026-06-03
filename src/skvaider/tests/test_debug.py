from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from skvaider.debug import BodyBuffer, DebugRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> DebugRecorder:
    request = MagicMock()

    def get(key: str, default: Any = None) -> Any:
        return default

    request.headers.get.side_effect = get
    return DebugRecorder(
        request=request,
        directory=tmp_path,
        receive=AsyncMock(),
        send=AsyncMock(),
    )


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


# BodyBuffer unit tests (use tiny max_bytes so tests stay fast and readable)


def test_body_buffer_small_body_unchanged():
    buf = BodyBuffer(max_bytes=20)
    buf.ingest(b"hello world")
    assert buf.data == b"hello world"


def test_body_buffer_at_limit_unchanged():
    buf = BodyBuffer(max_bytes=10)
    buf.ingest(b"1234567890")
    assert buf.data == b"1234567890"


def test_body_buffer_multi_chunk_under_limit_unchanged():
    # Two chunks that straddle the head/tail boundary must still be byte-identical.
    buf = BodyBuffer(max_bytes=10)  # head=5, tail=5
    buf.ingest(b"12345")
    buf.ingest(b"6789")  # total 9 bytes — just under limit
    assert buf.data == b"123456789"


def test_body_buffer_over_limit_cuts_from_middle():
    buf = BodyBuffer(max_bytes=10)  # head=5, tail=5
    data = b"HHHHH" + b"M" * 6 + b"TTTTT"  # 16 bytes; 6 in the middle are lost
    buf.ingest(data)
    result = buf.data
    assert result[:5] == b"HHHHH"
    assert result[-5:] == b"TTTTT"
    assert b"omitted" in result
    assert b"M" not in result


def test_body_buffer_over_limit_multi_chunk():
    buf = BodyBuffer(max_bytes=10)  # head=5, tail=5
    buf.ingest(b"HHHHH")  # fills head
    buf.ingest(b"MMMM")  # goes to tail
    buf.ingest(b"TTTTT")  # pushes out the M bytes
    result = buf.data
    assert result[:5] == b"HHHHH"
    assert result[-5:] == b"TTTTT"
    assert b"omitted" in result


# DebugRecorder integration tests — only public API, no private attribute access


async def test_recorder_request_body_captured(recorder: DebugRecorder):
    data = b"the request body"
    recorder._orig_receive = AsyncMock(
        return_value={"type": "http.request", "body": data}
    )  #
    await recorder.capture_request()
    assert recorder.captured_request_body == data


async def test_recorder_response_body_captured(recorder: DebugRecorder):
    data = b"the response body"
    await recorder.capture_response(
        {"type": "http.response.start", "status": 200, "headers": []}
    )
    await recorder.capture_response(
        {"type": "http.response.body", "body": data}
    )
    assert recorder.captured_response_body == data
