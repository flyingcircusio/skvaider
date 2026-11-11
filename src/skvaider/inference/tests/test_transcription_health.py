import http.server
import threading
from collections.abc import Generator
from typing import Any

import pytest

from skvaider.inference.config import VllmModelConfig
from skvaider.inference.model import VllmModel


class TranscriptionServer:
    """Minimal stand-in for vLLM's /v1/audio/transcriptions endpoint."""

    status = 200
    body = b'{"text": ""}'
    last_content_type: str = ""
    last_body: bytes = b""

    def __init__(self) -> None:
        handler = type("Handler", (_Handler,), {"server_state": self})
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.thread.join()


class _Handler(http.server.BaseHTTPRequestHandler):
    server_state: TranscriptionServer

    def do_POST(self) -> None:
        state = self.server_state
        length = int(self.headers.get("content-length", 0))
        state.last_content_type = self.headers.get("content-type", "")
        state.last_body = self.rfile.read(length)
        self.send_response(state.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(state.body)))
        self.end_headers()
        self.wfile.write(state.body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def transcription_server() -> Generator[TranscriptionServer, None, None]:
    server = TranscriptionServer()
    try:
        yield server
    finally:
        server.stop()


def transcription_model(server: TranscriptionServer) -> VllmModel:
    config = VllmModelConfig(
        id="qwen3-asr",
        task="transcription",
        repo="Qwen/Qwen3-ASR-1.7B",
        revision="main",
        port=server.port,
        context_size="auto",
    )
    model = VllmModel(config, lambda: None)
    model.endpoint = server.endpoint
    return model


async def test_transcription_health_check_sends_audio(
    transcription_server: TranscriptionServer,
):
    model = transcription_model(transcription_server)

    checks = await model._check_health()  # pyright: ignore[reportPrivateUsage]

    assert checks == {"transcription": ""}
    assert transcription_server.last_content_type.startswith(
        "multipart/form-data; boundary="
    )
    body = transcription_server.last_body
    assert b'name="model"' in body
    assert b"qwen3-asr" in body
    # A real WAV file is uploaded, not an empty placeholder.
    assert b"RIFF" in body
    assert b'filename="health.wav"' in body


async def test_transcription_health_check_reports_http_error(
    transcription_server: TranscriptionServer,
):
    transcription_server.status = 500
    model = transcription_model(transcription_server)

    checks = await model._check_health()  # pyright: ignore[reportPrivateUsage]

    assert checks == {"transcription": "HTTP 500"}


async def test_transcription_health_check_reports_missing_text(
    transcription_server: TranscriptionServer,
):
    transcription_server.body = b"{}"
    model = transcription_model(transcription_server)

    checks = await model._check_health()  # pyright: ignore[reportPrivateUsage]

    assert checks == {"transcription": "response has no text"}
