import base64
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from pathlib import Path

import svcs
import svcs.fastapi
from fastapi.testclient import TestClient

from skvaider import app_factory
from skvaider.auth import AuthTokens
from skvaider.config import (
    AuthConfig,
    Config,
    LoggingConfig,
    ModelInstanceConfig,
    ServerConfig,
)
from skvaider.conftest import DUMMY_TOKENS, hasher
from skvaider.multipart import text_fields
from skvaider.proxy.backends import DummyBackend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool


def transcription_model_config(model_id: str) -> ModelInstanceConfig:
    return ModelInstanceConfig(
        id=model_id,
        instances=1,
        memory={"ram": 100},
        task="transcription",
    )


def auth_header() -> dict[str, str]:
    DUMMY_TOKENS.data["user"] = {"secret_hash": hasher.hash("asdf")}
    token = base64.b64encode(
        json.dumps({"id": "user", "secret": "asdf"}).encode()
    ).decode()
    return {"Authorization": f"Bearer {token}"}


@contextmanager
def transcription_client(
    tmp_path: Path,
) -> Generator[tuple[TestClient, DummyBackend], None, None]:
    backend = DummyBackend("http://backend")
    backend.healthy = True
    backend.map_up.mark("up")
    backend.memory = {"ram": {"free": 1000, "total": 1000}}

    @svcs.fastapi.lifespan
    async def lifespan(
        app: object, registry: svcs.Registry
    ) -> AsyncGenerator[None, None]:
        pool = Pool(
            [
                transcription_model_config("qwen3-asr"),
                ModelInstanceConfig(
                    id="chat-model",
                    instances=1,
                    memory={"ram": 100},
                    task="chat",
                ),
            ],
            [backend],
        )
        for model_id in ("qwen3-asr", "chat-model"):
            model = AIModel(id=model_id, owned_by="test", backend=backend)
            model.is_loaded = True
            backend.models[model_id] = model
        registry.register_value(  # pyright: ignore[reportUnknownMemberType]
            Pool, pool
        )
        registry.register_factory(  # pyright: ignore[reportUnknownMemberType]
            AuthTokens,
            DUMMY_TOKENS.get_collection_with_session,
            enter=False,
        )
        try:
            yield
        finally:
            pool.close()

    config = Config(
        auth=AuthConfig(admin_tokens=["asdf"]),
        server=ServerConfig(directory=tmp_path),
        backend=[],
        models=[],
        logging=LoggingConfig(),
    )
    with TestClient(app_factory(config, lifespan=lifespan)) as client:
        yield client, backend


def test_text_fields_extracts_routing_fields_only():
    body = (
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="model"\r\n\r\n'
        b"qwen3-asr\r\n"
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="stream"\r\n\r\n'
        b"true\r\n"
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
        b"\xff\xfeRIFF\x00\x01\r\n"
        b"--BOUND--\r\n"
    )
    content_type = "multipart/form-data; boundary=BOUND"

    assert text_fields(body, content_type) == {
        "model": "qwen3-asr",
        "stream": "true",
    }
    # Not multipart, or malformed: routing fails later with a clear error.
    assert text_fields(b'{"model": "m"}', "application/json") == {}
    assert text_fields(b"garbage", content_type) == {}


def test_transcription_requires_model(tmp_path: Path):
    with transcription_client(tmp_path) as (client, backend):
        response = client.post(
            "/openai/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
            headers=auth_header(),
        )
    assert response.status_code == 400
    assert "model" in response.json()["detail"]
    assert backend.last_multipart_body == b""


def test_transcription_rejects_unknown_model(tmp_path: Path):
    with transcription_client(tmp_path) as (client, backend):
        response = client.post(
            "/openai/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
            data={"model": "nope"},
            headers=auth_header(),
        )
    assert response.status_code == 400
    assert "not known" in response.json()["detail"]
    assert backend.last_multipart_body == b""


def test_transcription_rejects_chat_model(tmp_path: Path):
    with transcription_client(tmp_path) as (client, backend):
        response = client.post(
            "/openai/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
            data={"model": "chat-model"},
            headers=auth_header(),
        )
    assert response.status_code == 400
    assert "not a transcription model" in response.json()["detail"]
    assert backend.last_multipart_body == b""


def test_transcription_forwards_multipart_body_unchanged(tmp_path: Path):
    audio = b"RIFF\x00\x01\x02\x03-not-really-a-wav"
    with transcription_client(tmp_path) as (client, backend):
        response = client.post(
            "/openai/v1/audio/transcriptions",
            files={"file": ("meeting.wav", audio, "audio/wav")},
            data={
                "model": "QWEN3-ASR",
                "language": "de",
                "response_format": "verbose_json",
            },
            headers=auth_header(),
        )

    # The backend response is passed through with its status and media type.
    assert response.status_code == 200
    assert response.json() == {"text": "dummy transcription"}
    assert response.headers["content-type"].startswith("application/json")

    # The body is untouched: the backend parses the upload, not the proxy.
    assert backend.last_multipart_model == "qwen3-asr"
    assert backend.last_multipart_content_type.startswith(
        "multipart/form-data; boundary="
    )
    assert audio in backend.last_multipart_body
    assert b'name="language"' in backend.last_multipart_body
    assert b"de" in backend.last_multipart_body


def test_transcription_relayed_streaming(tmp_path: Path):
    audio = b"RIFF\x00\x01\x02\x03-not-really-a-wav"
    with (
        transcription_client(tmp_path) as (client, backend),
        client.stream(
            "POST",
            "/openai/v1/audio/transcriptions",
            files={"file": ("meeting.wav", audio, "audio/wav")},
            data={"model": "qwen3-asr", "stream": "true"},
            headers=auth_header(),
        ) as response,
    ):
        chunks = list(response.iter_bytes(chunk_size=16))

    # The streaming path was taken: the backend's SSE events are relayed
    # verbatim, not buffered into a JSON body.
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"".join(chunks) == b'data: {"text": "dummy"}\n\ndata: [DONE]\n\n'

    # Routing still worked: the model was normalized and the upload
    # reached the backend.
    assert backend.last_multipart_model == "qwen3-asr"
    assert audio in backend.last_multipart_body
