import http.server
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Generator

import pytest
import svcs
import svcs.fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skvaider.inference import app_factory
from skvaider.inference.config import ModelConfig, ModelFile
from skvaider.inference.manager import Manager, Model


@pytest.fixture
def services() -> Generator[svcs.Container, None, None]:
    reg = svcs.Registry()
    with svcs.Container(reg) as container:
        yield container


@pytest.fixture
def client(manager: Manager, gemma: Model) -> Generator[TestClient, None, None]:
    @svcs.fastapi.lifespan
    async def test_lifespan(
        app: FastAPI, registry: svcs.Registry
    ) -> AsyncGenerator[None, None]:
        registry.register_value(  # pyright: ignore[reportUnknownMemberType]
            Manager, manager
        )

        yield

    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    p = tmp_path / "models"
    p.mkdir()
    return p


@pytest.fixture
def models_cache() -> Path:
    # This is on purpose not in a tmp_path as we want to cache this
    # over multiple runs.
    cache_dir = Path(".models").absolute()
    if not cache_dir.exists():
        cache_dir.mkdir()
    return cache_dir


@pytest.fixture
async def manager(model_path: Path) -> AsyncGenerator[Manager, None]:
    from skvaider.inference.config import ManagerConfig

    config = ManagerConfig(backend="cpu")
    m = Manager(model_path, config)
    yield m
    await m.shutdown()


async def prepare_model(
    id: str,
    context: int,
    args: list[str],
    file: ModelFile,
    models_cache: Path,
    manager: Manager,
) -> Model:
    config = ModelConfig(
        id=id,
        files=[file],
        context_size=context,
        cmd_args=args,
    )

    model = Model(config)
    manager.add_model(model)

    cache_dir = models_cache / model.slug
    if cache_dir.exists() and cache_dir.is_file():
        cache_dir.unlink()
    if not cache_dir.exists():
        await model.download()
        cache_dir.mkdir()
        for f in model.model_files:
            # f.rename(cache_dir / f.name)
            shutil.move(
                f, cache_dir / f.name
            )  # cannot move accross filesystems, for example in github actions runner
    # We had data in the cache. The download method is unaware of the test-fixture
    # caching. Maybe there could be a real world use case to make download() smarter.
    # The test harness needs to ensure that we restore the data from the cache as expected.
    for f in model.model_files:
        f.symlink_to(cache_dir / f.name)
    model.integrity_marker_file.touch()

    return model


@pytest.fixture
async def gemma(models_cache: Path, manager: Manager) -> Model:
    return await prepare_model(
        "gemma",
        4096,
        [],
        ModelFile(
            url="https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/c90975dbd40c0c7b275fefaae758c3415c906238/gemma-3-270m-it-UD-Q4_K_XL.gguf?download=true",
            hash="e5420636e0cbfee24051ff22e9719380a3a93207a472edb18dd0c89a95f6ef80",
        ),
        models_cache,
        manager,
    )


@pytest.fixture
async def embeddinggemma(models_cache: Path, manager: Manager) -> Model:
    return await prepare_model(
        "embeddinggemma",
        4096,
        ["--embeddings", "-ngl", "0"],
        ModelFile(
            url="https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-F32.gguf",
            hash="a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57",
        ),
        models_cache,
        manager,
    )


@pytest.fixture
def gguf_http_server() -> Generator[str, None, None]:
    """Serve static GGUF fixture files via HTTP."""
    server_address = ("localhost", 0)  # let the OS pick an available port
    MY_FILE_DIR = Path(__file__).parent.resolve()

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(
                *args,
                directory=str(
                    MY_FILE_DIR
                    / "tests"
                    / "fixtures"
                    / "gguf_http_server_files"
                ),
                **kwargs,
            )

    httpd = http.server.HTTPServer(server_address, CustomHandler)
    port = httpd.server_address[1]

    import threading

    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://localhost:{port}"

    httpd.shutdown()
    thread.join()
