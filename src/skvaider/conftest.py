from multiprocessing import Process

import httpx
import pytest
import uvicorn

from skvaider import app_factory


def run_server():
    uvicorn.run(app_factory, factory=True)


@pytest.fixture
def server():
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start()
    while True:
        try:
            httpx.get("http://localhost:8000/")
        except httpx.ConnectError:
            continue
        break
    yield
    proc.kill()  # Cleanup after test
