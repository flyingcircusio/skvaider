from multiprocessing import Process

import httpx
import pytest
import svcs
import uvicorn
from sqlalchemy import delete

from skvaider import app_factory
from skvaider.db import DBSession, DBSessionManager
from skvaider.models import AuthToken


def run_server():
    uvicorn.run(app_factory, factory=True)


@pytest.fixture
def services():
    reg = svcs.Registry()
    with svcs.Container(reg) as container:
        yield container


@pytest.fixture
async def database(services):
    sessionmanager = DBSessionManager(
        "postgresql+psycopg://skvaider:foobar@localhost:5432/skvaider"
    )
    async with sessionmanager.session() as session:
        services.register_local_value(DBSession, session)
        await session.execute(delete(AuthToken))
        yield session

    await sessionmanager.close()


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
