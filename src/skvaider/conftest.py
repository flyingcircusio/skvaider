import os

import pytest
import svcs
from fastapi.testclient import TestClient
from sqlalchemy import delete

from skvaider import app_factory
from skvaider.db import DBSession, DBSessionManager
from skvaider.models import AuthToken

TEST_SERVER_PORT = 8001

CONFIG_TEMPLATE = """
# Sample config used by the tests.

[database]
url = "{db_url}"

[[backend]]
type = "openai"
url = "http://127.0.0.1:11435"
"""


@pytest.fixture
def services():
    reg = svcs.Registry()
    with svcs.Container(reg) as container:
        yield container


@pytest.fixture
def db_url():
    return "postgresql+psycopg://skvaider:foobar@localhost:5432/test"


@pytest.fixture
def skvaider_url():
    return f"http://localhost:{TEST_SERVER_PORT}"


async def cleanup_db(session):
    await session.execute(delete(AuthToken))
    await session.commit()


@pytest.fixture(autouse=True)
async def session(services, db_url):  # autouse to always ensure a clean DB
    sessionmanager = DBSessionManager(db_url)
    async with sessionmanager.session() as session:
        services.register_local_value(DBSession, session)
        await cleanup_db(session)
        yield session
        await cleanup_db(session)
    await sessionmanager.close()


@pytest.fixture
async def auth_token(session):
    """Return a valid auth token."""
    token = await AuthToken.create(
        session, username="user", password="password"
    )
    await session.commit()  # make this visible to other sessions, too.
    yield f"{token.username}-{token.password}"


@pytest.fixture
async def auth_header(client, auth_token):
    """Inject a valid auth header into all client requests."""
    header = {"Authorization": f"Bearer {auth_token}"}
    client.headers.update(header)
    yield


@pytest.fixture
def client(skvaider_url, db_url, tmp_path):
    cfg_path = tmp_path / "config.test.toml"
    with cfg_path.open("w") as f:
        f.write(CONFIG_TEMPLATE.format(db_url=db_url))
        os.environ["SKVAIDER_CONFIG_FILE"] = str(cfg_path)
    with TestClient(app_factory()) as client:
        yield client
