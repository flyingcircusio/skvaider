import pytest

from skvaider.db import DBSessionManager, SessionManagerClosed
from skvaider.models import AuthToken


async def test_close_no_engine(db_url):
    manager = DBSessionManager(db_url)
    await manager.close()
    await manager.close()


async def test_no_session_after_close(db_url):
    manager = DBSessionManager(db_url)
    await manager.close()
    with pytest.raises(SessionManagerClosed):
        async with manager.session():
            pass  # pragma: no cover


async def test_session_rolls_back_on_error(db_url):
    manager = DBSessionManager(db_url)
    async with manager.session() as session:
        token = await session.get(AuthToken, "user")
        assert token is None

    orig_e = None
    try:
        async with manager.session() as session:
            await AuthToken.create(session, username="user", password="*")
            orig_e = Exception()
            raise orig_e
    except Exception as e:
        assert e is orig_e

    async with manager.session() as session:
        token = await session.get(AuthToken, "user")
        assert token is None
