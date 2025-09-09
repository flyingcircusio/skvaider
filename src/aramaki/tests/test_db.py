import pytest

from aramaki.db import DBSessionManager, SessionManagerClosed


async def test_session_manager(tmpdir):
    manager = DBSessionManager(tmpdir)
    manager.upgrade()

    assert manager._engine is not None
    assert manager._sessionmaker is not None

    await manager.close()

    assert manager._engine is None
    assert manager._sessionmaker is None

    # idempotent

    await manager.close()

    assert manager._engine is None
    assert manager._sessionmaker is None


async def test_session_manager_context_manager(tmpdir):
    manager = DBSessionManager(tmpdir)
    manager.upgrade()

    async with manager.session():
        pass

    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        async with manager.session():
            raise CustomError()

    await manager.close()

    with pytest.raises(SessionManagerClosed):
        async with manager.session():
            pass
