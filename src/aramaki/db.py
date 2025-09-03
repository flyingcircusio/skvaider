import contextlib
from typing import AsyncIterator

import structlog.stdlib
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DBSession = AsyncSession  # Alias for easier referencing
log = structlog.stdlib.get_logger()


class SessionManagerClosed(Exception):
    pass


class Base(DeclarativeBase):
    @classmethod
    async def create(cls, db: AsyncSession, **kwargs):
        transaction = cls(**kwargs)
        db.add(transaction)
        return transaction

    def delete(self, db: AsyncSession):
        return db.delete(self)


class DBSessionManager:
    state_directory: str

    def __init__(self, state_directory: str):
        self.state_directory = state_directory
        db_url = f"sqlite+aiosqlite:///{self.state_directory}/aramaki.sqlite3"
        self._engine = create_async_engine(db_url)
        self._sessionmaker = async_sessionmaker(
            autocommit=False, bind=self._engine, expire_on_commit=False
        )

    def upgrade(self):
        log.info("Upgrading aramaki database")
        config = Config()
        db_url = f"sqlite:///{self.state_directory}/aramaki.sqlite3"
        from sqlalchemy import create_engine

        config.set_main_option("script_location", "aramaki:alembic")
        config.set_main_option("sqlalchemy.url", db_url)
        engine = create_engine(db_url)
        with engine.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    async def close(self):
        if self._engine is None:
            return
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise SessionManagerClosed()

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
        finally:
            await session.close()
