import svcs
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncTransaction

from skvaider.db import DBSession
from skvaider.models import AuthToken


class AITokenReplicator:
    def __init__(self, registry: svcs.Registry):
        self.registry = registry
        self.full_sync_connection: AsyncConnection | None = None
        self.full_sync_transaction: AsyncTransaction | None = None
        self.full_sync_session: DBSession | None = None

    async def update(self, msg: dict) -> None:
        with svcs.Container(self.registry) as container:
            if self.full_sync_session is not None:
                db_session = self.full_sync_session
            else:
                db_session = await container.aget(DBSession)
            token_obj = await db_session.get(AuthToken, msg["record_id"])
            match msg["change"]:
                case "update":
                    if token_obj is None:
                        await AuthToken.create(
                            db_session,
                            id=msg["record_id"],
                            resource_group=msg["data"]["resource_group"],
                            secret_hash=msg["data"]["secret_hash"],
                        )
                case "delete":
                    if token_obj is not None:
                        await db_session.delete(token_obj)

    async def start_full_sync(self, msg: dict) -> None:
        with svcs.Container(self.registry) as container:
            db_session = await container.aget(DBSession)
            self.full_sync_connection = await db_session.connection()
            self.full_sync_transaction = self.full_sync_connection.begin()
            self.full_sync_session = DBSession(
                bind=self.full_sync_connection,
                join_transaction_mode="create_savepoint",
            )

    async def end_full_sync(self, msg: dict) -> None:
        await self.full_sync_transaction.commit()
        await self.full_sync_transaction.close()
        self.full_sync_connection = self.full_sync_transaction = (
            self.full_sync_session
        ) = None
