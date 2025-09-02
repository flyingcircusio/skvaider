import asyncio
import json
from typing import TYPE_CHECKING

import sqlalchemy
from sqlalchemy import UniqueConstraint, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from aramaki.db import Base

if TYPE_CHECKING:
    from aramaki import Manager


class CollectionReplicationStatus(Base):
    __tablename__ = "collection_replication_status"
    __table_args__ = (
        UniqueConstraint(
            "collection",
            "partition",
            name="collection_partition_unique",
        ),
    )

    collection: Mapped[str] = mapped_column(primary_key=True)
    partition: Mapped[str] = mapped_column(primary_key=True)
    record_id: Mapped[str] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column()
    data: Mapped[bytes] = mapped_column()


class Collection:
    """Read-only access to a collection."""

    collection: str  # The identifier for the collection in Aramaki

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, key: str) -> dict | None:
        result = (
            (
                await self.session.execute(
                    select(CollectionReplicationStatus).filter_by(
                        collection=self.collection, record_id=key
                    )
                )
            )
            .scalars()
            .one_or_none()
        )
        if not result:
            return
        return json.loads(result.data)

    async def keys(self) -> list[str]:
        result = []
        result = result = (
            (
                await self.session.execute(
                    select(CollectionReplicationStatus).filter_by(
                        collection=self.collection,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [x.record_id for x in result]

    async def currently_known_partition_and_version(
        self,
    ) -> tuple[str | None, int]:
        # Use the fact that (collection, partition) is unique for the client view here.
        maybe_result = (
            await self.session.execute(
                select(
                    CollectionReplicationStatus.partition,
                    func.max(CollectionReplicationStatus.version),
                )
                .filter_by(collection=self.collection)
                .group_by(CollectionReplicationStatus.partition)
            )
        ).one_or_none()
        if maybe_result is None:
            return None, 0
        return maybe_result


class ReplicationManager:
    """Manages the replication for a single collection."""

    def __init__(
        self,
        aramaki: "Manager",
        collection: type["Collection"],
    ):
        self.collection = collection
        self.aramaki = aramaki
        self.new_update_message_event = asyncio.Event()
        self.new_catchup_step_message_event = asyncio.Event()
        self.catchup_finished_event = asyncio.Event()
        self.tasks = set()
        # We have two types of locks here:
        #
        # The update lock to ensure that no update runs concurrently:
        # This is required because we could otherwise get an inconsistent
        # DB state and buffer state, because we delete data in them
        # during catchup.
        self.update_lock = asyncio.Lock()
        # The catchup lock to ensure that all requirements are fulfilled before
        # processing a catchup.step message. This is especially important for
        # the full sync case, as the internal DB (and also the consumer side DB)
        # is wiped.
        self.catchup_lock = asyncio.Lock()

        self.update_buffer: dict[int, dict] = {}
        self.catchup_buffer: dict[int, dict] = {}

        task = asyncio.create_task(self.start())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        self.aramaki.register_callback(
            "directory.collection.catchup.start",
            {
                "principal": self.aramaki.principal,
                "application": self.aramaki.application,
                "collection": self.collection,
            },
            self.process_start_catchup_message,
        )
        self.aramaki.register_callback(
            "directory.collection.catchup.step",
            {
                "principal": self.aramaki.principal,
                "application": self.aramaki.application,
                "collection": self.collection,
            },
            self.process_catchup_step_message,
        )
        self.aramaki.register_callback(
            "directory.collection.update",
            {
                "collection": self.collection,
            },
            self.process_update_message,
        )

        catchup = asyncio.create_task(self.request_catchup())
        self.tasks.add(catchup)

    async def start(self):
        asyncio.create_task(self.process_update_buffer())
        asyncio.create_task(self.process_catchup_step_buffer())

    async def process_update_message(self, msg: dict):
        async with self.aramaki.db.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )
            if (
                current_partition is None
                or current_partition != msg["partition"]
            ):
                await self.request_catchup()
                return

            # Discard the message if we have a newer version
            if msg["version"] < current_version:
                return

        self.update_buffer[msg["version"]] = msg
        self.new_update_message_event.set()

    async def process_update_buffer(self):
        while True:
            await self.new_update_message_event.wait()
            with self.update_lock:
                version = min(self.update_buffer.keys())
                msg = self.update_buffer[version]
                async with self.aramaki.db.session() as db_session:
                    # This is duplicated, but we may want that
                    (
                        current_partition,
                        current_version,
                    ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                        db_session, self.collection
                    )

                    if (
                        current_partition is None
                        or current_partition != msg["partition"]
                    ):
                        asyncio.create_task(self.request_catchup())
                        return
                    if msg["version"] < current_version:
                        # Discard the message, we already processed a newer one
                        del self.update_buffer[msg["version"]]
                    if msg["version"] > current_version + 1:
                        # This message is too new, keep it in the buffer and wait for another one
                        return

                    record = await db_session.get(
                        CollectionReplicationStatus,
                        {
                            "collection": self.collection,
                            "partition": msg["partition"],
                            "record_id": msg["record_id"],
                        },
                    )
                    if record is None:
                        await CollectionReplicationStatus.create(
                            db_session,
                            collection=self.collection,
                            partition=msg["partition"],
                            record_id=msg["record_id"],
                        )

                    record.version = msg["version"]
                    record.data = msg["data"]
                # XXX: Maybe we want to return another structure here instead?
                await self.update_callback(msg)
                del self.update_buffer[msg["version"]]
                if len(self.update_buffer.keys()) > 0:
                    # Trigger itself to process messages in the buffer with a higher version that
                    # are now processable
                    self.new_update_message_event.set()

    async def get_collection_with_session(self):
        async with self.aramaki.db.session() as db_session:
            return self.collection(db_session)

    async def request_catchup(self):
        async with self.aramaki.db.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )

        await self.aramaki.send_message(
            "directory.collection.catchup.request",
            {
                "collection": self.collection,
                "partition": current_partition,
                "current_version": current_version,
            },
        )

    async def process_start_catchup_message(self, msg: dict):
        async with self.aramaki.db.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )
        # XXX: Maybe we need to signal to the server that our partition is not the one they think we have.
        # Otherwise, we might wait for an infinite time to receive a catchup.step with version_from = 1
        full_sync = (
            msg["start_version"] in (0, 1)
            or msg["partition"] != current_partition
        )

        async with self.update_lock:
            if full_sync:
                async with self.catchup_lock:
                    self.start_full_sync_callback()
                    async with self.aramaki.db.session() as db_session:
                        records = db_session.execute(
                            sqlalchemy.select(
                                CollectionReplicationStatus
                            ).filter_by(collection=self.collection)
                        )
                        for record in records:
                            record.delete()
                        # There is currently no record in the partition
                        if msg["start_version"] == 0:
                            # remember the partition of the collection here
                            pass

            await self.catchup_finished_event.wait()

    async def process_catchup_step_message(self, msg: dict):
        self.catchup_buffer[msg["from_version"]] = msg
        self.new_catchup_step_message_event.set()

    async def process_catchup_step_buffer(self):
        while True:
            await self.new_catchup_step_message_event.wait()
            version = min(self.catchup_buffer.keys())
            msg = self.update_buffer[version]
            async with self.aramaki.db.session() as db_session:
                (
                    current_partition,
                    current_version,
                ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                    db_session, self.collection
                )

                if msg["to_version"] < current_version:
                    # Discard the message, we already processed a newer one
                    del self.update_buffer[msg["version"]]
                if msg["from_version"] > current_version + 1:
                    # This message is too new, keep it in the buffer and wait for another one
                    return

                record = await db_session.get(
                    CollectionReplicationStatus,
                    {
                        "collection": self.collection,
                        "partition": msg["partition"],
                        "record_id": msg["record_id"],
                    },
                )
                if record is None:
                    await CollectionReplicationStatus.create(
                        db_session,
                        collection=self.collection,
                        partition=msg["partition"],
                        record_id=msg["record_id"],
                    )

                record.version = msg["to_version"]
                record.data = msg["data"]
            # XXX: Maybe we want to return another structure here instead?
            await self.update_callback(msg)
            del self.update_buffer[msg["version"]]
            if len(self.catchup_buffer.keys()) > 0:
                # Trigger itself to process messages in the buffer with a higher version that
                # are now processable
                self.new_catchup_step_message_event.set()

            if msg["is_final_version"]:
                self.catchup_finished_event.set()
