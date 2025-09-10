import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import sqlalchemy
from sqlalchemy import JSON, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from aramaki.db import Base

if TYPE_CHECKING:
    from aramaki import Manager


class Record(Base):
    __tablename__ = "collection_record"

    collection: Mapped[str] = mapped_column(primary_key=True)
    partition: Mapped[str] = mapped_column(primary_key=True)
    record_id: Mapped[str] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column()
    data: Mapped[dict] = mapped_column(type_=JSON)


async def _currently_known_partition_and_version(
    db_session: AsyncSession, collection: str
):
    # Use the fact that (collection, partition) is unique for the client view here.
    maybe_result = (
        await db_session.execute(
            select(
                Record.partition,
                func.max(Record.version),
            )
            .filter_by(collection=collection)
            .group_by(Record.partition)
        )
    ).one_or_none()
    if maybe_result is None:
        return None, 0
    return maybe_result


async def _set_null_record(
    session: AsyncSession, collection: str, partition: int, version: int
):
    record = await session.get(
        Record,
        {
            "collection": collection,
            "partition": partition,
            "record_id": "",
        },
    )
    if record is None:
        record = await Record.create(
            session,
            collection=collection,
            partition=partition,
            record_id="",
            data={},
        )
    record.version = version


class Collection:
    """Provides a read-only view to an Aramaki collection.

    The collection provides a simple interface:

        get(key, default=None)
        keys()

    The data is kept (eventually consistent) with the server side, but always available.
    From a CAP perspective, this drops the consistency

    It is updated in the background automatically and automatically detects the need for partial and full syncs.

    """

    collection: str  # The identifier for the collection in Aramaki

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, key: str, default=None) -> dict | None:
        assert key  # key must be non-empty - defensive against clients breaking protocol that "" is our internal null record
        result = (
            (
                await self.session.execute(
                    select(Record).filter_by(
                        collection=self.collection, record_id=key
                    )
                )
            )
            .scalars()
            .one_or_none()
        )
        if not result:
            return default
        return result.data

    async def keys(self) -> list[str]:
        result = (
            (
                await self.session.execute(
                    select(Record)
                    .filter_by(
                        collection=self.collection,
                    )
                    .filter(Record.record_id != "")
                )
            )
            .scalars()
            .all()
        )
        return [x.record_id for x in result]


class ReplicationManager:
    """Manages the replication for a single collection.

    The replication manager interacts with aramaki by sending and receiving
    replication-related messages.

    We generally buffer messages as we need to ensure proper order (based on the
    versions we receive) when processing them and we need to isolate sync
    activities from regular update activities.

    """

    catchup_running = False

    def __init__(
        self,
        aramaki: "Manager",
        collection: type["Collection"],
    ):
        self.collection = collection
        self.aramaki = aramaki
        self.catchup_finished = asyncio.Event()
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

        self.update_buffer = PriorityPushbackQueue()
        self.catchup_buffer = PriorityPushbackQueue()

        self.aramaki.register_message_handler(
            "directory.collection.catchup.start",
            self.process_start_catchup_message,
            principal=self.aramaki.principal,
            application=self.aramaki.application,
            collection=self.collection.collection,
        )
        self.aramaki.register_message_handler(
            "directory.collection.catchup.step",
            self.process_catchup_step_message,
            principal=self.aramaki.principal,
            application=self.aramaki.application,
            collection=self.collection.collection,
        )
        self.aramaki.register_message_handler(
            "directory.collection.update",
            self.process_update_message,
            collection=self.collection.collection,
        )

        for t in [
            self.request_catchup,
            self.process_update_buffer,
            self.process_catchup_step_buffer,
        ]:
            task = asyncio.create_task(t())
            self.tasks.add(task)

    def stop(self):
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

    @asynccontextmanager
    async def get_collection_with_session(self):
        async with self.aramaki.db.session() as db_session:
            yield self.collection(db_session)

    async def process_update_message(self, msg: dict):
        await self.update_buffer.put(msg["version"], msg)

    async def process_update_buffer(self):
        while True:
            update_version, msg = await self.update_buffer.get()
            assert msg["record_id"]  # defensive against peers breaking protocol
            async with (
                self.update_lock,
                self.aramaki.db.session() as db_session,
            ):
                (
                    current_partition,
                    current_version,
                ) = await _currently_known_partition_and_version(
                    db_session, self.collection.collection
                )

                if (
                    current_partition is None
                    or current_partition != msg["partition"]
                ):
                    # There's a certain chance here that, if we get too much of chaotic messages, then we may end up in a flapping
                    # state, e.g. if there's a number of update messages for two partitions (while partitions change) that can cause
                    # (due to version numbers of the new partition being lower) us to early on detect the partition change,
                    # but then find the old partition again in the higher version numbers, this would trigger multiple catchups
                    # but I think this should settle nicely if the upstream partition assignment hasn't changed, so we'd
                    # basically request more catchups but those will quickly converge and not replay everything ...
                    asyncio.create_task(self.request_catchup())
                    self.update_buffer.task_done()
                    continue

                if update_version <= current_version:
                    # This version is old, ignore it.
                    self.update_buffer.task_done()
                    continue

                if update_version > current_version + 1:
                    # This message is too new, keep it in the buffer and wait for another one
                    await self.update_buffer.put_back(update_version, msg)
                    continue

                assert update_version == current_version + 1

                record = await db_session.get(
                    Record,
                    {
                        "collection": self.collection.collection,
                        "partition": msg["partition"],
                        "record_id": msg["record_id"],
                    },
                )
                if msg["change"] == "update":
                    if record is None:
                        record = await Record.create(
                            db_session,
                            collection=self.collection.collection,
                            partition=msg["partition"],
                            record_id=msg["record_id"],
                        )
                    record.version = msg["version"]
                    record.data = msg["data"]
                if msg["change"] == "delete" and record is not None:
                    await record.delete(db_session)
                if msg["change"] == "delete" or msg["change"] == "null":
                    await _set_null_record(
                        db_session,
                        self.collection.collection,
                        msg["partition"],
                        msg["version"],
                    )
            self.update_buffer.task_done()

    async def request_catchup(self):
        async with self.aramaki.db.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await _currently_known_partition_and_version(
                db_session, self.collection.collection
            )

        await self.aramaki.send_message(
            "directory.collection.catchup.request",
            {
                "collection": self.collection.collection,
                "partition": current_partition,
                "current_version": current_version,
            },
        )

    async def process_start_catchup_message(self, msg: dict):
        """Start and manage the overall catchup process.

        This can happen in various combinations.

        However, one thing is important: we want to avoid removing records
        too early and delay deletion of records as long as possible to avoid
        flapping for our clients.

        For example:

        - we only mass delete records if the server signals that the partition is empty
        - in partial catchups we only delete those records that we receive deletions for
        - in full catchups we only delete those records that have not received data after the sync

        """
        self.catchup_finished.clear()

        start_version = msg["start_version"]
        partition = msg["partition"]

        async with self.aramaki.db.session() as db_session:
            r = await _currently_known_partition_and_version(
                db_session, self.collection.collection
            )
            current_partition, current_version = r

        async with self.update_lock, self.catchup_lock:
            if start_version == 0:
                # The partition is empty and we will not receive further messages.
                async with self.aramaki.db.session() as session:
                    await session.execute(
                        sqlalchemy.delete(Record).filter_by(
                            collection=self.collection.collection
                        )
                    )
                    await _set_null_record(
                        session, self.collection.collection, partition, 0
                    )
                return

            if partition != current_partition and start_version > 1:
                # We switched partitions but that means we can't do a partial sync.
                # Re-request a full sync (not re-using request_catchup because
                # we don't want to use the persistent data we have).
                await self.aramaki.send_message(
                    "directory.collection.catchup.request",
                    {
                        "collection": self.collection.collection,
                        "partition": partition,
                        "current_version": 0,
                    },
                )
                return

            is_full_sync = start_version == 1

            if is_full_sync:
                # Remove version markers from all records for now (and delete them when the catchup has finished.)
                # We keep the records to avoid "holes" for our clients while we're updating from an earlier state.
                async with self.aramaki.db.session() as db_session:
                    await db_session.execute(
                        sqlalchemy.update(Record)
                        .filter_by(collection=self.collection.collection)
                        .values(version=None)
                    )

            # Let the sync continue
            await self.catchup_finished.wait()

            if is_full_sync:
                # Delete all items that haven't received version markers during the catchup.
                async with self.aramaki.db.session() as db_session:
                    await db_session.execute(
                        sqlalchemy.delete(Record).filter_by(
                            collection=self.collection.collection, version=None
                        )
                    )

    async def process_catchup_step_message(self, msg: dict):
        await self.catchup_buffer.put(msg["from_version"], msg)

    async def process_catchup_step_buffer(self):
        while True:
            from_version, msg = await self.catchup_buffer.get()
            async with self.aramaki.db.session() as db_session:
                (
                    current_partition,
                    current_version,
                ) = await _currently_known_partition_and_version(
                    db_session, self.collection.collection
                )
                if msg["from_version"] < current_version:
                    # Discard the message, we already processed a newer one
                    self.catchup_buffer.task_done()
                    continue
                if msg["from_version"] > current_version:
                    # This message is too new, keep it in the buffer and wait for another one
                    await self.catchup_buffer.put_back(from_version, msg)
                    continue

                record = await db_session.get(
                    Record,
                    {
                        "collection": self.collection.collection,
                        "partition": msg["partition"],
                        "record_id": msg.get("record_id"),
                    },
                )
                if msg["change"] == "update":
                    if record is None:
                        record = await Record.create(
                            db_session,
                            collection=self.collection.collection,
                            partition=msg["partition"],
                            record_id=msg["record_id"],
                        )

                    record.version = msg["to_version"]
                    record.data = msg["data"]

                if msg["change"] == "delete" and record is not None:
                    await record.delete(db_session)

                if msg["change"] == "delete" or msg["change"] == "null":
                    await _set_null_record(
                        db_session,
                        self.collection.collection,
                        msg["partition"],
                        msg["to_version"],
                    )
            # After commit
            if msg.get("is_final_record"):
                self.catchup_finished.set()
            self.catchup_buffer.task_done()


class PriorityPushbackQueue:
    """A priority queue that allows putting items back.

    It also supports non-hashable objects.

    Putting an item back means we do not pass it out to waiters
    until at least one another item has been put in.

    """

    metric_put_back = 0

    def __init__(self):
        self.items = {}
        self.queue = asyncio.PriorityQueue()
        self.new_item = asyncio.Event()

    async def put(self, priority, item):
        self.items.setdefault(priority, []).append(item)
        await self.queue.put(priority)
        self.new_item.set()

    async def put_back(self, priority, item):
        self.metric_put_back += 1
        self.items.setdefault(priority, []).append(item)
        self.new_item.clear()
        await self.queue.put(priority)
        # The client should not mark this as "task done", we do that for it.
        self.queue.task_done()

    async def get(self):
        await self.new_item.wait()
        priority = await self.queue.get()
        return priority, self.items[priority].pop(0)

    async def join(self):
        return await self.queue.join()

    def task_done(self):
        self.queue.task_done()
