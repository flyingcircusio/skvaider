import asyncio
import importlib.resources
from pathlib import Path
from typing import Callable

import sqlalchemy

from skvaider.aramaki import Manager
from skvaider.aramaki.db import DBSessionManager
from skvaider.aramaki.models import CollectionReplicationStatus


class CollectionReplicationManager:
    def __init__(
        self,
        aramaki_manager: Manager,
        collection: str,
        state_directory: Path,
        # TODO: improve types
        update_callback: Callable,
        start_full_sync_callback: Callable,
        end_full_sync_callback: Callable,
    ):
        self.collection = collection
        self.state_directory: Path = state_directory
        self.aramaki_manager = aramaki_manager
        self.new_update_message_event = asyncio.Event()
        self.new_catchup_step_message_event = asyncio.Event()
        self.catchup_finished_event = asyncio.Event()
        self.tasks = set()
        self.update_callback: Callable = update_callback
        self.start_full_sync_callback: Callable = start_full_sync_callback
        self.end_full_sync_callback: Callable = end_full_sync_callback
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

        self.db_sessionmanager: DBSessionManager | None = None
        self.update_buffer: dict[int, dict] = {}
        self.catchup_buffer: dict[int, dict] = {}

        task = asyncio.create_task(self.start())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        self.aramaki_manager.register_callback(
            "directory.collection.catchup.start",
            {
                "principal": self.aramaki_manager.principal,
                "application": self.aramaki_manager.application,
                "collection": self.collection,
            },
            self.process_start_catchup_message,
        )
        self.aramaki_manager.register_callback(
            "directory.collection.catchup.step",
            {
                "principal": self.aramaki_manager.principal,
                "application": self.aramaki_manager.application,
                "collection": self.collection,
            },
            self.process_update_message,
        )
        self.aramaki_manager.register_callback(
            "directory.collection.update",
            {
                "collection": self.collection,
            },
            self.process_update_message,
        )

        self.request_catchup()

    async def start(self):
        db_url = f"sqlite+aiosqlite://{self.state_directory}/aramaki.sqlite3"
        self.db_sessionmanager = DBSessionManager(db_url)

        async with self.db_sessionmanager.session() as db_session:
            await db_session.get_bind().execute(
                sqlalchemy.text(
                    (importlib.resources.files(__package__) / "migrations" / "0001.sql")
                    .open("r", encoding="utf-8")
                    .read()
                )
            )

        asyncio.create_task(self.process_update_buffer())
        asyncio.create_task(self.process_catchup_step_buffer())

    async def process_update_message(self, msg: dict):
        async with self.db_sessionmanager.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )
            if current_partition is None or current_partition != msg["partition"]:
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
                async with self.db_sessionmanager.session() as db_session:
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

    async def request_catchup(self):
        async with self.db_sessionmanager.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )

        await self.aramaki_manager.send_message(
            "directory.collection.catchup.request",
            {
                "collection": self.collection,
                "partition": current_partition,
                "current_version": current_version,
            },
        )

    async def process_start_catchup(self, msg: dict):
        async with self.db_sessionmanager.session() as db_session:
            (
                current_partition,
                current_version,
            ) = await CollectionReplicationStatus.currently_known_partition_and_version(
                db_session, self.collection
            )
        # XXX: Maybe we need to signal to the server that our partition is not the one they think we have.
        # Otherwise, we might wait for an infinite time to receive a catchup.step with version_from = 1
        full_sync = (
            msg["start_version"] in (0, 1) or msg["partition"] != current_partition
        )

        async with self.update_lock:
            if full_sync:
                async with self.catchup_lock:
                    self.start_full_sync_callback()
                    async with self.db_sessionmanager.session() as db_session:
                        records = db_session.execute(
                            sqlalchemy.select(CollectionReplicationStatus).filter_by(
                                collection=self.collection
                            )
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
            async with self.db_sessionmanager.session() as db_session:
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
