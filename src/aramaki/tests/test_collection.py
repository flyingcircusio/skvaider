import asyncio
import unittest.mock

import pytest

import aramaki
import aramaki.collection
import aramaki.db


class DummyCollection(aramaki.Collection):
    collection = "mydummycollection"


async def test_collection_basics(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    manager = unittest.mock.Mock()
    manager.aramaki.db.session = db.session

    collection = DummyCollection(manager)

    async with db.session() as session:
        assert (
            await aramaki.collection._currently_known_partition_and_version(
                session, collection.collection
            )
        ) == (
            None,
            0,
        )

    assert await collection.get("test") is None
    assert await collection.keys() == []

    async with db.session() as session:
        item = aramaki.collection.Record(
            collection="mydummycollection",
            partition="x",
            record_id="test",
            data={},
            version=1,
        )
        session.add(item)

    assert await collection.get("test") == {}
    assert await collection.keys() == ["test"]

    async with db.session() as session:
        assert (
            await aramaki.collection._currently_known_partition_and_version(
                session, collection.collection
            )
        ) == (
            "x",
            1,
        )

    assert await collection.get("foobar") is None
    assert await collection.get("foobar", 5) == 5


async def test_null_record(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    async with db.session() as session:
        collection = DummyCollection(session)

        with pytest.raises(AssertionError):
            await collection.get("")

        assert (
            await session.get(
                aramaki.collection.Record,
                {
                    "collection": "collection-1",
                    "partition": "partition-1",
                    "record_id": "",
                },
            )
        ) is None

        await aramaki.collection._set_null_record(
            session, "collection-1", "partition-1", 10
        )

        assert (
            await session.get(
                aramaki.collection.Record,
                {
                    "collection": "collection-1",
                    "partition": "partition-1",
                    "record_id": "",
                },
            )
        ).version == 10

        await aramaki.collection._set_null_record(
            session, "collection-1", "partition-1", 20
        )

        assert (
            await session.get(
                aramaki.collection.Record,
                {
                    "collection": "collection-1",
                    "partition": "partition-1",
                    "record_id": "",
                },
            )
        ).version == 20


async def test_pushback_queue():
    queue = aramaki.collection.PriorityPushbackQueue()
    await queue.put(3, "3")
    await queue.put(1, "1")
    await queue.put(2, "2")
    assert await queue.get() == (1, "1")
    assert await queue.get() == (2, "2")
    assert await queue.get() == (3, "3")

    await queue.put(3, "3")
    await queue.put(2, "2")
    await queue.get() == 2
    await queue.put_back(2, "2")

    task = asyncio.create_task(queue.get())
    await asyncio.sleep(0.1)
    assert not task.done()
    await queue.put(1, "1")
    await task
    assert task.result() == (1, "1")
    assert await queue.get() == (2, "2")
    assert await queue.get() == (3, "3")


class AramakiDummy:
    principal = "host1"
    application = "test"

    def __init__(self):
        self.message = None
        self.message_received = asyncio.Event()

    def register_message_handler(self, type_, target, **scope):
        pass

    async def send_message(self, *args, **kw):
        self.message = args, kw
        self.message_received.set()


async def test_replication(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()
    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 0,
                "partition": None,
            },
        ),
        {},
    )
    # let's run through a full sync
    assert await collection.get("1") is None
    catchup_task = asyncio.create_task(
        manager.process_start_catchup_message(
            {"start_version": 1, "partition": "partition-1"}
        )
    )
    await manager.process_catchup_step_message(
        {
            "from_version": 0,
            "record_id": "1",
            "partition": "partition-1",
            "to_version": 1,
            "change": "update",
            "is_final_record": False,
            "data": {"key": "value"},
        }
    )
    await manager.process_catchup_step_message(
        {
            "from_version": 1,
            "record_id": "1",
            "partition": "partition-1",
            "to_version": 2,
            "change": "update",
            "is_final_record": True,
            "data": {"key": "value"},
        }
    )
    await catchup_task
    assert await collection.get("1") == {"key": "value"}

    # now lets run some updates
    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 3,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )

    await manager.update_buffer.join()

    assert await collection.get("1") == {"key": "other-value"}

    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 4,
            "change": "delete",
        }
    )

    await manager.update_buffer.join()

    assert await collection.get("1") is None

    # This is an outdated version
    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 2,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )
    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 3,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )

    await manager.update_buffer.join()

    assert await collection.get("1") is None

    # This is too new
    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 6,
            "change": "update",
            "data": {"key": "value-6"},
        }
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(manager.update_buffer.join(), timeout=0.1)

    assert await collection.get("1") is None

    # This is the right next version and we then
    # catch up to version 5 immediately.
    await manager.process_update_message(
        {
            "record_id": "2",
            "partition": "partition-1",
            "version": 5,
            "change": "update",
            "data": {"key": "value-5"},
        }
    )

    await manager.update_buffer.join()

    assert await collection.get("1") == {"key": "value-6"}
    assert await collection.get("2") == {"key": "value-5"}

    manager.stop()


async def test_replication_empty_sync(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    await aramaki_manager.message_received.wait()
    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 0,
                "partition": None,
            },
        ),
        {},
    )

    collection = manager.bound_collection

    # let's run through a full sync
    assert await collection.keys() == []

    await manager.process_start_catchup_message(
        {"start_version": 0, "partition": "partition-1"}
    )

    assert await collection.keys() == []

    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 1,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )

    await asyncio.wait_for(manager.update_buffer.join(), timeout=1)

    assert await collection.keys() == ["1"]

    await manager.process_start_catchup_message(
        {"start_version": 0, "partition": "partition-1"}
    )

    assert await collection.keys() == []


async def test_replication_catchup_sync(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    async with db.session() as session:
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="1",
            version=4,
            data={},
        )
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="2",
            version=5,
            data={},
        )

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()

    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 5,
                "partition": "partition-1",
            },
        ),
        {},
    )

    # let's run through a partial sync
    catchup_task = asyncio.create_task(
        manager.process_start_catchup_message(
            {"start_version": 6, "partition": "partition-1"}
        )
    )
    await manager.process_catchup_step_message(
        {
            "from_version": 5,
            "record_id": "1",
            "partition": "partition-1",
            "to_version": 6,
            "change": "delete",
        }
    )
    await manager.process_catchup_step_message(
        {
            "from_version": 6,
            "record_id": "2",
            "partition": "partition-1",
            "to_version": 10,
            "change": "update",
            "data": {"key": "value-10"},
            "is_final_record": True,
        }
    )

    await catchup_task

    assert await collection.get("1") is None
    assert await collection.get("2") == {"key": "value-10"}


async def test_full_sync_deletes_superfluous_records_at_end(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    async with db.session() as session:
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="1",
            version=4,
            data={},
        )
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="2",
            version=5,
            data={},
        )

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()

    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 5,
                "partition": "partition-1",
            },
        ),
        {},
    )

    # let's run through a partial sync
    catchup_task = asyncio.create_task(
        manager.process_start_catchup_message(
            {"start_version": 0, "partition": "partition-1"}
        )
    )
    assert await collection.keys() == ["1", "2"]
    await manager.process_catchup_step_message(
        {
            "from_version": 0,
            "record_id": "1",
            "partition": "partition-1",
            "to_version": 1,
            "change": "null",
            "is_final_record": True,
        }
    )
    await catchup_task
    assert await collection.keys() == []


async def test_update_with_missing_partition_triggers_catchup(
    tmpdir,
):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()
    aramaki_manager.message = None

    assert await collection.keys() == []
    # Send an update message -> triggers a catchup

    await manager.process_update_message(
        {
            "record_id": "1",
            "partition": "partition-1",
            "version": 5,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )

    await manager.update_buffer.join()

    assert await collection.keys() == []

    while aramaki_manager.message is None:
        await asyncio.sleep(0.1)

    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 0,
                "partition": None,
            },
        ),
        {},
    )


async def test_update_with_wrong_partition_triggers_catchup(
    tmpdir,
):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    async with db.session() as session:
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="1",
            version=4,
            data={},
        )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()
    aramaki_manager.message = None

    assert await collection.keys() == ["1"]

    # Send an update message -> triggers a catchup
    await manager.process_update_message(
        {
            "record_id": "5",
            "partition": "partition-2",
            "version": 5,
            "change": "update",
            "data": {"key": "other-value"},
        }
    )

    await manager.update_buffer.join()

    assert await collection.keys() == ["1"]
    while aramaki_manager.message is None:
        await asyncio.sleep(0.1)

    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 4,
                "partition": "partition-1",
            },
        ),
        {},
    )


async def test_partial_sync_wrong_partition_restart_catchup(
    tmpdir,
):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    async with db.session() as session:
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="1",
            version=4,
            data={},
        )

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()
    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 4,
                "partition": "partition-1",
            },
        ),
        {},
    )
    aramaki_manager.message = None

    assert await collection.keys() == ["1"]

    # Get a partial catchup initiation but with the wrong partition -> request a new full sync
    catchup_task = asyncio.create_task(
        manager.process_start_catchup_message(
            {"start_version": 2, "partition": "partition-2"}
        )
    )
    assert await collection.keys() == ["1"]
    await catchup_task

    assert await collection.keys() == ["1"]
    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 0,
                "partition": "partition-2",
            },
        ),
        {},
    )


async def test_replication_catchup_out_of_order(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    aramaki_manager = AramakiDummy()
    aramaki_manager.db = db

    async with db.session() as session:
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="1",
            version=4,
            data={},
        )
        await aramaki.collection.Record.create(
            session,
            collection="mydummycollection",
            partition="partition-1",
            record_id="2",
            version=5,
            data={},
        )

    manager = aramaki.collection.ReplicationManager(
        aramaki_manager, DummyCollection
    )

    collection = manager.bound_collection

    await aramaki_manager.message_received.wait()

    assert aramaki_manager.message == (
        (
            "directory.collection.catchup.request",
            {
                "collection": "mydummycollection",
                "current_version": 5,
                "partition": "partition-1",
            },
        ),
        {},
    )

    # let's run through a partial sync
    catchup_task = asyncio.create_task(
        manager.process_start_catchup_message(
            {"start_version": 6, "partition": "partition-1"}
        )
    )
    await manager.process_catchup_step_message(
        {
            "from_version": 6,
            "record_id": "2",
            "partition": "partition-1",
            "to_version": 10,
            "change": "update",
            "data": {"key": "value-10"},
            "is_final_record": True,
        }
    )

    while manager.catchup_buffer.metric_put_back == 0:
        await asyncio.sleep(0.1)

    await manager.process_catchup_step_message(
        {
            "from_version": 5,
            "record_id": "1",
            "partition": "partition-1",
            "to_version": 6,
            "change": "delete",
        }
    )

    await catchup_task

    assert await collection.get("1") is None
    assert await collection.get("2") == {"key": "value-10"}
