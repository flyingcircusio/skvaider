import aramaki
import aramaki.collection
import aramaki.db


class DummyCollection(aramaki.Collection):
    collection = "mydummycollection"


async def test_collection_basics(tmpdir):
    db = aramaki.db.DBSessionManager(tmpdir)
    db.upgrade()

    async with db.session() as session:
        collection = DummyCollection(session)

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
