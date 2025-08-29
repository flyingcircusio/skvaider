from sqlalchemy import UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, mapped_column

from skvaider.aramaki.db import Base, DBSession


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

    @classmethod
    async def currently_known_partition_and_version(
        cls, db_session: DBSession, collection: str
    ) -> tuple[str | None, int]:
        # Use the fact that (collection, partition) is unique for the client view here.
        maybe_result = (
            (
                await db_session.execute(
                    select(cls.partition, func.max(cls.version))
                    .filter_by(collection=collection)
                    .group_by(cls.partition)
                )
            )
            .scalar()
            .one_or_none()
        )
        if maybe_result is None:
            return None, 0
        return maybe_result
