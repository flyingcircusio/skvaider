"""collection_record.version nullable for in-progress full-sync sentinel

Revision ID: 3a1f8c2d4e5b
Revises: 092d0a531642
Create Date: 2026-04-16 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3a1f8c2d4e5b"
down_revision: Union[str, Sequence[str], None] = "092d0a531642"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("collection_record") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("collection_record") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Integer(),
            nullable=False,
        )
