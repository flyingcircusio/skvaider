"""nullable version in collection_record

Revision ID: a1b2c3d4e5f6
Revises: 092d0a531642
Create Date: 2026-05-05 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "092d0a531642"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("collection_record") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("collection_record") as batch_op:
        batch_op.alter_column(
            "version",
            existing_type=sa.Integer(),
            nullable=False,
        )
