from sqlalchemy.orm import Mapped, mapped_column

from skvaider.db import Base


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    resource_group: Mapped[str] = mapped_column()
    secret_hash: Mapped[str] = mapped_column()
