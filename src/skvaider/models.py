from sqlalchemy.orm import Mapped, mapped_column

from skvaider.db import Base


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    username: Mapped[str] = mapped_column(primary_key=True)
    password: Mapped[str] = mapped_column()
