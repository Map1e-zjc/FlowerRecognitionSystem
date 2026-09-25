"""用户表。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    records: Mapped[list["RecognitionRecord"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} username={self.username!r}>"
