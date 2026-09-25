"""数据库层：声明基类、引擎与会话工厂。

SQLAlchemy 2.0 风格（``Mapped`` / ``mapped_column``）。
SQLite 需显式开启外键约束（默认关闭），否则 FK 约束形同虚设。
"""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的声明基类。"""


_is_sqlite = settings.sqlalchemy_url.startswith("sqlite")

engine = create_engine(
    settings.sqlalchemy_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    echo=False,
    future=True,
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db():
    """FastAPI 依赖：每请求一个会话，结束时关闭。"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
