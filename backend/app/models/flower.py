"""花卉百科表（102 类，id 与数据集 class_id 0-based 对齐）。"""

from __future__ import annotations

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Flower(Base):
    __tablename__ = "flowers"

    # 与数据集类别 ID 严格对齐（0—101），不使用自增
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name_en: Mapped[str] = mapped_column(String(80), nullable=False)
    name_cn: Mapped[str] = mapped_column(String(80), nullable=False, index=True)

    category: Mapped[str] = mapped_column(String(40), default="", server_default="")
    category_cn: Mapped[str] = mapped_column(String(40), default="", server_default="")
    family: Mapped[str] = mapped_column(String(60), default="", server_default="")
    genus: Mapped[str] = mapped_column(String(60), default="", server_default="")
    bloom_season: Mapped[str] = mapped_column(String(60), default="", server_default="")
    color: Mapped[str] = mapped_column(String(80), default="", server_default="")
    light: Mapped[str] = mapped_column(String(120), default="", server_default="")
    watering: Mapped[str] = mapped_column(String(120), default="", server_default="")
    soil: Mapped[str] = mapped_column(String(120), default="", server_default="")
    propagation: Mapped[str] = mapped_column(String(120), default="", server_default="")
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    care_tips: Mapped[str] = mapped_column(Text, default="", server_default="")
    sample_image: Mapped[str] = mapped_column(String(255), default="", server_default="")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Flower id={self.id} name={self.name_cn!r}>"
