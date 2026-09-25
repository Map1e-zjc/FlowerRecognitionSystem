"""识别记录表。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class RecognitionRecord(Base):
    __tablename__ = "recognition_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_class_id: Mapped[int] = mapped_column(
        ForeignKey("flowers.id"), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top5_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]", server_default="[]")
    model_name: Mapped[str] = mapped_column(String(50), nullable=False, default="", server_default="")
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )

    user: Mapped["User"] = relationship(back_populates="records")  # noqa: F821
    flower: Mapped["Flower"] = relationship()  # noqa: F821

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RecognitionRecord id={self.id} user={self.user_id} "
            f"class={self.predicted_class_id} conf={self.confidence:.3f}>"
        )
