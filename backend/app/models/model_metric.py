"""模型指标表（由 ml/outputs/metrics.json 种子导入，供前端"模型对比"页读取）。"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    tag: Mapped[str] = mapped_column(String(50), default="", server_default="")
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    pretrained: Mapped[str] = mapped_column(String(120), default="", server_default="")

    top1: Mapped[float] = mapped_column(Float, nullable=False)
    top5: Mapped[float] = mapped_column(Float, nullable=False)
    macro_f1: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    params_m: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    infer_ms: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    infer_p95_ms: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    train_minutes: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    epochs_run: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    best_epoch: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    image_size: Mapped[int] = mapped_column(Integer, default=224, server_default="224")
    num_samples: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    is_default: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    weights_path: Mapped[str] = mapped_column(String(255), default="", server_default="")
    confusion_figure: Mapped[str] = mapped_column(String(255), default="", server_default="")
    curves_figure: Mapped[str] = mapped_column(String(255), default="", server_default="")
    error_cases_figure: Mapped[str] = mapped_column(String(255), default="", server_default="")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModelMetric {self.name} top1={self.top1}>"
