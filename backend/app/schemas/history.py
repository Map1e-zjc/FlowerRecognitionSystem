"""识别历史的响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.predict import PredictionItem


class HistoryItem(BaseModel):
    """历史列表项。"""

    id: int
    image_url: str = Field(..., description="原图访问路径")
    class_id: int = Field(..., description="Top-1 类别 ID")
    name_cn: str
    name_en: str
    confidence: float
    model_name: str
    latency_ms: float
    created_at: datetime


class HistoryDetail(HistoryItem):
    """历史详情（含完整 Top-5）。"""

    top5: list[PredictionItem] = Field(default_factory=list)
