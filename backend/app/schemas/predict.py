"""识别接口的请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_id: int = Field(..., description="类别 ID（0—101，与数据集一致）")
    name_en: str = Field(..., description="英文名")
    name_cn: str = Field(..., description="中文名")
    confidence: float = Field(..., description="置信度（0—1）")


class PredictOut(BaseModel):
    record_id: int = Field(..., description="识别记录 ID，可用于历史查询与删除")
    model_name: str = Field(..., description="使用的模型标识")
    model_display_name: str = Field(..., description="模型展示名")
    model_pretrained: str = Field("", description="预训练来源")
    latency_ms: float = Field(..., description="纯推理耗时（毫秒）")
    image_url: str = Field(..., description="已保存图片的访问路径")
    predictions: list[PredictionItem] = Field(..., description="Top-5 候选，按置信度降序")
