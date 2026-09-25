"""模型指标与统计的响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMetricOut(BaseModel):
    name: str
    tag: str = ""
    display_name: str
    pretrained: str = ""
    top1: float = Field(..., description="测试集 Top-1（%）")
    top5: float = Field(..., description="测试集 Top-5（%）")
    macro_f1: float = 0.0
    params_m: float = 0.0
    infer_ms: float = 0.0
    infer_p95_ms: float = 0.0
    train_minutes: float = 0.0
    epochs_run: int = 0
    best_epoch: int = 0
    image_size: int = 224
    num_samples: int = 0
    is_default: bool = False
    confusion_figure: str = ""
    error_cases_figure: str = ""


class CurvePoint(BaseModel):
    epoch: int
    train_loss: float | None = None
    train_top1: float | None = None
    val_loss: float | None = None
    val_top1: float | None = None
    val_top5: float | None = None
    epoch_seconds: float | None = None


class CurvesOut(BaseModel):
    name: str
    tag: str
    points: list[CurvePoint] = Field(default_factory=list)


class ConfusionPair(BaseModel):
    true_id: int
    true_name: str
    pred_id: int
    pred_name: str
    count: int


class ConfusionOut(BaseModel):
    name: str
    tag: str
    top1: float = Field(0.0, description="测试集 Top-1（%），用于校验对角线和一致性")
    class_names: list[str] = Field(default_factory=list)
    matrix: list[list[int]] = Field(default_factory=list, description="102×102 混淆矩阵（行=真实，列=预测）")
    per_class_acc: list[float] = Field(default_factory=list)
    per_class_support: list[int] = Field(default_factory=list)
    top_pairs: list[ConfusionPair] = Field(default_factory=list)
    figure: str = Field("", description="混淆矩阵图片 URL")
    num_samples: int = 0


class ConfidenceBucket(BaseModel):
    label: str
    count: int


class DailyPoint(BaseModel):
    date: str
    count: int


class TopClass(BaseModel):
    class_id: int
    name_cn: str
    name_en: str
    count: int


class StatsOverview(BaseModel):
    total: int = Field(..., description="识别总次数")
    today: int = Field(..., description="今日识别次数")
    unique_classes: int = Field(..., description="识别过的不同花卉数")
    avg_confidence: float = Field(..., description="平均 Top-1 置信度（%）")
    confidence_buckets: list[ConfidenceBucket] = Field(default_factory=list)
    daily_trend: list[DailyPoint] = Field(default_factory=list, description="最近 7 天")
    top_classes: list[TopClass] = Field(default_factory=list, description="识别最多的 10 类")
