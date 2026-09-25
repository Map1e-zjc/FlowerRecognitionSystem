"""模型对比与可视化接口（FR-05）。无需登录。

数据来源全部是**离线训练产物**：``metrics.json``、``*_train.csv``、``*_eval.json``、``figures/*.png``。
不在前端硬编码任何指标。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import APIError, Envelope, ErrorCode, ok
from app.core.logging import get_logger
from app.db.session import get_db
from app.models import ModelMetric
from app.schemas.model_metric import (
    ConfusionOut,
    ConfusionPair,
    CurvePoint,
    CurvesOut,
    ModelMetricOut,
)
from app.services import metrics as metrics_service

log = get_logger("models")
router = APIRouter()


def _to_out(m: ModelMetric) -> ModelMetricOut:
    return ModelMetricOut(
        name=m.name,
        tag=m.tag or m.name,
        display_name=m.display_name,
        pretrained=m.pretrained or "",
        top1=m.top1,
        top5=m.top5,
        macro_f1=m.macro_f1,
        params_m=m.params_m,
        infer_ms=m.infer_ms,
        infer_p95_ms=m.infer_p95_ms,
        train_minutes=m.train_minutes,
        epochs_run=m.epochs_run,
        best_epoch=m.best_epoch,
        image_size=m.image_size,
        num_samples=m.num_samples,
        is_default=m.is_default,
        confusion_figure=metrics_service.figure_url(m.confusion_figure),
        error_cases_figure=metrics_service.figure_url(m.error_cases_figure),
    )


def _get_metric(db: Session, name: str) -> ModelMetric:
    metric = db.scalar(select(ModelMetric).where(ModelMetric.name == name))
    if metric is None:
        available = [m.name for m in db.scalars(select(ModelMetric)).all()]
        raise APIError(
            ErrorCode.NOT_FOUND,
            f"未知模型 {name!r}，可选：{'、'.join(available) or '（无）'}",
            http_status=404,
        )
    return metric


@router.get("", response_model=Envelope[list[ModelMetricOut]], summary="全部模型指标（按 Top-1 降序）")
def list_models(db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(ModelMetric).order_by(ModelMetric.top1.desc())).all()
    return ok([_to_out(m).model_dump() for m in rows])


@router.get(
    "/{name}/curves",
    response_model=Envelope[CurvesOut],
    summary="训练曲线数据点（loss / Top-1 / Top-5）",
)
def get_curves(name: str = Path(..., description="模型名，如 vit_b16"), db: Session = Depends(get_db)) -> dict:
    metric = _get_metric(db, name)
    tag = metric.tag or metric.name
    rows = metrics_service.read_curves(tag)
    points = [
        CurvePoint(
            epoch=int(r.get("epoch") or 0),
            train_loss=r.get("train_loss"),
            train_top1=r.get("train_top1"),
            val_loss=r.get("val_loss"),
            val_top1=r.get("val_top1"),
            val_top5=r.get("val_top5"),
            epoch_seconds=r.get("epoch_seconds"),
        )
        for r in rows
    ]
    return ok(CurvesOut(name=metric.name, tag=tag, points=points).model_dump())


@router.get(
    "/{name}/confusion",
    response_model=Envelope[ConfusionOut],
    summary="102×102 混淆矩阵与最易混淆类别对",
)
def get_confusion(name: str = Path(...), db: Session = Depends(get_db)) -> dict:
    metric = _get_metric(db, name)
    tag = metric.tag or metric.name
    payload = metrics_service.read_eval(tag)
    if not payload:
        raise APIError(
            ErrorCode.NOT_FOUND,
            f"未找到 {tag} 的评估结果，请先执行 python -m ml.evaluate",
            http_status=404,
        )

    # 评估 JSON 里没有独立的类别名列表，直接复用训练时的类别名（与权重同源，保证一致）
    from ml.dataset import get_class_names

    class_names = get_class_names()

    pairs = [
        ConfusionPair(
            true_id=int(p["true_id"]),
            true_name=p.get("true_name", ""),
            pred_id=int(p["pred_id"]),
            pred_name=p.get("pred_name", ""),
            count=int(p["count"]),
        )
        for p in payload.get("confusion_pairs", [])
    ]

    out = ConfusionOut(
        name=metric.name,
        tag=tag,
        top1=metric.top1,
        class_names=class_names,
        matrix=payload.get("confusion_matrix", []),
        per_class_acc=payload.get("per_class_acc", []),
        per_class_support=payload.get("per_class_support", []),
        top_pairs=pairs,
        figure=metrics_service.figure_url(metric.confusion_figure),
        num_samples=int(payload.get("num_samples") or 0),
    )
    return ok(out.model_dump())
