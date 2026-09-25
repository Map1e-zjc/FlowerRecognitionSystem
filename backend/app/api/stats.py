"""统计概览接口（FR-05）。必须登录，统计范围限当前用户。"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.errors import Envelope, ok
from app.db.session import get_db
from app.models import Flower, RecognitionRecord, User
from app.schemas.model_metric import (
    ConfidenceBucket,
    DailyPoint,
    StatsOverview,
    TopClass,
)

router = APIRouter()

# 置信度分档（用于前端饼图/柱状图）
_BUCKETS: list[tuple[str, float, float]] = [
    ("<50%", 0.0, 0.5),
    ("50—70%", 0.5, 0.7),
    ("70—85%", 0.7, 0.85),
    ("85—95%", 0.85, 0.95),
    ("95—100%", 0.95, 1.01),
]


@router.get("/overview", response_model=Envelope[StatsOverview], summary="识别统计概览")
def overview(
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    mine = RecognitionRecord.user_id == current.id

    total = int(db.scalar(select(func.count()).select_from(RecognitionRecord).where(mine)) or 0)
    unique_classes = int(
        db.scalar(select(func.count(func.distinct(RecognitionRecord.predicted_class_id))).where(mine))
        or 0
    )
    avg_conf = db.scalar(select(func.avg(RecognitionRecord.confidence)).where(mine))
    avg_confidence = round(float(avg_conf or 0) * 100, 2)

    # 今日次数
    today_start = datetime.combine(date.today(), datetime.min.time())
    today = int(
        db.scalar(
            select(func.count())
            .select_from(RecognitionRecord)
            .where(mine, RecognitionRecord.created_at >= today_start)
        )
        or 0
    )

    # 置信度分档：用一条 CASE + GROUP BY，避免把全部记录拉进内存
    bucket_expr = case(
        *[(RecognitionRecord.confidence < hi, label) for label, _lo, hi in _BUCKETS[:-1]],
        else_=_BUCKETS[-1][0],
    )
    bucket_rows = db.execute(
        select(bucket_expr.label("bucket"), func.count())
        .where(mine)
        .group_by(bucket_expr)
    ).all()
    counts = {str(b): int(c) for b, c in bucket_rows}
    confidence_buckets = [
        ConfidenceBucket(label=label, count=counts.get(label, 0)) for label, _lo, _hi in _BUCKETS
    ]

    # 最近 7 天趋势
    week_ago = datetime.combine(date.today() - timedelta(days=6), datetime.min.time())
    day_expr = func.date(RecognitionRecord.created_at)
    day_rows = db.execute(
        select(day_expr.label("d"), func.count()).where(mine, RecognitionRecord.created_at >= week_ago).group_by(day_expr)
    ).all()
    per_day = {str(d): int(c) for d, c in day_rows}
    daily_trend = []
    for offset in range(6, -1, -1):
        day = (date.today() - timedelta(days=offset)).isoformat()
        daily_trend.append(DailyPoint(date=day, count=per_day.get(day, 0)))

    # 识别最多的 10 类
    top_rows = db.execute(
        select(RecognitionRecord.predicted_class_id, func.count().label("c"))
        .where(mine)
        .group_by(RecognitionRecord.predicted_class_id)
        .order_by(func.count().desc())
        .limit(10)
    ).all()
    id_to_flower = {
        f.id: f
        for f in db.scalars(
            select(Flower).where(Flower.id.in_([int(r[0]) for r in top_rows] or [-1]))
        ).all()
    }
    top_classes = [
        TopClass(
            class_id=int(cid),
            name_cn=(id_to_flower[cid].name_cn if cid in id_to_flower else "未知"),
            name_en=(id_to_flower[cid].name_en if cid in id_to_flower else ""),
            count=int(cnt),
        )
        for cid, cnt in top_rows
    ]

    out = StatsOverview(
        total=total,
        today=today,
        unique_classes=unique_classes,
        avg_confidence=avg_confidence,
        confidence_buckets=confidence_buckets,
        daily_trend=daily_trend,
        top_classes=top_classes,
    )
    return ok(out.model_dump())
