"""识别历史接口（FR-03）。必须登录，且只能访问自己的数据。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.errors import APIError, Envelope, ErrorCode, ok
from app.core.logging import get_logger
from app.db.session import get_db
from app.models import RecognitionRecord, User
from app.schemas.common import PageOut
from app.schemas.history import HistoryDetail, HistoryItem
from app.schemas.predict import PredictionItem

log = get_logger("history")
router = APIRouter()


def _image_url(record: RecognitionRecord) -> str:
    return f"/uploads/{record.image_path}"


def _to_item(record: RecognitionRecord) -> HistoryItem:
    flower = record.flower
    return HistoryItem(
        id=record.id,
        image_url=_image_url(record),
        class_id=record.predicted_class_id,
        name_cn=(flower.name_cn if flower else "未知"),
        name_en=(flower.name_en if flower else ""),
        confidence=round(record.confidence, 4),
        model_name=record.model_name,
        latency_ms=record.latency_ms,
        created_at=record.created_at,
    )


@router.get(
    "",
    response_model=Envelope[PageOut[HistoryItem]],
    summary="我的识别历史（分页 / 按种类筛选）",
)
def list_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    class_id: int | None = Query(None, ge=0, le=101, description="只看某个花卉类别"),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    cond = [RecognitionRecord.user_id == current.id]
    if class_id is not None:
        cond.append(RecognitionRecord.predicted_class_id == class_id)

    total = db.scalar(
        select(func.count()).select_from(RecognitionRecord).where(*cond)
    ) or 0

    rows = db.scalars(
        select(RecognitionRecord)
        .options(joinedload(RecognitionRecord.flower))  # 避免 N+1 查询
        .where(*cond)
        .order_by(RecognitionRecord.created_at.desc(), RecognitionRecord.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    page_out = PageOut.build(
        [_to_item(r) for r in rows], total=int(total), page=page, page_size=page_size
    )
    return ok(page_out.model_dump())


def _get_own_record(db: Session, record_id: int, user: User) -> RecognitionRecord:
    """取自己的记录；不存在或不属于自己都返回 404（不泄露资源是否存在）。"""
    record = db.get(RecognitionRecord, record_id)
    if record is None or record.user_id != user.id:
        raise APIError(ErrorCode.NOT_FOUND, "未找到该识别记录", http_status=404)
    return record


@router.get(
    "/{record_id}",
    response_model=Envelope[HistoryDetail],
    summary="识别记录详情（含完整 Top-5）",
)
def get_history(
    record_id: int = Path(..., ge=1),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    record = _get_own_record(db, record_id, current)
    base = _to_item(record)
    try:
        raw = json.loads(record.top5_json or "[]")
    except ValueError:
        raw = []
    detail = HistoryDetail(
        **base.model_dump(),
        top5=[PredictionItem(**p) for p in raw if isinstance(p, dict)],
    )
    return ok(detail.model_dump(mode="json"))


@router.delete(
    "/{record_id}",
    response_model=Envelope[dict],
    summary="删除一条识别记录",
)
def delete_history(
    record_id: int = Path(..., ge=1),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    record = _get_own_record(db, record_id, current)

    # 同时删除磁盘上的原图，避免 uploads 目录无限增长
    image_path = settings.upload_path / record.image_path
    try:
        if image_path.is_file():
            image_path.unlink()
    except OSError as exc:  # 删文件失败不应阻断删除记录
        log.warning("删除图片失败（记录仍会删除）：%s → %s", image_path, exc)

    db.delete(record)
    db.commit()
    log.info("删除识别记录：user=%s record=%s", current.username, record_id)
    return ok({"record_id": record_id, "deleted": True}, message="已删除")
