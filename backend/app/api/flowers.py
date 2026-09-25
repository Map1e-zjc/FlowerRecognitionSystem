"""花卉百科接口（FR-02）。无需登录。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.errors import APIError, Envelope, ErrorCode, ok
from app.db.session import get_db
from app.schemas.common import PageOut
from app.schemas.flower import FlowerBrief, FlowerDetail
from app.services.knowledge import get_flower, list_flowers

router = APIRouter()


@router.get(
    "",
    response_model=Envelope[PageOut[FlowerBrief]],
    summary="花卉列表 / 搜索（分页）",
)
def list_flower_api(
    keyword: str | None = Query(None, description="按中文名 / 英文名 / 科 / 属模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict:
    rows, total = list_flowers(db, keyword=keyword, page=page, page_size=page_size)
    page_out = PageOut.build(
        [FlowerBrief.of(r) for r in rows], total=total, page=page, page_size=page_size
    )
    return ok(page_out.model_dump())


@router.get(
    "/{class_id}",
    response_model=Envelope[FlowerDetail],
    summary="花卉详情",
    responses={404: {"description": "类别不存在"}},
)
def get_flower_api(class_id: int, db: Session = Depends(get_db)) -> dict:
    if not 0 <= class_id <= 101:
        raise APIError(
            ErrorCode.NOT_FOUND, "类别 ID 需在 0—101 之间", http_status=404
        )
    flower = get_flower(db, class_id)
    if flower is None:
        raise APIError(ErrorCode.NOT_FOUND, "未找到该花卉资料", http_status=404)
    return ok(FlowerDetail.of(flower).model_dump())
