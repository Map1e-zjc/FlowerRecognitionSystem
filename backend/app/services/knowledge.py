"""知识库服务：花卉百科的查询封装（供识别接口与百科接口共用）。"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import Flower


def flowers_map(db: Session, class_ids: Iterable[int]) -> dict[int, Flower]:
    """批量取类别，返回 ``{class_id: Flower}``。

    识别接口用它把 Top-5 的 class_id 一次性映射成中英文名，避免 N 次查询。
    """
    ids = list({int(c) for c in class_ids})
    if not ids:
        return {}
    rows = db.scalars(select(Flower).where(Flower.id.in_(ids))).all()
    return {f.id: f for f in rows}


def get_flower(db: Session, class_id: int) -> Flower | None:
    return db.get(Flower, class_id)


def list_flowers(
    db: Session,
    *,
    keyword: str | None = None,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[Flower], int]:
    """按中英文名模糊搜索并分页，返回 ``(当前页数据, 总数)``。"""
    stmt = select(Flower)
    count_stmt = select(func.count()).select_from(Flower)

    if keyword:
        kw = f"%{keyword.strip()}%"
        cond = or_(
            Flower.name_cn.like(kw),
            Flower.name_en.like(kw),
            Flower.family.like(kw),
            Flower.genus.like(kw),
        )
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    total = db.scalar(count_stmt) or 0
    rows = db.scalars(
        stmt.order_by(Flower.id).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return list(rows), int(total)
