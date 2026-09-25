"""通用分页响应模型。"""

from __future__ import annotations

import math
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PageOut(BaseModel, Generic[T]):
    items: list[T] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 10
    pages: int = 0

    @classmethod
    def build(cls, items: list[T], total: int, page: int, page_size: int) -> "PageOut[T]":
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=max(1, math.ceil(total / page_size)) if total else 0,
        )
