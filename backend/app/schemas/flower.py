"""花卉百科的响应模型。

字段名与前端契约一致（``class_id`` 而非 ORM 的 ``id``）。
用显式 ``of()`` 从 ORM 对象构造，避免依赖 pydantic 的 alias 语义（易错且隐晦）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FlowerBrief(BaseModel):
    """列表项：够卡片展示即可。"""

    class_id: int
    name_cn: str
    name_en: str
    category_cn: str = ""
    family: str = ""
    genus: str = ""
    color: str = ""
    bloom_season: str = ""
    sample_image: str = ""

    @classmethod
    def of(cls, f: Any) -> "FlowerBrief":
        return cls(
            class_id=f.id,
            name_cn=f.name_cn,
            name_en=f.name_en,
            category_cn=f.category_cn or "",
            family=f.family or "",
            genus=f.genus or "",
            color=f.color or "",
            bloom_season=f.bloom_season or "",
            sample_image=f.sample_image or "",
        )


class FlowerDetail(FlowerBrief):
    """详情：追加养护要点。空字段由前端显示"暂无资料"。"""

    category: str = ""
    light: str = ""
    watering: str = ""
    soil: str = ""
    propagation: str = ""
    description: str = ""
    care_tips: str = ""

    @classmethod
    def of(cls, f: Any) -> "FlowerDetail":  # type: ignore[override]
        return cls(
            class_id=f.id,
            name_cn=f.name_cn,
            name_en=f.name_en,
            category=f.category or "",
            category_cn=f.category_cn or "",
            family=f.family or "",
            genus=f.genus or "",
            color=f.color or "",
            bloom_season=f.bloom_season or "",
            light=f.light or "",
            watering=f.watering or "",
            soil=f.soil or "",
            propagation=f.propagation or "",
            description=f.description or "",
            care_tips=f.care_tips or "",
            sample_image=f.sample_image or "",
        )


class FlowerRecommendation(BaseModel):
    """识别结果中用于跳转百科的最小信息。"""

    class_id: int = Field(..., description="类别 ID")
    encyclopedia_url: str = Field(..., description="百科详情页前端路由")
