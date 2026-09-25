"""建表与种子数据导入（幂等，可重复执行）。

用法
----
    python -m app.db.init_db            # 建表 + 导入种子（已存在则更新）
    python -m app.db.init_db --drop     # 先删表再重建（会清空用户与历史记录）
    python -m app.db.init_db --check     # 只打印当前数据状态

种子来源
--------
* ``backend/app/data/flowers.json``（102 条花卉百科，由 ``ml.build_flowers_json`` 生成）
* ``ml/outputs/metrics.json``（4 个模型的测试集指标，由 ``ml.export_metrics`` 生成）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import Base, SessionLocal, engine
from app.models import Flower, ModelMetric, RecognitionRecord, User  # noqa: F401 (建表需要)

log = get_logger("init_db")

# 预训练来源（metrics.json 里没有，按模型名补全，供前端展示）
PRETRAINED = {
    "resnet50": "torchvision IMAGENET1K_V2（ImageNet-1k）",
    "efficientnet_b0": "timm tf_efficientnet_b0.ns_jft_in1k（NoisyStudent JFT→IN-1k）",
    "convnext_tiny": "timm convnext_tiny.fb_in22k_ft_in1k（ImageNet-22k→1k）",
    "vit_b16": "timm vit_base_patch16_224.augreg_in21k_ft_in1k（AugReg IN-21k→1k）",
}


def create_tables(drop: bool = False) -> None:
    if drop:
        log.warning("--drop：删除全部表（用户与识别历史将丢失）")
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    tables = ", ".join(sorted(Base.metadata.tables))
    log.info("表已就绪：%s", tables)


def seed_flowers(db: Session) -> int:
    path: Path = settings.flowers_path
    if not path.exists():
        raise FileNotFoundError(
            f"未找到百科数据 {path}。请先执行：python -m ml.build_flowers_json --export-thumbs"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    existing = {f.id: f for f in db.scalars(select(Flower)).all()}

    created = updated = 0
    for r in records:
        cid = int(r["class_id"])
        payload = {
            "name_en": r.get("name_en", ""),
            "name_cn": r.get("name_cn", ""),
            "category": r.get("category", ""),
            "category_cn": r.get("category_cn", ""),
            "family": r.get("family", ""),
            "genus": r.get("genus", ""),
            "bloom_season": r.get("bloom_season", ""),
            "color": r.get("color", ""),
            "light": r.get("light", ""),
            "watering": r.get("watering", ""),
            "soil": r.get("soil", ""),
            "propagation": r.get("propagation", ""),
            "description": r.get("description", ""),
            "care_tips": r.get("care_tips", ""),
            "sample_image": r.get("sample_image", ""),
        }
        obj = existing.get(cid)
        if obj is None:
            db.add(Flower(id=cid, **payload))
            created += 1
        else:
            for k, v in payload.items():
                setattr(obj, k, v)
            updated += 1
    db.commit()
    log.info("花卉百科：新增 %d 条，更新 %d 条，共 %d 条", created, updated, len(records))
    return len(records)


def seed_model_metrics(db: Session) -> int:
    path: Path = settings.metrics_path
    if not path.exists():
        log.warning(
            "未找到 %s（跳过模型指标导入）。请先执行：python -m ml.export_metrics", path
        )
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    existing = {m.name: m for m in db.scalars(select(ModelMetric)).all()}

    for m in models:
        figures = m.get("figures") or {}
        values = {
            "tag": m.get("tag", m.get("name", "")),
            "display_name": m.get("display_name", m.get("name", "")),
            "pretrained": PRETRAINED.get(m.get("name", ""), ""),
            "top1": float(m.get("top1") or 0.0),
            "top5": float(m.get("top5") or 0.0),
            "macro_f1": float(m.get("macro_f1") or 0.0),
            "params_m": float(m.get("params_m") or 0.0),
            "infer_ms": float(m.get("infer_ms") or 0.0),
            "infer_p95_ms": float(m.get("infer_p95_ms") or 0.0),
            "train_minutes": float(m.get("train_minutes") or 0.0),
            "epochs_run": int(m.get("epochs_run") or 0),
            "best_epoch": int(m.get("best_epoch") or 0),
            "image_size": int(m.get("image_size") or 224),
            "num_samples": int(m.get("num_samples") or 0),
            "is_default": bool(m.get("is_default")),
            "weights_path": m.get("weights_path") or "",
            "confusion_figure": figures.get("confusion", ""),
            "curves_figure": "",  # 训练曲线由日志 CSV 在线生成（M3-7）
            "error_cases_figure": figures.get("error_cases", ""),
        }
        obj = existing.get(m["name"])
        if obj is None:
            db.add(ModelMetric(name=m["name"], **values))
        else:
            for k, v in values.items():
                setattr(obj, k, v)
    db.commit()
    log.info("模型指标：导入 %d 个模型（best=%s）", len(models), payload.get("best_model"))
    return len(models)


def check(db: Session) -> None:
    log.info(
        "当前数据状态：users=%d, flowers=%d, records=%d, model_metrics=%d",
        db.query(User).count(),
        db.query(Flower).count(),
        db.query(RecognitionRecord).count(),
        db.query(ModelMetric).count(),
    )
    for m in db.scalars(select(ModelMetric).order_by(ModelMetric.top1.desc())).all():
        log.info("  %-16s Top-1=%.2f%%  Top-5=%.2f%%  %s",
                 m.display_name, m.top1, m.top5, "← 默认" if m.is_default else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建表与种子数据导入（幂等）")
    ap.add_argument("--drop", action="store_true", help="先删表再重建（清空用户与历史）")
    ap.add_argument("--check", action="store_true", help="只打印当前数据状态")
    args = ap.parse_args(argv)

    with SessionLocal() as db:
        if args.check:
            check(db)
            return 0
        create_tables(drop=args.drop)
        seed_flowers(db)
        seed_model_metrics(db)
        check(db)
    log.info("数据库：%s", settings.sqlalchemy_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
