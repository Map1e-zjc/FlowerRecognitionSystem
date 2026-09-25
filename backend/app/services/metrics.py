"""指标数据读取：训练曲线 CSV、混淆矩阵与指标 JSON。

后端只读这些**离线训练产物**，不参与训练（docs/02 §1 关键运行原则）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings
from app.core.logging import get_logger

log = get_logger("metrics")

# 训练日志目录由 ml.train 写入；与 ml/outputs/metrics.json 同级
LOG_DIR = PROJECT_ROOT / "ml" / "outputs" / "logs"
FIGURES_DIR = PROJECT_ROOT / "ml" / "outputs" / "figures"


def read_metrics_json() -> dict:
    path: Path = settings.metrics_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("读取 metrics.json 失败：%s", exc)
        return {}


def read_curves(tag: str, *, max_points: int = 500) -> list[dict]:
    """读取 ``ml/outputs/logs/{tag}_train.csv``（逐 epoch 指标）。"""
    path = LOG_DIR / f"{tag}_train.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                point: dict = {}
                for key, raw in row.items():
                    if raw is None or raw == "":
                        point[key] = None
                        continue
                    try:
                        point[key] = int(raw) if key == "epoch" else float(raw)
                    except ValueError:
                        point[key] = None
                rows.append(point)
    except OSError as exc:
        log.warning("读取曲线 %s 失败：%s", path, exc)
        return []
    return rows[-max_points:]


def read_eval(tag: str) -> dict:
    """读取 ``ml/outputs/logs/{tag}_eval.json``（含 102×102 混淆矩阵）。"""
    path = LOG_DIR / f"{tag}_eval.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("读取 %s 失败：%s", path, exc)
        return {}


def figure_url(path_or_name: str) -> str:
    """把图表的文件路径转成可直接访问的 URL（供前端 <img> 使用）。"""
    if not path_or_name:
        return ""
    name = Path(path_or_name).name
    return f"/figures/{name}"
