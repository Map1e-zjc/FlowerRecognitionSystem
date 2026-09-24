"""汇总各模型评估结果 → ml/outputs/metrics.json（后端 /api/v1/models 直接读取）。

用法
----
    python -m ml.export_metrics                       # 扫描 ml/outputs/logs/*_eval.json
    python -m ml.export_metrics --out ml/outputs/metrics.json

产出
----
    ml/outputs/metrics.json          4 个模型的指标汇总（docs/03 §4.2 结构）
    ml/outputs/figures/compare_models.png   指标对比柱状图（报告用）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.models import DEFAULTS
from ml.utils import Tee, load_json, save_json

MT_YAHEI = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

# 数据集元信息（写入 metrics.json，供前端展示与报告引用）
DATASET_INFO = {
    "name": "Oxford 102 Category Flower Dataset (Flowers-102)",
    "num_classes": 102,
    "train": 1020,
    "val": 1020,
    "test": 6149,
    "protocol": "P1-official-split",
}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = MT_YAHEI
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _discover(log_dir: Path) -> list[tuple[str, dict, dict | None]]:
    """返回 [(tag, eval_payload, summary_payload_or_None)]。"""
    found: list[tuple[str, dict, dict | None]] = []
    for ev in sorted(log_dir.glob("*_eval.json")):
        tag = ev.name[: -len("_eval.json")]
        payload = load_json(ev)
        if not payload:
            continue
        summary = load_json(log_dir / f"{tag}_summary.json")
        found.append((tag, payload, summary))
    return found


def plot_compare(models: list[dict], out_path: Path) -> None:
    plt = _setup_matplotlib()
    import numpy as np

    names = [m["display_name"] for m in models]
    top1 = [m.get("top1", 0) for m in models]
    top5 = [m.get("top5", 0) for m in models]
    params = [m.get("params_m", 0) for m in models]
    infer = [(m.get("timing") or {}).get("mean_ms") or 0 for m in models]

    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].bar(x, top1, color="#4C78A8")
    axes[0, 0].set_title("Top-1 准确率 (%)", fontsize=13)
    axes[0, 0].set_ylim(min(top1) - 3 if top1 else 0, 100)
    for i, v in enumerate(top1):
        axes[0, 0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=10)

    axes[0, 1].bar(x, top5, color="#54A24B")
    axes[0, 1].set_title("Top-5 准确率 (%)", fontsize=13)
    axes[0, 1].set_ylim(min(top5) - 1 if top5 else 0, 100)
    for i, v in enumerate(top5):
        axes[0, 1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)

    axes[1, 0].bar(x, params, color="#F58518")
    axes[1, 0].set_title("参数量 (M)", fontsize=13)
    for i, v in enumerate(params):
        axes[1, 0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    axes[1, 1].bar(x, infer, color="#E45756")
    axes[1, 1].set_title("单张推理耗时 (ms, GPU, batch=1)", fontsize=13)
    for i, v in enumerate(infer):
        axes[1, 1].text(i, v + max(infer + [1]) * 0.02, f"{v:.1f}", ha="center", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Flowers-102 多模型对比", fontsize=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="汇总模型指标 → metrics.json")
    ap.add_argument("--out-dir", default="ml/outputs", dest="out_dir")
    ap.add_argument("--out", default=None, help="输出路径（默认 <out-dir>/metrics.json）")
    ap.add_argument("--figures-only", action="store_true", help="只重画对比图")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    out_path = Path(args.out) if args.out else out_dir / "metrics.json"
    log = Tee(log_dir / "export_metrics.log")

    discovered = _discover(log_dir)
    if not discovered:
        log(f"✗ 未在 {log_dir} 找到任何 *_eval.json；请先运行 python -m ml.evaluate")
        log.close()
        return 1

    models: list[dict] = []
    for tag, ev, summary in discovered:
        timing = ev.get("timing") or {}
        models.append({
            "name": ev.get("model", tag),
            "tag": tag,
            "display_name": str(DEFAULTS.get(ev.get("model", ""), {}).get("display_name", tag)),
            "top1": ev.get("top1"),
            "top5": ev.get("top5"),
            "macro_f1": ev.get("macro_f1"),
            "weighted_f1": ev.get("weighted_f1"),
            "params_m": ev.get("params_m"),
            "infer_ms": timing.get("mean_ms"),
            "infer_p95_ms": timing.get("p95_ms"),
            "train_minutes": (summary or {}).get("train_minutes"),
            "epochs_run": (summary or {}).get("epochs_run"),
            "best_epoch": (summary or {}).get("best_epoch"),
            "image_size": ev.get("image_size"),
            "weights_path": ev.get("weights"),
            "split": ev.get("split"),
            "num_samples": ev.get("num_samples"),
            "worst_classes": ev.get("worst_classes", [])[:5],
            "confusion_pairs": ev.get("confusion_pairs", [])[:10],
            "figures": ev.get("figures", {}),
            "per_class_acc": ev.get("per_class_acc"),
            "per_class_support": ev.get("per_class_support"),
            "is_default": False,
        })

    models.sort(key=lambda m: -(m["top1"] or 0))
    if models:
        models[0]["is_default"] = True

    payload = {
        "dataset": DATASET_INFO,
        "protocol": DATASET_INFO["protocol"],
        "num_models": len(models),
        "best_model": models[0]["name"] if models else None,
        "best_tag": models[0]["tag"] if models else None,
        "models": models,
    }
    save_json(payload, out_path)

    fig = out_dir / "figures" / "compare_models.png"
    try:
        plot_compare(models, fig)
        log(f"对比图：{fig}")
    except Exception as exc:  # noqa: BLE001
        log(f"对比图绘制失败（不阻塞）：{type(exc).__name__}: {exc}")

    log(f"=== 已汇总 {len(models)} 个模型 ===")
    for m in models:
        log(f"  {m['display_name']:16s} Top-1={m['top1']:6.2f}%  Top-5={m['top5']:6.2f}%  "
            f"Macro-F1={m['macro_f1']:6.2f}%  参数={m['params_m']:5.1f}M  "
            f"推理={m['infer_ms']}ms  {'← best' if m['is_default'] else ''}")
    log(f"输出：{out_path}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
