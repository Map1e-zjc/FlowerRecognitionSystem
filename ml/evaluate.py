"""测试集评估：指标、混淆矩阵、错误案例、推理耗时。

用法
----
    python -m ml.evaluate --weights ml/outputs/checkpoints/vit_b16_best.pth
    python -m ml.evaluate --weights ... --split val --out-dir ml/outputs

产出
----
    ml/outputs/figures/{tag}_confusion.png     102×102 混淆矩阵热力图
    ml/outputs/figures/{tag}_per_class.png     每类准确率分布
    ml/outputs/figures/{tag}_error_cases.png   高置信度错分样本拼图
    ml/outputs/figures/error_cases/{tag}/      错分样本原图
    ml/outputs/logs/{tag}_eval.json            指标 + 混淆矩阵 + 混淆对（供 export_metrics 汇总）
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.dataset import NUM_CLASSES, build_dataloader, get_class_names
from ml.models import build_model
from ml.utils import (
    Tee,
    accuracy_topk,
    device_summary,
    ensure_cache_env,
    make_bar,
    resolve_device,
    save_json,
    set_progress_mode,
)

MT_YAHEI = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = MT_YAHEI
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# --------------------------------------------------------------------------
# 加载权重
# --------------------------------------------------------------------------
def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt["model_name"]
    image_size = int(ckpt.get("image_size", 224))
    model = build_model(name, int(ckpt.get("num_classes", NUM_CLASSES)), pretrained=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[evaluate] 警告：missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"[evaluate]   missing 示例：{missing[:3]}")
        if unexpected[:3]:
            print(f"[evaluate]   unexpected 示例：{unexpected[:3]}")
    model.to(device).eval()
    return model, name, image_size, ckpt


# --------------------------------------------------------------------------
# 全量推理
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device, *, use_amp: bool):
    """返回 (logits, targets) 两个 numpy 数组。"""
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    bar = make_bar(loader, total=len(loader), desc="推理")
    for images, targets in bar:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.numpy())
    bar.close()
    return np.concatenate(all_logits), np.concatenate(all_targets)


@torch.no_grad()
def measure_latency(model, device, image_size: int, *, warmup: int = 10, runs: int = 100) -> dict:
    """单张推理耗时（batch=1，含预处理与否都说明清楚：这里只测前向）。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.asarray(times)
    return {
        "device": device_summary(device),
        "batch_size": 1,
        "runs": runs,
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
    }


# --------------------------------------------------------------------------
# 图表
# --------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt = _setup_matplotlib()
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(26, 22))
    sns.heatmap(
        cm, ax=ax, cmap="Blues", square=True, cbar_kws={"shrink": 0.6, "label": "样本数"},
        xticklabels=5, yticklabels=5, linewidths=0,
    )
    ax.set_xlabel("预测类别（每 5 类一个刻度）", fontsize=13)
    ax.set_ylabel("真实类别（每 5 类一个刻度）", fontsize=13)
    ax.set_title(title, fontsize=16, pad=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(per_class: np.ndarray, support: np.ndarray, out_path: Path, title: str) -> None:
    plt = _setup_matplotlib()

    order = np.argsort(per_class)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].bar(range(NUM_CLASSES), per_class[order] * 100, color="#4C78A8")
    axes[0].set_xlabel("类别（按准确率升序）")
    axes[0].set_ylabel("准确率 (%)")
    axes[0].set_title("每类准确率（升序）", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(support, per_class * 100, s=18, alpha=0.7, color="#E45756")
    axes[1].set_xlabel("该类测试样本数")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("样本数与准确率的关系", fontsize=13)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_cases(records: list[dict], out_path: Path, title: str) -> None:
    """records: [{path, true_name, pred_name, confidence, true_id, pred_id}]"""
    if not records:
        return
    from PIL import Image

    plt = _setup_matplotlib()
    n = len(records)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, rec in zip(axes, records):
        with Image.open(rec["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.axis("off")
        ax.set_title(
            f"真：{rec['true_name']}\n判：{rec['pred_name']} ({rec['confidence'] * 100:.1f}%)",
            fontsize=9,
        )
    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flowers-102 模型评估")
    ap.add_argument("--weights", type=Path, required=True, help="checkpoint 路径（如 ..._best.pth）")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-root", default="data/flowers-102")
    ap.add_argument("--out-dir", default="ml/outputs", dest="out_dir")
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None, help="输出前缀（默认取权重文件名前缀）")
    ap.add_argument("--no-timing", action="store_true", help="跳过推理耗时测量")
    ap.add_argument("--top-errors", type=int, default=20, dest="top_errors")
    ap.add_argument("--progress", dest="progress", action="store_true", default=None,
                    help="强制显示进度条（默认 auto：仅终端下显示）")
    ap.add_argument("--no-progress", dest="progress", action="store_false",
                    help="强制关闭进度条（后台作业推荐）")
    args = ap.parse_args(argv)

    # 缓存目录兜底：必须在加载权重之前调用
    ensure_cache_env()

    if args.progress is not None:
        set_progress_mode("1" if args.progress else "0")

    tag = args.tag or args.weights.stem.replace("_best", "").replace("_last", "")
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    log = Tee(log_dir / f"{tag}_eval.log")

    device = resolve_device(args.device)
    log(f"=== 评估 {tag} ===")
    log(f"权重：{args.weights}")
    log(f"设备：{device_summary(device)}")

    model, model_name, image_size, ckpt = load_checkpoint(args.weights, device)
    log(f"模型：{model_name}  image_size={image_size}  "
        f"训练时最佳 val Top-1={ckpt.get('best_top1', float('nan')):.2f}%")

    loader = build_dataloader(
        args.data_root, args.split, batch_size=args.batch_size,
        image_size=image_size, num_workers=args.num_workers, shuffle=False,
    )
    log(f"评估集：{args.split}（{len(loader.dataset)} 张）")

    logits, targets = collect_predictions(model, loader, device, use_amp=device.type == "cuda")

    # ---- 指标 ----
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    preds = logits.argmax(axis=1)
    acc = float((preds == targets).mean() * 100)
    top1, top5 = accuracy_topk(torch.from_numpy(logits), torch.from_numpy(targets), (1, 5))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0) * 100)
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0) * 100)

    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    support = cm.sum(axis=1)
    per_class = np.divide(
        cm.diagonal(), support, out=np.zeros(NUM_CLASSES), where=support > 0
    )

    log(f"Top-1 = {top1:.2f}%   Top-5 = {top5:.2f}%")
    log(f"Macro-F1 = {macro_f1:.2f}%   Weighted-F1 = {weighted_f1:.2f}%   准确率(acc) = {acc:.2f}%")

    names = get_class_names()
    worst = np.argsort(per_class)[:10]
    log("最差 10 个类别（准确率升序）：")
    for cid in worst:
        log(f"  class {cid:3d} {names[cid]:28s} acc={per_class[cid] * 100:5.1f}%  n={int(support[cid])}")

    # ---- 混淆对 ----
    np.fill_diagonal(cm, 0)
    flat = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0]
    pairs = [
        {
            "true_id": int(t), "true_name": names[int(t)],
            "pred_id": int(p), "pred_name": names[int(p)],
            "count": int(cm[t, p]),
        }
        for t, p in flat[:15] if cm[t, p] > 0
    ]
    log("Top 混淆对（真实 → 误判）：")
    for pr in pairs[:10]:
        log(f"  {pr['true_name']:28s} → {pr['pred_name']:28s} {pr['count']:4d} 次")

    # ---- 图表 ----
    plot_confusion(
        cm, fig_dir / f"{tag}_confusion.png",
        f"{tag} 混淆矩阵（{args.split}，{len(targets)} 张，Top-1={top1:.2f}%）",
    )
    plot_per_class(
        per_class, support, fig_dir / f"{tag}_per_class.png",
        f"{tag} 每类表现（Macro-F1={macro_f1:.2f}%）",
    )
    log(f"混淆矩阵图：{fig_dir / f'{tag}_confusion.png'}")

    # ---- 错误案例（高置信度错分）----
    wrong = np.where(preds != targets)[0]
    conf = logits.max(axis=1)
    wrong = wrong[np.argsort(-conf[wrong])][: args.top_errors]
    ds = loader.dataset
    err_dir = fig_dir / "error_cases" / tag
    err_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i in wrong:
        src = ds.samples[int(i)].path
        dst = err_dir / f"true{int(targets[i]):03d}_pred{int(preds[i]):03d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        records.append({
            "path": str(src),
            "true_id": int(targets[i]),
            "pred_id": int(preds[i]),
            "true_name": names[int(targets[i])],
            "pred_name": names[int(preds[i])],
            "confidence": round(float(conf[i]), 4),
        })
    plot_error_cases(
        records, fig_dir / f"{tag}_error_cases.png",
        f"{tag} 高置信度错分案例（Top {len(records)}）",
    )
    log(f"错误案例拼图：{fig_dir / f'{tag}_error_cases.png'}（原图目录 {err_dir}）")

    # ---- 推理耗时 ----
    timing = None
    if not args.no_timing:
        timing = measure_latency(model, device, image_size)
        log(f"单张推理耗时：mean={timing['mean_ms']} ms  median={timing['median_ms']} ms  "
            f"p95={timing['p95_ms']} ms（{timing['device']}）")

    # ---- 落盘 ----
    payload = {
        "tag": tag,
        "model": model_name,
        "weights": str(args.weights),
        "split": args.split,
        "num_samples": int(len(targets)),
        "image_size": image_size,
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(acc, 4),
        "train_minutes": (ckpt.get("config", {}) or {}).get("_train_minutes"),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "timing": timing,
        "worst_classes": [
            {"class_id": int(c), "name": names[int(c)],
             "acc": round(float(per_class[c]) * 100, 2), "support": int(support[c])}
            for c in worst
        ],
        "confusion_pairs": pairs,
        "per_class_acc": [round(float(v) * 100, 2) for v in per_class],
        "per_class_support": [int(v) for v in support],
        "confusion_matrix": cm.tolist(),
        "error_cases": records,
        "figures": {
            "confusion": str(fig_dir / f"{tag}_confusion.png"),
            "per_class": str(fig_dir / f"{tag}_per_class.png"),
            "error_cases": str(fig_dir / f"{tag}_error_cases.png"),
        },
        "classification_report": classification_report(
            targets, preds, labels=list(range(NUM_CLASSES)),
            target_names=[f"{i:03d}_{n}" for i, n in enumerate(names)],
            zero_division=0, output_dict=True,
        ),
    }
    out_json = log_dir / f"{tag}_eval.json"
    save_json(payload, out_json)
    log(f"评估结果：{out_json}")
    log("=" * 78)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
