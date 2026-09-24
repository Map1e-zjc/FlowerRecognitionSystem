"""数据探索（EDA）：类别分布 + 每类样本可视化。

用法
----
    python -m ml.eda --data-root data/flowers-102 --out-dir ml/outputs

产出
----
    ml/outputs/figures/dataset_samples.png     102 类每类 1 张网格图（M1-7 判据）
    ml/outputs/figures/class_distribution.png  三个划分的类别分布
    ml/outputs/dataset_stats.json              统计摘要（写入报告）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.dataset import (
    NUM_CLASSES,
    SPLIT_SIZES,
    build_inference_transform,
    get_class_names,
    load_manifest,
)
from ml.utils import Tee, save_json

MT_YAHEI = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = MT_YAHEI
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flowers-102 数据探索")
    ap.add_argument("--data-root", default="data/flowers-102", dest="data_root")
    ap.add_argument("--out-dir", default="ml/outputs", dest="out_dir")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    log = Tee(out_dir / "logs" / "eda.log")
    plt = _setup_matplotlib()
    from PIL import Image

    manifest = load_manifest(args.data_root)
    names = get_class_names()
    log(f"=== EDA ===  数据源：{manifest.source}   根目录：{manifest.root}")
    if manifest.class_names_match is True:
        log("✓ 源侧类别名与内置 CLASS_NAMES_EN 逐项一致（标签顺序已确认）")
    elif manifest.class_names_match is False:
        log("✗ 源侧类别名与内置表不一致 —— 英文名/中文百科存在错位风险，请先核对！")
    else:
        log("… manifest 未记录源侧类别名，改为依赖下方样本图人工确认")

    # ---------- 统计 ----------
    counts = {split: {} for split in SPLIT_SIZES}
    for split, files in manifest.splits.items():
        for f in files:
            lbl = manifest.labels[f]
            counts[split][lbl] = counts[split].get(lbl, 0) + 1

    stats = {
        "source": manifest.source,
        "num_classes": NUM_CLASSES,
        "class_names_match_source": manifest.class_names_match,
        "totals": {s: len(manifest.splits.get(s, [])) for s in SPLIT_SIZES},
        "expected": SPLIT_SIZES,
        "per_class": {
            split: {
                "min": min(counts[split].values()),
                "max": max(counts[split].values()),
                "mean": round(sum(counts[split].values()) / len(counts[split]), 2),
            }
            for split in SPLIT_SIZES
        },
        "classes_present": {s: len(counts[s]) for s in SPLIT_SIZES},
    }
    for split in SPLIT_SIZES:
        pc = stats["per_class"][split]
        log(f"{split:5s}: {stats['totals'][split]:5d} 张  "
            f"类别数 {stats['classes_present'][split]:3d}  "
            f"每类 {pc['min']}—{pc['max']} 张（均值 {pc['mean']}）")

    # ---------- 每类样本网格 ----------
    transform = build_inference_transform(224)
    cols = 10
    rows = (NUM_CLASSES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.15))

    # 建立 类别 → 该类的代表文件名
    class_to_file: dict[int, str] = {}
    for f in manifest.splits["train"]:
        lbl = manifest.labels[f]
        class_to_file.setdefault(lbl, f)

    undecoded = transform.transforms[-1]  # Normalize
    mean = undecoded.mean
    std = undecoded.std

    jpg_dir = manifest.root / manifest.image_dir
    for cid in range(NUM_CLASSES):
        ax = axes.ravel()[cid]
        fname = class_to_file.get(cid)
        if fname is None:
            ax.axis("off")
            ax.set_title(f"{cid}: 缺失", fontsize=7)
            continue
        with Image.open(jpg_dir / fname) as im:
            rgb = im.convert("RGB")
            arr = transform(rgb).permute(1, 2, 0).numpy()
        arr = arr * std + mean  # 反归一化，还原可视
        ax.imshow(arr.clip(0, 1))
        ax.axis("off")
        ax.set_title(f"{cid}: {names[cid][:22]}", fontsize=6.5)

    for ax in axes.ravel()[NUM_CLASSES:]:
        ax.axis("off")

    fig.suptitle("Flowers-102 每类样本示例（train 划分，每类 1 张）", fontsize=15)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    samples_path = fig_dir / "dataset_samples.png"
    fig.savefig(samples_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"每类样本网格：{samples_path}")

    # ---------- 类别分布 ----------
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.6), sharey=False)
    for ax, split in zip(axes, ["train", "val", "test"]):
        vals = [counts[split].get(c, 0) for c in range(NUM_CLASSES)]
        ax.bar(range(NUM_CLASSES), vals, color="#4C78A8", width=0.85)
        ax.set_title(f"{split} 划分（n={sum(vals)}，每类 {min(vals)}—{max(vals)}）", fontsize=12)
        ax.set_xlabel("类别 ID")
        ax.set_ylabel("样本数")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Flowers-102 三个划分的类别分布", fontsize=15)
    fig.tight_layout()
    dist_path = fig_dir / "class_distribution.png"
    fig.savefig(dist_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"类别分布图：{dist_path}")

    save_json(stats, out_dir / "dataset_stats.json")
    log(f"统计摘要：{out_dir / 'dataset_stats.json'}")
    log("=== EDA 完成 ===")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
