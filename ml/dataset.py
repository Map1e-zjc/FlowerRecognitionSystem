"""数据管道：Oxford Flowers-102 数据集、增强管道、DataLoader 工厂。

关键约定（与推理侧共享，务必遵守）
----------------------------------
1. 标签统一转为 **0-based**（0..101）。官方 ``imagelabels.mat`` 是 1-based。
2. 只有一份预处理定义 :func:`build_transforms`，训练与后端推理必须共用，
   避免出现「报告 96%、线上 70%」的精度落差。
3. 归一化使用 ImageNet 统计量（预训练主干要求）。
4. 所有随机性由 ``--seed`` 固定，DataLoader 使用显式 generator 保证可复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

Split = Literal["train", "val", "test"]

# ImageNet 归一化参数（预训练主干要求，不可随意更改）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_ROOT = Path("data/flowers-102")
NUM_CLASSES = 102
SPLIT_SIZES = {"train": 1020, "val": 1020, "test": 6149}

MANIFEST_NAME = "manifest.json"

# 标签顺序校验提示只打印一次（load_manifest 会被多次调用）
_NAME_ORDER_NOTICE_SHOWN = False


# --------------------------------------------------------------------------
# 清单（manifest.json）—— 划分与标签的唯一权威来源
# --------------------------------------------------------------------------
@dataclass
class Manifest:
    """数据集清单。

    ``splits`` 是 {划分: [图片文件名]}，``labels`` 是 {图片文件名: 0-based 标签}。
    由 ``python -m ml.download_data`` 生成，不管数据来自 HF 镜像还是官方 tgz，
    下游代码都只认这一份清单，因此换源不影响训练代码。
    """

    root: Path
    image_dir: str
    splits: dict[str, list[str]]
    labels: dict[str, int]
    source: str = "unknown"
    num_classes: int = NUM_CLASSES
    class_names_from_source: list[str] | None = None
    class_names_match: bool | None = None

    def names_for(self, split: Split) -> list[str]:
        if split not in self.splits:
            raise KeyError(f"manifest 中没有划分 {split!r}，实际：{sorted(self.splits)}")
        return self.splits[split]


def find_manifest_path(data_root: Path) -> Path:
    """兼容 ``data/flowers-102/manifest.json`` 与把 ``data`` 传进来的两种写法。"""
    for cand in (data_root / MANIFEST_NAME, data_root / "flowers-102" / MANIFEST_NAME):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"未找到 {MANIFEST_NAME}（已查找 {data_root} 及其 flowers-102 子目录）。\n"
        f"请先执行：python -m ml.download_data --root data"
    )


def load_manifest(data_root: str | Path = DEFAULT_DATA_ROOT) -> Manifest:
    path = find_manifest_path(Path(data_root))
    payload = json.loads(path.read_text(encoding="utf-8"))

    entries = payload.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} 缺少 'images' 列表，文件可能损坏，请重新生成清单")

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    labels: dict[str, int] = {}
    for e in entries:
        fname, lbl, split = e["f"], int(e["l"]), e["s"]
        if split not in splits:
            raise ValueError(f"{path} 中出现未知划分 {split!r}")
        splits[split].append(fname)
        labels[fname] = lbl

    for split, expect in SPLIT_SIZES.items():
        got = len(splits.get(split, []))
        if got != expect:
            raise AssertionError(
                f"manifest 中 {split} 划分有 {got} 张，期望 {expect} 张；"
                f"清单可能损坏，请重新执行 python -m ml.download_data --verify"
            )

    bad = [v for v in labels.values() if v < 0 or v >= NUM_CLASSES]
    if bad:
        raise ValueError(f"manifest 标签越界（示例 {bad[:5]}），期望范围 [0, {NUM_CLASSES - 1}]")

    src_names_raw = payload.get("class_names_from_source")
    src_names = [str(x) for x in src_names_raw] if src_names_raw else None

    match: bool | None = None
    if src_names:
        # 关键校验：源侧类别名顺序必须与项目内置 cat_to_name 表一致，
        # 否则所有中文名与百科信息都会错位到别的类别上（致命且难查）。
        #
        # 实测：hf-mirror 的 dpdl-benchmark 源只有 index 59 写成
        # "pink-yellow dahlia?"（多个问号），其余 101 项逐字相同 —— 属无害差异，
        # 因此比较前先做规范化，只在规范化后仍不同才报警。
        global _NAME_ORDER_NOTICE_SHOWN

        def _norm(xs: list[str]) -> list[str]:
            return [x.strip().rstrip("?").strip().lower().replace("_", " ") for x in xs]

        match = _norm(src_names) == _norm(CLASS_NAMES_EN)
        if not _NAME_ORDER_NOTICE_SHOWN:
            _NAME_ORDER_NOTICE_SHOWN = True
            raw_diffs = [i for i, (a, b) in enumerate(zip(src_names, CLASS_NAMES_EN)) if a != b]
            if match:
                if raw_diffs:
                    preview = ", ".join(
                        f"[{i}] 源={src_names[i]!r} 内置={CLASS_NAMES_EN[i]!r}"
                        for i in raw_diffs[:5]
                    )
                    print(
                        f"ℹ 源侧类别名与内置表规范化后一致（标签顺序正确），"
                        f"仅 {len(raw_diffs)} 处字面差异（无害）：{preview}"
                    )
                else:
                    print("ℹ 源侧类别名与内置 CLASS_NAMES_EN 完全逐字一致（102 项）")
            else:
                diffs = [
                    f"  [{i}] 源={a!r} 内置={b!r}"
                    for i, (a, b) in enumerate(zip(src_names, CLASS_NAMES_EN))
                    if _norm([a])[0] != _norm([b])[0]
                ][:10]
                print(
                    "⚠ 警告：manifest 记录的源侧类别名与 ml/dataset.py::CLASS_NAMES_EN 不一致！\n"
                    f"  源侧 {len(src_names)} 项 / 内置 {len(CLASS_NAMES_EN)} 项，前 10 处差异：\n"
                    + "\n".join(diffs)
                    + "\n  影响：英文名与中文百科可能错位。请核对后更新 CLASS_NAMES_EN。"
                )

    return Manifest(
        root=path.parent,
        image_dir=str(payload.get("image_dir", "jpg")),
        splits=splits,
        labels=labels,
        source=str(payload.get("source", "unknown")),
        num_classes=int(payload.get("num_classes", NUM_CLASSES)),
        class_names_from_source=src_names,
        class_names_match=match,
    )


# --------------------------------------------------------------------------
# 类别名称
# --------------------------------------------------------------------------
# Flowers-102 无官方英文名清单，使用社区通用的 102 类名称（1-based 顺序）。
# 该表在 M1-7 EDA 后由 get_class_names() 落盘为 class_names.json 供前端/百科复用。
CLASS_NAMES_EN: list[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


def get_class_names() -> list[str]:
    """返回 102 个英文类别名（0-based 顺序）。"""
    if len(CLASS_NAMES_EN) != NUM_CLASSES:
        raise AssertionError(
            f"类别名表长度 {len(CLASS_NAMES_EN)} != {NUM_CLASSES}，请检查 CLASS_NAMES_EN"
        )
    return list(CLASS_NAMES_EN)


def dump_class_names(path: Path) -> Path:
    """把类别名落盘为 JSON（0-based id → 英文名），供百科与前端使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"class_id": i, "name_en": n} for i, n in enumerate(get_class_names())]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 增强管道（训练与推理共享的唯一实现）
# --------------------------------------------------------------------------
def build_transforms(split: Split, image_size: int = 224, *, strong: bool = False):
    """构建指定划分的预处理/增强管道。

    Args:
        split: ``train`` 使用随机增强；``val``/``test`` 使用确定性的 resize+centercrop。
        image_size: 网络输入边长。
        strong: 训练时是否启用 RandAugment + RandomErasing（M2-7 调优回路第 1 步）。
    """
    if split == "train":
        ops: list = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
        ]
        if strong:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if strong:
            ops.append(transforms.RandomErasing(p=0.25, value="random"))
        return transforms.Compose(ops)

    # 验证 / 测试：确定性，推理侧必须与此完全一致
    resize_to = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_inference_transform(image_size: int = 224):
    """推理专用 transform —— 与 ``build_transforms('test')`` 完全一致。

    后端 ``app/services/inference.py`` 必须调用本函数，不得自行复制粘贴预处理代码。
    """
    return build_transforms("test", image_size)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class Sample:
    index: int      # 官方图片序号（1-based）
    label: int      # 0-based 类别
    path: Path


class Flowers102Dataset(Dataset):
    """Oxford Flowers-102 数据集（本地已下载的 jpg + .mat 标注）。"""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split: Split = "train",
        transform=None,
        image_size: int = 224,
        *,
        strong: bool = False,
        manifest: Manifest | None = None,
    ) -> None:
        if split not in SPLIT_SIZES:
            raise ValueError(f"split 必须是 {sorted(SPLIT_SIZES)} 之一，收到 {split!r}")
        self.split = split
        self.manifest = manifest if manifest is not None else load_manifest(data_root)
        self.data_root = self.manifest.root

        jpg_dir = self.data_root / self.manifest.image_dir
        if not jpg_dir.is_dir():
            raise FileNotFoundError(
                f"未找到图片目录 {jpg_dir}。请先执行：python -m ml.download_data --root data"
            )

        self.samples: list[Sample] = []
        missing: list[str] = []
        for i, fname in enumerate(self.manifest.names_for(split)):
            img_path = jpg_dir / fname
            if not img_path.exists():
                missing.append(fname)
                continue
            self.samples.append(
                Sample(index=i, label=self.manifest.labels[fname], path=img_path)
            )

        if missing:
            raise FileNotFoundError(
                f"{split} 划分有 {len(missing)} 张图片缺失（例如 {missing[:3]}）。\n"
                f"数据集不完整，请执行：python -m ml.download_data --root data --verify"
            )
        if len(self.samples) != SPLIT_SIZES[split]:
            raise AssertionError(
                f"{split} 划分应有 {SPLIT_SIZES[split]} 张，实际 {len(self.samples)} 张"
            )

        self.transform = transform if transform is not None else build_transforms(
            split, image_size, strong=strong
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        s = self.samples[idx]
        with Image.open(s.path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, s.label

    # ---- 便捷视图 ----
    def labels(self) -> list[int]:
        return [s.label for s in self.samples]

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts


# --------------------------------------------------------------------------
# DataLoader 工厂
# --------------------------------------------------------------------------
def build_dataloader(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split: Split = "train",
    *,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    strong: bool = False,
    shuffle: bool | None = None,
) -> DataLoader:
    """构建 DataLoader。

    Windows 上默认 ``num_workers=0``（避免子进程重导入与句柄问题）；
    如确认稳定可设 4 以加速（见 docs/04 M2 备注）。
    """
    dataset = Flowers102Dataset(
        data_root, split, image_size=image_size, strong=strong
    )
    if shuffle is None:
        shuffle = split == "train"

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_dataloaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    strong: bool = False,
) -> dict[str, DataLoader]:
    """一次构建 train / val / test 三个 DataLoader。"""
    return {
        split: build_dataloader(
            data_root, split, batch_size=batch_size, image_size=image_size,
            num_workers=num_workers, seed=seed, strong=strong,
        )
        for split in ("train", "val", "test")
    }


# --------------------------------------------------------------------------
# 自检入口：python -m ml.dataset
# --------------------------------------------------------------------------
def _self_check() -> int:
    print("=== 数据管道自检 ===")
    root = Path("data/flowers-102")
    names = get_class_names()
    print(f"✓ 类别名数量：{len(names)}，示例：class_0={names[0]!r}, class_73={names[73]!r}")

    if not (root / "jpg").is_dir():
        print("✗ 数据集未下载，跳过 DataLoader 检查")
        print("  请先执行：python -m ml.download_data --root data")
        return 1

    # 标签顺序交叉校验（防止中文名/百科错位）
    mf = load_manifest(root)
    if mf.class_names_match is True:
        print(f"✓ 源侧类别名与内置 CLASS_NAMES_EN **逐项一致**"
              f"（{len(mf.class_names_from_source or [])} 项）")
    elif mf.class_names_match is False:
        print("✗ 源侧类别名与内置表不一致（见上方差异）→ 必须修正 CLASS_NAMES_EN 后再训练")
        return 1
    else:
        print("… manifest 未记录源侧类别名（官方 .mat 源无此信息），跳过顺序校验；"
              "请依赖 M1-7 的 dataset_samples.png 人工确认")

    for split in ("train", "val", "test"):
        ds = Flowers102Dataset(root, split)
        counts = ds.class_counts()
        print(
            f"✓ {split:5s} 样本 {len(ds):5d}  类别数 {len(counts):3d}  "
            f"每类 {min(counts.values())}—{max(counts.values())} 张  "
            f"标签范围 [{min(ds.labels())}, {max(ds.labels())}]"
        )

    loader = build_dataloader(root, "train", batch_size=32)
    images, targets = next(iter(loader))
    print(f"✓ batch 形状 {tuple(images.shape)}  dtype={images.dtype}")
    print(f"✓ batch 标签 {targets.tolist()[:8]} …  范围 [{int(targets.min())}, {int(targets.max())}]")
    print(f"✓ 归一化后像素范围 [{images.min():.3f}, {images.max():.3f}]（应约在 -2.1 ~ 2.6）")

    t = build_inference_transform()
    print(f"✓ 推理 transform 已构建：{type(t).__name__}（{len(t.transforms)} 步）")

    out = dump_class_names(Path("ml/outputs/class_names.json"))
    print(f"✓ 类别名已落盘：{out}")
    print("=== 自检通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
