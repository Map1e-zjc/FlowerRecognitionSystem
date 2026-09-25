"""训练/评估共享工具：随机种子、设备解析、指标、日志、JSON。"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
# 可复现性
# --------------------------------------------------------------------------
def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """固定所有随机源（NFR-06 可复现性）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 允许 cudnn 自动挑选最快卷积算法（速度优先，精度波动在可接受范围）
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------
# 设备
# --------------------------------------------------------------------------
def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda 但 torch.cuda.is_available() 为 False（是否装了 CPU 版 torch？）")
    return dev


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total = getattr(props, "total_memory", 0) / 1024**3
        return f"cuda:{idx} {props.name} ({total:.1f} GB) | torch {torch.__version__} | cuda {torch.version.cuda}"
    return f"cpu | torch {torch.__version__}"


def gpu_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    """返回每个 k 的 Top-k 准确率（百分比，0—100）。"""
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out: list[float] = []
    for k in topk:
        kk = min(k, maxk)
        correct_k = correct[:kk].reshape(-1).float().sum().item()
        out.append(100.0 * correct_k / logits.size(0))
    return out


class AverageMeter:
    """滑动平均计量器（loss / acc 用）。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.name}={self.avg:.4f}"


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
class CSVLogger:
    """逐 epoch 追加写 CSV（崩溃后可续，训练曲线数据源）。"""

    FIELDS = [
        "epoch", "lr_backbone", "lr_head", "train_loss", "train_top1",
        "val_loss", "val_top1", "val_top5", "epoch_seconds", "gpu_mem_mb",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self.FIELDS).writeheader()

    def log(self, **row) -> None:
        clean = {k: row.get(k, "") for k in self.FIELDS}
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.FIELDS).writerow(clean)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = []
            for r in csv.DictReader(fh):
                parsed: dict = {}
                for k, v in r.items():
                    if v == "":
                        parsed[k] = None
                        continue
                    try:
                        parsed[k] = int(v) if k == "epoch" else float(v)
                    except ValueError:
                        parsed[k] = v
                rows.append(parsed)
            return rows


class Tee:
    """把输出同时写到终端与文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class ETATimer:
    """按已完成 epoch 的平均耗时估算剩余时间。"""

    def __init__(self, total_epochs: int) -> None:
        self.total = total_epochs
        self.times: list[float] = []
        self._t0 = time.time()

    def tick(self) -> None:
        self.times.append(time.time() - self._t0)
        self._t0 = time.time()

    @property
    def eta(self) -> str:
        if not self.times:
            return "--:--"
        avg = sum(self.times) / len(self.times)
        remaining = self.total - len(self.times)
        return format_seconds(avg * remaining)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def save_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置（pyyaml 为必需依赖）。"""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在：{p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射：{p}")
    return data


# --------------------------------------------------------------------------
# 进度条：非交互环境自动禁用，避免把日志刷爆
# --------------------------------------------------------------------------
_PROGRESS_MODE = os.environ.get("ML_PROGRESS", "auto").strip().lower()


def set_progress_mode(mode: str) -> None:
    """设置进度条策略：``auto``（默认）/ ``1`` 强制开 / ``0`` 强制关。"""
    global _PROGRESS_MODE
    _PROGRESS_MODE = str(mode).strip().lower()


def progress_enabled() -> bool:
    """是否显示 tqdm 进度条。

    规则（按优先级）：

    1. ``ML_PROGRESS=0/off``（或 ``--no-progress``）→ 关闭；
    2. ``ML_PROGRESS=1/on``（或 ``--progress``）→ 开启，即使不是终端；
    3. 默认 ``auto``：仅当 stderr 是真实终端时才显示。

    **为什么必须做这件事**：tqdm 在非 TTY（后台作业、管道、日志采集）下依然会
    把每个 step 输出成一行。实测 ConvNeXt-Tiny 只跑 40 秒就产生 **157.8 KB**
    控制台日志，按 60 epoch 推算可达 10 MB 级，会把作业输出缓冲区打爆、
    淹没真正有用的信息。

    后台运行时进度条没有价值 —— 真正可靠的进度来源是 ``*_train.csv`` 与业务日志。
    """
    if _PROGRESS_MODE in ("0", "false", "off", "no"):
        return False
    if _PROGRESS_MODE in ("1", "true", "on", "yes"):
        return True
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def make_bar(iterable=None, *, total: int | None = None, desc: str = "", **kwargs):
    """统一的 tqdm 工厂：非交互环境自动禁用（见 :func:`progress_enabled`）。

    两种用法都支持：

    * ``for x in make_bar(loader, total=len(loader), desc="推理")`` —— 直接迭代；
    * ``bar = make_bar(total=n, desc="...")`` 然后 ``for x in loader: bar.update(1)``。
    """
    from tqdm import tqdm

    kwargs.setdefault("ncols", 100)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", False)
    return tqdm(iterable, total=total, desc=desc,
                disable=not progress_enabled(), **kwargs)


# --------------------------------------------------------------------------
# 缓存目录兜底（不依赖调用者记得设环境变量）
# --------------------------------------------------------------------------
def project_root() -> Path:
    """返回项目根目录（``ml/utils.py`` 的上一级）。"""
    return Path(__file__).resolve().parents[1]


def ensure_cache_env(root: str | Path | None = None) -> dict[str, str]:
    """把缓存类环境变量默认指向项目内 ``.cache/``，并默认启用 HF 镜像。

    **为什么需要兜底**：torchvision/timm 的预训练权重默认下载到
    ``C:\\Users\\<user>\\.cache\\{torch,huggingface}``。本机 C 盘余量仅约 15 GB，
    在受限环境下写该目录还会直接抛 ``PermissionError: [WinError 5]``，
    报错信息完全指不到真正原因（实测踩到过）。

    已显式设置的变量优先级更高，本函数不会覆盖它们。

    Returns:
        本次实际设置的 ``{变量名: 值}``（便于日志打印）。
    """
    base = Path(root) if root is not None else project_root()
    cache = base / ".cache"
    defaults = {
        "PIP_CACHE_DIR": str(cache / "pip"),
        "TORCH_HOME": str(cache / "torch"),
        "HF_HOME": str(cache / "hf"),
        "HF_ENDPOINT": "https://hf-mirror.com",  # huggingface.co 直连超时（ADR-014）
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    for key in ("PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return applied
