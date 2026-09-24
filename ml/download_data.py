"""Oxford 102 Category Flower Dataset 下载、转换与校验工具。

为什么有两个数据源
------------------
本项目所在网络**无法解析** ``www.robots.ox.ac.uk``（本地 DNS 返回 SERVFAIL），
因此官方 tgz 源不可用。默认改用 HuggingFace 镜像 ``hf-mirror.com`` 上的
``dpdl-benchmark/oxford_flowers102``，它使用完全相同的官方划分
（train 1020 / validation 1020 / test 6149），仅封装格式为 parquet。

两个源最终都产出**统一的目录结构 + manifest.json**，下游代码只认 manifest：

    data/flowers-102/
    ├─ jpg/image_00001.jpg … image_08189.jpg
    ├─ manifest.json          # 划分与标签的唯一权威来源
    ├─ class_distribution.csv # 校验报告
    └─ _raw/                  # 下载的原始压缩包/parquet（可删）

用法
----
    python -m ml.download_data --root data                    # 默认 hf 源：下载+转换+校验
    python -m ml.download_data --root data --stage download    # 只下载原始文件
    python -m ml.download_data --root data --stage extract     # 只转换（需已装 pyarrow）
    python -m ml.download_data --root data --verify            # 只校验
    python -m ml.download_data --root data --source oxford     # 官方源（网络可达时）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
EXPECTED_NUM_CLASSES = 102
EXPECTED_NUM_IMAGES = 8189
EXPECTED_SPLIT_SIZES = {"train": 1020, "val": 1020, "test": 6149}
MANIFEST_VERSION = 1

HF_REPO = "dpdl-benchmark/oxford_flowers102"
HF_MIRROR = "https://hf-mirror.com"
HF_FILES: dict[str, list[str]] = {
    "train": ["data/train-00000-of-00001.parquet"],
    "val": ["data/validation-00000-of-00001.parquet"],
    "test": [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)],
}

OXFORD_SOURCES: dict[str, dict[str, str]] = {
    "oxford": {
        "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "setid": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
    },
}

_MAT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------
# 通用下载
# --------------------------------------------------------------------------
def download_file(
    url: str,
    dest: Path,
    *,
    retries: int = 3,
    timeout: int = 60,
    skip_if_exists: bool = True,
    resume: bool = True,
) -> Path:
    """带进度、重试与**断点续传**的下载。默认已存在且非空则跳过。

    本网络环境下大文件（torch wheel 2.6 GB、数据分片 400 MB）中途掉线是常态，
    因此失败时**保留** ``.part`` 文件，下次用 HTTP Range 从断点继续，
    而不是从 0 重下（原先的实现会删掉半成品，代价极高）。
    """
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        _log(f"已存在，跳过：{dest.name} ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # 清理上一次遗留的空文件，避免 Range 请求写成 bytes=0-
    if tmp.exists() and tmp.stat().st_size == 0:
        tmp.unlink()

    for attempt in range(1, retries + 1):
        try:
            offset = tmp.stat().st_size if (resume and tmp.exists()) else 0
            headers = {"User-Agent": "Mozilla/5.0 flower-recognition/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            _log(
                f"下载 {dest.name}（{attempt}/{retries}）"
                + (f" 从 {_human(offset)} 续传" if offset else "")
                + f" ← {url}"
            )
            req = urllib.request.Request(url, headers=headers)
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                partial = resp.status == 206
                if offset and not partial:
                    # 服务器不支持续传，只能从头来
                    _log("服务器不支持 Range，改为从头下载")
                    offset = 0
                remaining = int(resp.headers.get("Content-Length") or 0)
                total = offset + remaining if remaining else 0
                done = offset
                last = 0.0
                with open(tmp, "ab" if offset else "wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last >= 2.0:
                            last = now
                            speed = (done - offset) / max(now - start, 1e-6)
                            if total:
                                eta = (total - done) / max(speed, 1e-6)
                                print(
                                    f"\r  {done / total * 100:5.1f}%  {_human(done)}/{_human(total)}"
                                    f"  {_human(speed)}/s  ETA {eta:4.0f}s",
                                    end="", flush=True,
                                )
                            else:
                                print(f"\r  {_human(done)}  {_human(speed)}/s", end="", flush=True)
            print()
            if total and tmp.stat().st_size != total:
                raise IOError(f"下载不完整：{tmp.stat().st_size} != {total} 字节")
            tmp.replace(dest)
            _log(f"完成：{dest.name} ({_human(dest.stat().st_size)})")
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            keep = tmp.stat().st_size if tmp.exists() else 0
            _log(f"失败：{type(exc).__name__}: {exc}")
            if keep:
                _log(f"已保留断点 {_human(keep)}，下次将续传")
            if attempt == retries:
                raise
            wait = 3 * attempt
            _log(f"{wait}s 后重试……")
            time.sleep(wait)

    raise RuntimeError("unreachable")


def md5_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# --------------------------------------------------------------------------
# MATLAB v5 .mat 最小解析器（官方源用）
# --------------------------------------------------------------------------
def load_mat_v5(path: Path) -> dict[str, object]:
    """解析 MATLAB v5 .mat 中的数值矩阵；有 scipy 时优先用 scipy。"""
    try:
        from scipy.io import loadmat  # type: ignore

        raw = loadmat(str(path))
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except ImportError:
        pass

    import struct

    data = path.read_bytes()
    if len(data) < 128:
        raise ValueError(f"{path.name} 太小，不是合法 .mat 文件")

    result: dict[str, object] = {}
    pos = 128
    endian = "<"

    while pos + 8 <= len(data):
        tag_type, tag_size = struct.unpack_from(endian + "II", data, pos)
        pos += 8
        if tag_type != 14:  # 只处理 miMATRIX
            pos += tag_size + ((8 - tag_size % 8) % 8)
            continue

        nxt = pos + tag_size
        flags_type, _flags_size = struct.unpack_from(endian + "II", data, pos)
        pos += 8
        cls = flags_type & 0xFF
        _dims_type, dims_size = struct.unpack_from(endian + "II", data, pos)
        pos += 8
        ndim = dims_size // 4
        dims = struct.unpack_from(endian + f"{ndim}I", data, pos)
        pos += dims_size

        _name_type, name_size = struct.unpack_from(endian + "II", data, pos)
        pos += 8
        name = data[pos : pos + name_size].decode("latin-1")
        pos += name_size + ((8 - name_size % 8) % 8)

        data_type, data_size = struct.unpack_from(endian + "II", data, pos)
        pos += 8
        payload = data[pos : pos + data_size]
        pos += data_size + ((8 - data_size % 8) % 8)

        if cls in (8, 9, 10, 12, 13) or data_type in (1, 2, 3, 4, 5, 6, 9):
            fmt_map = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 9: "d"}
            fmt = fmt_map.get(data_type, "d")
            count = data_size // struct.calcsize(fmt)
            values = list(struct.unpack_from(endian + f"{count}{fmt}", payload, 0))
            cols = dims[0] if ndim > 0 else count
            rows = dims[1] if ndim > 1 else 1
            if cols == 1 or rows == 1:
                result[name] = [int(v) if fmt != "d" else float(v) for v in values]
            else:
                result[name] = values
        pos = nxt

    return result


# --------------------------------------------------------------------------
# 源 1：HuggingFace 镜像 parquet
# --------------------------------------------------------------------------
def _parquet_url(rel_path: str) -> str:
    return f"{HF_MIRROR}/datasets/{HF_REPO}/resolve/main/{rel_path}"


def stage_download_hf(root: Path) -> Path:
    raw = root / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split, files in HF_FILES.items():
        for rel in files:
            dest = raw / f"{split}__{Path(rel).name}"
            download_file(_parquet_url(rel), dest, timeout=120)
    return raw


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in columns:
        if any(cand in c.lower() for cand in candidates):
            return c
    return None


def stage_extract_hf(root: Path) -> Path:
    """把 parquet 中的图片落盘为 jpg，并生成 manifest.json。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 pyarrow，无法读取 parquet。请先安装：pip install pyarrow"
        ) from exc

    raw = root / "_raw"
    jpg_dir = root / "jpg"
    jpg_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    counter = 0
    label_min, label_max = 10**9, -10**9
    source_names: list[str] | None = None

    for split in ("train", "val", "test"):
        files = sorted(raw.glob(f"{split}__*.parquet"))
        if not files:
            raise FileNotFoundError(f"未找到 {split} 的 parquet，请先执行 --stage download")
        _log(f"转换 {split}：{len(files)} 个 parquet 文件")
        for pf in files:
            table = pq.read_table(pf)
            columns = table.column_names
            img_col = _pick_column(columns, ("image", "img", "picture"))
            lbl_col = _pick_column(columns, ("label", "labels", "fine_label"))
            if img_col is None or lbl_col is None:
                raise ValueError(
                    f"{pf.name} 列名无法识别：{columns}（需要图像列与标签列）"
                )

            # 从 parquet 的 HF 元数据里取源侧类别名，用于交叉校验标签顺序
            if source_names is None:
                source_names = _hf_class_label_names(table, lbl_col)
                if source_names:
                    _log(f"  源侧类别名：{len(source_names)} 个（取自 {pf.name} 元数据）")
                else:
                    _log(f"  ⚠ {pf.name} 未提供 ClassLabel 元数据，跳过标签顺序交叉校验")

            images = table.column(img_col).to_pylist()
            labels = table.column(lbl_col).to_pylist()
            name_to_index = {n: i for i, n in enumerate(source_names)} if source_names else {}

            for img, lbl in zip(images, labels):
                counter += 1
                name = f"image_{counter:05d}.jpg"
                out = jpg_dir / name
                raw_bytes, img_path = _unwrap_image(img)
                if not out.exists():
                    _write_image(out, raw_bytes, img_path)
                # 标签列可能是 int，也可能是字符串（字典编码的 ClassLabel）
                if isinstance(lbl, str):
                    if lbl not in name_to_index:
                        raise ValueError(f"标签字符串 {lbl!r} 不在源侧类别名列表中")
                    lbl_i = name_to_index[lbl]
                else:
                    lbl_i = int(lbl)
                label_min = min(label_min, lbl_i)
                label_max = max(label_max, lbl_i)
                entries.append({"f": name, "l": lbl_i, "s": split})

            _log(f"  {pf.name} → 累计 {counter} 张")

    # 标签统一为 0-based
    if label_min == 1:
        _log("检测到标签为 1-based，统一转换为 0-based")
        for e in entries:
            e["l"] = e["l"] - 1
        label_min, label_max = label_min - 1, label_max - 1

    manifest = _build_manifest(
        entries,
        source=f"hf-mirror:{HF_REPO}",
        class_names_from_source=source_names,
    )
    _write_manifest(root, manifest)
    _log(f"manifest.json 已生成：{len(entries)} 条，标签范围 [{label_min}, {label_max}]")
    return root


def _hf_class_label_names(table, label_column: str) -> list[str] | None:
    """从 parquet 的 ``huggingface`` 元数据里取出 ClassLabel 的类别名列表。

    HF datasets 会把 features 描述写进 Arrow schema metadata（键名 ``huggingface``），
    形如 ``{"info": {"features": {"label": {"_type": "ClassLabel", "names": [...]}}}}``。
    拿到源侧顺序后，就能与项目内置的 cat_to_name 表逐项比对，
    防止「标签顺序错位导致中文名全错」这类致命问题。
    """
    meta = getattr(table.schema, "metadata", None) or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    try:
        info = json.loads(raw.decode("utf-8"))
        features = (info.get("info") or {}).get("features") or info.get("features") or {}
        for fname, spec in features.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("_type") == "ClassLabel" and spec.get("names"):
                if fname == label_column or len(features) == 1:
                    return [str(n) for n in spec["names"]]
        # 兜底：任意一个 ClassLabel 都接受
        for spec in features.values():
            if isinstance(spec, dict) and spec.get("_type") == "ClassLabel" and spec.get("names"):
                return [str(n) for n in spec["names"]]
    except (ValueError, AttributeError, KeyError) as exc:
        _log(f"  解析 huggingface 元数据失败（忽略）：{type(exc).__name__}: {exc}")
    return None


def _unwrap_image(img) -> tuple[bytes | None, str | None]:
    """HF 图像列可能是 bytes、{'bytes':..,'path':..} 或 {'path':..}。"""
    if isinstance(img, (bytes, bytearray)):
        return bytes(img), None
    if isinstance(img, dict):
        b = img.get("bytes")
        p = img.get("path")
        return (bytes(b) if b is not None else None), (str(p) if p else None)
    raise TypeError(f"无法识别的图像字段类型：{type(img)}")


def _write_image(out: Path, raw_bytes: bytes | None, img_path: str | None) -> None:
    if raw_bytes is None:
        raise ValueError(f"图像字段既无 bytes 也无可用数据：{img_path}")
    # 原数据本身多为 JPEG，直接落盘可保留原始质量
    if raw_bytes[:3] == b"\xff\xd8\xff":
        out.write_bytes(raw_bytes)
        return
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(raw_bytes)) as im:
        im.convert("RGB").save(out, format="JPEG", quality=95, subsampling=0)


# --------------------------------------------------------------------------
# 源 2：官方 ox.ac.uk
# --------------------------------------------------------------------------
def stage_download_oxford(root: Path) -> Path:
    urls = OXFORD_SOURCES["oxford"]
    raw = root / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    tgz = download_file(urls["images"], raw / "102flowers.tgz")
    download_file(urls["labels"], raw / "imagelabels.mat")
    download_file(urls["setid"], raw / "setid.mat")

    jpg_dir = root / "jpg"
    if not jpg_dir.exists() or len(list(jpg_dir.glob("*.jpg"))) < EXPECTED_NUM_IMAGES:
        _log("解压 102flowers.tgz ……")
        with tarfile.open(tgz, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".jpg")]
            tar.extractall(path=root, members=members)  # noqa: S202 - 官方可信压缩包
        _log(f"解压完成：{len(list(jpg_dir.glob('*.jpg')))} 张")
    return root


def stage_extract_oxford(root: Path) -> Path:
    raw = root / "_raw"
    labels_raw = load_mat_v5(raw / "imagelabels.mat")
    setid_raw = load_mat_v5(raw / "setid.mat")
    if "labels" not in labels_raw:
        raise KeyError(f"imagelabels.mat 缺少 'labels'，实际键：{sorted(labels_raw)}")

    labels = [int(v) for v in labels_raw["labels"]]  # type: ignore[arg-type]
    if labels and min(labels) == 1:
        labels = [v - 1 for v in labels]

    entries: list[dict] = []
    for split, key in _MAT_KEYS.items():
        if key not in setid_raw:
            raise KeyError(f"setid.mat 缺少 '{key}'，实际键：{sorted(setid_raw)}")
        for idx in (int(v) for v in setid_raw[key]):  # type: ignore[arg-type]
            entries.append({
                "f": f"image_{idx:05d}.jpg",
                "l": labels[idx - 1],
                "s": split,
            })

    manifest = _build_manifest(entries, source="oxford-official")
    _write_manifest(root, manifest)
    _log(f"manifest.json 已生成：{len(entries)} 条")
    return root


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def _build_manifest(
    entries: list[dict],
    *,
    source: str,
    class_names_from_source: list[str] | None = None,
) -> dict:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["s"]] = counts.get(e["s"], 0) + 1
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset": "Oxford 102 Category Flower Dataset (Flowers-102)",
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_classes": EXPECTED_NUM_CLASSES,
        "num_images": len(entries),
        "image_dir": "jpg",
        "counts": counts,
        "images": entries,
    }
    if class_names_from_source:
        manifest["class_names_from_source"] = class_names_from_source
    return manifest


def _write_manifest(root: Path, manifest: dict) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def load_manifest(root: Path) -> dict:
    path = (root / "manifest.json") if (root / "manifest.json").exists() else (root / "flowers-102" / "manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest.json（已查找 {path}）")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def verify(root: Path) -> bool:
    jpg_dir = root / "jpg"
    problems: list[str] = []

    try:
        manifest = load_manifest(root)
    except FileNotFoundError:
        # 回退：老式 .mat 布局
        labels_path = root / "imagelabels.mat"
        setid_path = root / "setid.mat"
        if labels_path.exists() and setid_path.exists():
            _log("未找到 manifest.json，回退到 .mat 校验")
            return _verify_mat(root, labels_path, setid_path)
        _log(f"✗ 未找到 manifest.json，且无 .mat 标注：{root}")
        return False

    if not jpg_dir.is_dir():
        problems.append(f"缺少图片目录：{jpg_dir}")

    entries = manifest.get("images", [])
    if len(entries) != EXPECTED_NUM_IMAGES:
        problems.append(f"manifest 记录 {len(entries)} 张，期望 {EXPECTED_NUM_IMAGES}")

    counts = manifest.get("counts", {})
    for split, expect in EXPECTED_SPLIT_SIZES.items():
        got = int(counts.get(split, 0))
        if got != expect:
            problems.append(f"{split} 划分 {got} 张，期望 {expect}")
        else:
            _log(f"✓ {split:5s} 划分 {got} 张")

    labels = [int(e["l"]) for e in entries]
    if labels:
        if min(labels) != 0 or max(labels) != EXPECTED_NUM_CLASSES - 1:
            problems.append(f"标签范围 [{min(labels)}, {max(labels)}]，期望 [0, {EXPECTED_NUM_CLASSES - 1}]")
        n_classes = len(set(labels))
        if n_classes != EXPECTED_NUM_CLASSES:
            problems.append(f"实际类别数 {n_classes}，期望 {EXPECTED_NUM_CLASSES}")
        else:
            _log(f"✓ 类别数 {n_classes}")

    # 抽查文件存在性（全量 stat 太慢时也够用）
    if jpg_dir.is_dir() and not problems:
        missing = [e["f"] for e in entries[::37] if not (jpg_dir / e["f"]).exists()]
        if missing:
            problems.append(f"抽查发现 {len(missing)} 个文件缺失，例如 {missing[:3]}")

    # 类别分布报告
    if labels and not problems:
        counts_by_class: dict[int, int] = {}
        for v in labels:
            counts_by_class[v] = counts_by_class.get(v, 0) + 1
        report = root / "class_distribution.csv"
        with open(report, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["class_id_0based", "num_images"])
            for cid in range(EXPECTED_NUM_CLASSES):
                w.writerow([cid, counts_by_class.get(cid, 0)])
        _log(
            f"✓ 类别分布报告：{report}（每类 "
            f"{min(counts_by_class.values())}—{max(counts_by_class.values())} 张）"
        )

    if problems:
        _log("=" * 62)
        for p in problems:
            _log(f"✗ {p}")
        _log("=" * 62)
        return False

    _log("=" * 62)
    _log(f"OK: {EXPECTED_NUM_CLASSES} classes, {EXPECTED_NUM_IMAGES} images, splits 1020/1020/6149")
    _log(f"    数据源：{manifest.get('source')}")
    _log("=" * 62)
    return True


def _verify_mat(root: Path, labels_path: Path, setid_path: Path) -> bool:
    problems: list[str] = []
    files = sorted((root / "jpg").glob("*.jpg")) if (root / "jpg").is_dir() else []
    if len(files) != EXPECTED_NUM_IMAGES:
        problems.append(f"图片数量 {len(files)}，期望 {EXPECTED_NUM_IMAGES}")

    labels_raw = load_mat_v5(labels_path)
    setid_raw = load_mat_v5(setid_path)
    lab = [int(v) for v in labels_raw.get("labels", [])]  # type: ignore[arg-type]
    if lab and (min(lab) != 1 or max(lab) != EXPECTED_NUM_CLASSES):
        problems.append(f"标签范围 [{min(lab)}, {max(lab)}]，期望 [1, {EXPECTED_NUM_CLASSES}]")

    for key, alias in _MAT_KEYS.items():
        got = len(setid_raw.get(alias, []))  # type: ignore[arg-type]
        expect = EXPECTED_SPLIT_SIZES[key]
        if got != expect:
            problems.append(f"{key} 划分 {got} 张，期望 {expect}")
        else:
            _log(f"✓ {key:5s} 划分 {got} 张")

    if problems:
        for p in problems:
            _log(f"✗ {p}")
        return False
    _log(f"OK: {EXPECTED_NUM_CLASSES} classes, {EXPECTED_NUM_IMAGES} images（.mat 布局）")
    return True


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oxford Flowers-102 下载 / 转换 / 校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="数据源：hf（默认，hf-mirror.com parquet）、oxford（官方 tgz，需可访问 ox.ac.uk）",
    )
    ap.add_argument("--root", type=Path, default=Path("data"), help="数据根目录（默认 data）")
    ap.add_argument("--source", default="hf", choices=["hf", "oxford"], help="下载源")
    ap.add_argument("--stage", default="all", choices=["all", "download", "extract"],
                    help="all=下载+转换；download=只下载；extract=只转换")
    ap.add_argument("--verify", action="store_true", help="仅校验，不下载")
    ap.add_argument("--md5", action="store_true", help="打印原始文件 MD5（写入报告用）")
    args = ap.parse_args(argv)

    dataset_dir = args.root.resolve() / "flowers-102"

    if args.verify:
        return 0 if verify(dataset_dir) else 1

    if args.md5:
        for f in sorted((dataset_dir / "_raw").glob("*")):
            if f.is_file():
                _log(f"{f.name}  MD5={md5_of(f)}  size={_human(f.stat().st_size)}")
        return 0

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _log(f"数据源：{args.source}   阶段：{args.stage}   目标：{dataset_dir}")

    if args.source == "hf":
        if args.stage in ("all", "download"):
            stage_download_hf(dataset_dir)
        if args.stage in ("all", "extract"):
            stage_extract_hf(dataset_dir)
    else:
        if args.stage in ("all", "download"):
            stage_download_oxford(dataset_dir)
        if args.stage in ("all", "extract"):
            stage_extract_oxford(dataset_dir)

    return 0 if verify(dataset_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
