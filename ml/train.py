"""统一训练脚本：迁移学习微调 Flowers-102（4 个主干共用）。

用法
----
    python -m ml.train --config ml/configs/resnet50.yaml                  # 按配置训练
    python -m ml.train --config ml/configs/vit_b16.yaml --epochs 2        # 覆盖 epoch
    python -m ml.train --config ... --resume                              # 从 last 断点续训
    python -m ml.train --config ... --max-train-batches 3                 # 冒烟测试（M1-8）

关键实现（对应 docs/03 §3.2、docs/04 M2-1）
-------------------------------------------
* 前 ``warmup_epochs`` 个 epoch 冻结主干，只训分类头，学习率从 0.1× 线性升到 1×；
* 之后解冻全网络，重建优化器（分层学习率）与 CosineAnnealingLR（eta_min=1e-6）；
* AMP 混合精度 + 梯度裁剪 + 梯度累积（ViT 在 8 GB 显存下的等效 batch）；
* 早停按 val Top-1，保存最佳权重；逐 epoch 写 CSV，崩溃后可用 --resume 续训。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from ml.dataset import (
    DEFAULT_DATA_ROOT,
    NUM_CLASSES,
    SPLIT_SIZES,
    build_dataloaders,
    dump_class_names,
    get_class_names,
)
from ml.models import (
    DEFAULTS,
    SUPPORTED_MODELS,
    build_model,
    count_parameters,
    freeze_backbone,
    get_param_groups,
    unfreeze_backbone,
)
from ml.utils import (
    AverageMeter,
    CSVLogger,
    ETATimer,
    Tee,
    accuracy_topk,
    device_summary,
    ensure_cache_env,
    format_seconds,
    gpu_mem_mb,
    load_yaml,
    make_bar,
    resolve_device,
    save_json,
    set_progress_mode,
    set_seed,
)


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.config:
        cfg.update(load_yaml(args.config))

    model_name = args.model or cfg.get("model")
    if not model_name:
        raise SystemExit("必须通过 --config 或 --model 指定模型")
    if model_name not in SUPPORTED_MODELS:
        raise SystemExit(f"不支持的模型 {model_name!r}，可选：{', '.join(SUPPORTED_MODELS)}")

    merged = {**DEFAULTS[model_name], **cfg}
    merged["model"] = model_name

    # 命令行覆盖（仅覆盖显式传入的项）
    for key in (
        "epochs", "batch_size", "image_size", "lr_backbone", "lr_head", "weight_decay",
        "label_smoothing", "grad_clip", "warmup_epochs", "patience", "eta_min",
        "num_workers", "seed", "device", "grad_accum", "out_dir", "data_root",
        "strong", "tag",
    ):
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val
    if getattr(args, "config", None):
        merged["config_path"] = str(args.config)

    merged.setdefault("eta_min", 1e-6)
    merged.setdefault("label_smoothing", 0.1)
    merged.setdefault("grad_clip", 1.0)
    merged.setdefault("warmup_epochs", 3)
    merged.setdefault("patience", 10)
    merged.setdefault("out_dir", "ml/outputs")
    merged.setdefault("data_root", str(DEFAULT_DATA_ROOT))
    merged.setdefault("strong", False)
    merged.setdefault("num_workers", 0)
    merged.setdefault("grad_accum", 1)
    return merged


def run_tag(cfg: dict) -> str:
    return str(cfg.get("tag") or cfg["model"])


# --------------------------------------------------------------------------
# 训练 / 验证一个 epoch
# --------------------------------------------------------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, *,
    use_amp: bool, grad_clip: float, grad_accum: int, max_batches: int | None,
    log_fn, epoch: int, total_epochs: int,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("top1")

    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    limit = min(max_batches, n_batches) if max_batches else n_batches

    bar = make_bar(total=limit, desc=f"epoch {epoch + 1}/{total_epochs} train")
    for step, (images, targets) in enumerate(loader):
        if step >= limit:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == limit:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = targets.size(0)
        loss_meter.update(loss.item() * grad_accum, bs)
        top1, = accuracy_topk(logits.detach().float(), targets, (1,))
        acc_meter.update(top1, bs)
        bar.update(1)
        bar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{acc_meter.avg:.2f}%")
    bar.close()

    log_fn(
        f"  train   loss={loss_meter.avg:.4f}  top1={acc_meter.avg:.2f}%  "
        f"({limit} batches)"
    )
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, use_amp: bool, desc: str) -> tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    top5_meter = AverageMeter("top5")

    bar = make_bar(total=len(loader), desc=desc)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        bs = targets.size(0)
        top1, top5 = accuracy_topk(logits.float(), targets, (1, 5))
        loss_meter.update(loss.item(), bs)
        top1_meter.update(top1, bs)
        top5_meter.update(top5, bs)
        bar.update(1)
        bar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}%")
    bar.close()
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# --------------------------------------------------------------------------
# 优化器 / 调度器装配
# --------------------------------------------------------------------------
def make_optimizer(model, cfg: dict):
    groups = get_param_groups(
        model, cfg["model"], float(cfg["lr_backbone"]), float(cfg["lr_head"]),
        float(cfg["weight_decay"]),
    )
    return AdamW(groups)


def make_scaler(device: torch.device, use_amp: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # 兼容旧版 torch
        return torch.cuda.amp.GradScaler(enabled=use_amp)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Flowers-102 迁移学习训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=None, help="YAML 配置文件")
    ap.add_argument("--model", default=None, choices=list(SUPPORTED_MODELS), help="模型名（可替代 --config）")
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--image-size", dest="image_size", type=int, default=None)
    ap.add_argument("--lr-backbone", dest="lr_backbone", type=float, default=None)
    ap.add_argument("--lr-head", dest="lr_head", type=float, default=None)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    ap.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=None)
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    ap.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--eta-min", dest="eta_min", type=float, default=None)
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=None)
    ap.add_argument("--tag", default=None, help="输出文件前缀（默认等于模型名）")
    ap.add_argument("--strong", action="store_true", default=None, help="启用强增强")
    ap.add_argument("--resume", action="store_true", help="从 last 断点续训")
    ap.add_argument("--max-train-batches", dest="max_train_batches", type=int, default=None,
                    help="每 epoch 最多训练多少个 batch（冒烟测试用）")
    ap.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练（仅调试）")
    ap.add_argument("--progress", dest="progress", action="store_true", default=None,
                    help="强制显示进度条（默认 auto：仅终端下显示）")
    ap.add_argument("--no-progress", dest="progress", action="store_false",
                    help="强制关闭进度条（后台作业推荐，避免日志刷屏）")
    args = ap.parse_args(argv)

    # 缓存目录兜底：必须在构建模型（下载预训练权重）之前调用
    applied_env = ensure_cache_env()

    cfg = build_config(args)
    tag = run_tag(cfg)

    # 进度条策略：CLI 优先，其次配置文件 progress 字段，否则 auto（仅终端显示）
    if args.progress is not None:
        set_progress_mode("1" if args.progress else "0")
    elif str(cfg.get("progress", "auto")).lower() in ("0", "false", "off", "no"):
        set_progress_mode("0")

    out_dir = Path(str(cfg["out_dir"]))
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = Tee(log_dir / f"{tag}_train.log")
    log("=" * 78)
    log(f"模型：{cfg['model']}（{cfg.get('display_name', cfg['model'])}）    tag={tag}")
    if applied_env:
        log(f"缓存环境变量（本次自动兜底设置）：{json.dumps(applied_env, ensure_ascii=False)}")
    log(f"配置：{json.dumps({k: v for k, v in cfg.items()}, ensure_ascii=False, default=str)}")
    log("=" * 78)

    # --- 设备与可复现 ---
    device = resolve_device(str(cfg["device"]))
    set_seed(int(cfg["seed"]))
    use_amp = device.type == "cuda"
    log(f"设备：{device_summary(device)}")
    log(f"AMP ：{'开启' if use_amp else '关闭（CPU）'}   梯度累积：{cfg['grad_accum']}")

    # --- 数据 ---
    loaders = build_dataloaders(
        cfg["data_root"],
        batch_size=int(cfg["batch_size"]),
        image_size=int(cfg["image_size"]),
        num_workers=int(cfg["num_workers"]),
        seed=int(cfg["seed"]),
        strong=bool(cfg["strong"]),
    )
    for split, dl in loaders.items():
        log(f"数据 {split:5s}：{len(dl.dataset):5d} 张 / {len(dl):4d} batch"
            f"（期望 {SPLIT_SIZES[split]}）")

    # --- 模型 ---
    model = build_model(cfg["model"], NUM_CLASSES, pretrained=not args.no_pretrained)
    model.to(device)
    total, trainable = count_parameters(model)
    log(f"参数量：{total / 1e6:.2f} M（可训练 {trainable / 1e6:.2f} M）")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    epochs = int(cfg["epochs"])
    warmup_epochs = min(int(cfg["warmup_epochs"]), max(epochs - 1, 0))

    start_epoch = 0
    best_top1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict] = []

    # --- 优化器：前 warmup_epochs 冻结主干 ---
    backbone_frozen = warmup_epochs > 0
    if backbone_frozen:
        frozen_n = freeze_backbone(model, cfg["model"])
        log(f"warmup：冻结主干 {frozen_n} 个参数张量，前 {warmup_epochs} epoch 只训分类头")
        optimizer = make_optimizer(model, cfg)
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
    else:
        optimizer = make_optimizer(model, cfg)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(cfg["eta_min"]))
    scaler = make_scaler(device, use_amp)

    csv_logger = CSVLogger(log_dir / f"{tag}_train.csv")
    last_path = ckpt_dir / f"{tag}_last.pth"
    best_path = ckpt_dir / f"{tag}_best.pth"

    # --- 断点续训 ---
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_top1 = float(ckpt.get("best_top1", 0.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        history = list(ckpt.get("history", []))
        if not backbone_frozen or start_epoch > warmup_epochs:
            unfreeze_backbone(model)
            backbone_frozen = False
            optimizer = make_optimizer(model, cfg)
            remaining = max(epochs - start_epoch, 1)
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=float(cfg["eta_min"]))
            scaler = make_scaler(device, use_amp)
        log(f"续训：从 epoch {start_epoch + 1} 继续（best_top1={best_top1:.2f}%）")

    eta = ETATimer(epochs - start_epoch)
    train_start = time.time()

    for epoch in range(start_epoch, epochs):
        # 解冻主干并重建优化器/调度器
        if backbone_frozen and epoch >= warmup_epochs:
            unfreeze_backbone(model)
            backbone_frozen = False
            optimizer = make_optimizer(model, cfg)
            remaining = max(epochs - epoch, 1)
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=float(cfg["eta_min"]))
            scaler = make_scaler(device, use_amp)
            log(f"[epoch {epoch + 1}] 解冻主干，切换为余弦退火（T_max={remaining}）")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        log(f"--- epoch {epoch + 1}/{epochs} ---")
        tr_loss, tr_top1 = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            use_amp=use_amp, grad_clip=float(cfg["grad_clip"]),
            grad_accum=int(cfg["grad_accum"]), max_batches=args.max_train_batches,
            log_fn=log, epoch=epoch, total_epochs=epochs,
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, loaders["val"], criterion, device, use_amp=use_amp,
            desc=f"epoch {epoch + 1}/{epochs} val",
        )
        lrs = [g["lr"] for g in optimizer.param_groups]
        lr_backbone_now = min(lrs) if lrs else 0.0
        lr_head_now = max(lrs) if lrs else 0.0
        scheduler.step()
        epoch_secs = time.time() - t0
        eta.tick()

        mem = gpu_mem_mb(device)
        log(
            f"  val     loss={val_loss:.4f}  top1={val_top1:.2f}%  top5={val_top5:.2f}%  "
            f"| {format_seconds(epoch_secs)}  显存峰值 {mem:.0f} MB  ETA {eta.eta}"
        )

        row = {
            "epoch": epoch + 1,
            "lr_backbone": lr_backbone_now,
            "lr_head": lr_head_now,
            "train_loss": round(tr_loss, 6),
            "train_top1": round(tr_top1, 4),
            "val_loss": round(val_loss, 6),
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "epoch_seconds": round(epoch_secs, 2),
            "gpu_mem_mb": mem,
        }
        csv_logger.log(**row)
        history.append(row)

        improved = val_top1 > best_top1
        if improved:
            best_top1 = val_top1
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        payload = {
            "state_dict": model.state_dict(),
            "model_name": cfg["model"],
            "num_classes": NUM_CLASSES,
            "image_size": int(cfg["image_size"]),
            "epoch": epoch,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_loss": val_loss,
            "best_top1": best_top1,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "config": cfg,
            "class_names": get_class_names(),
            "history": history,
        }
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
            log(f"  ★ 新的最佳 val Top-1 = {best_top1:.2f}%（已保存 {best_path.name}）")

        if epochs_no_improve >= int(cfg["patience"]):
            log(f"早停：val Top-1 连续 {epochs_no_improve} 个 epoch 未提升（最佳 {best_top1:.2f}%）")
            break

    train_secs = time.time() - train_start
    summary = {
        "model": cfg["model"],
        "display_name": cfg.get("display_name", cfg["model"]),
        "tag": tag,
        "epochs_planned": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch + 1,
        "best_val_top1": round(best_top1, 4),
        "best_val_top5": round(history[best_epoch]["val_top5"], 4) if best_epoch >= 0 else None,
        "train_minutes": round(train_secs / 60, 2),
        "params_m": round(total / 1e6, 2),
        "image_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum": int(cfg["grad_accum"]),
        "strong_augment": bool(cfg["strong"]),
        "seed": int(cfg["seed"]),
        "device": device_summary(device),
        "checkpoint_best": str(best_path),
        "config": cfg,
        "history": history,
    }
    save_json(summary, log_dir / f"{tag}_summary.json")

    # 类别名落盘（供百科与前端复用）
    dump_class_names(out_dir / "class_names.json")

    log("=" * 78)
    log(f"训练完成：最佳 val Top-1 {best_top1:.2f}% @ epoch {best_epoch + 1}"
        f"   总耗时 {format_seconds(train_secs)}")
    log(f"最佳权重：{best_path}")
    log(f"训练摘要：{log_dir / f'{tag}_summary.json'}")
    log("=" * 78)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
