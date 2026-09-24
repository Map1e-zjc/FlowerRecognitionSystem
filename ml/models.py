"""模型构建：4 个预训练主干的统一接口。

支持的主干
----------
============== ============================== ========== ==========
name           来源                            参数量      分类头参数前缀
============== ============================== ========== ==========
resnet50       torchvision IMAGENET1K_V2       25.6 M     ``fc.``
efficientnet_b0 timm tf_efficientnet_b0         5.3 M     ``classifier.``
convnext_tiny  timm convnext_tiny.fb_in22k_ft_in1k 28.6 M  ``head.``
vit_b16        timm vit_base_patch16_224.augreg_in21k_ft_in1k 86 M ``head.``
============== ============================== ========== ==========

设计要点
--------
* 统一签名 :func:`build_model`，配置文件里只写名字即可切换。
* :func:`get_param_groups` 实现分层学习率：主干小 lr、新分类头大 lr。
* :func:`freeze_backbone` / :func:`unfreeze_backbone` 支持 warmup 阶段先冻结主干。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

NUM_CLASSES = 102

# 分类头参数名前缀（用于拆分 param_groups 与冻结主干）
HEAD_PREFIXES: dict[str, tuple[str, ...]] = {
    "resnet50": ("fc.",),
    "efficientnet_b0": ("classifier.",),
    "convnext_tiny": ("head.",),
    "vit_b16": ("head.",),
}

# 各主干的推荐默认超参（configs/*.yaml 会覆盖）
DEFAULTS: dict[str, dict[str, object]] = {
    "resnet50": {
        "batch_size": 32, "epochs": 60, "lr_backbone": 1e-4, "lr_head": 1e-3,
        "image_size": 224, "weight_decay": 0.05, "display_name": "ResNet-50",
    },
    "efficientnet_b0": {
        "batch_size": 32, "epochs": 60, "lr_backbone": 1e-4, "lr_head": 1e-3,
        "image_size": 224, "weight_decay": 0.05, "display_name": "EfficientNet-B0",
    },
    "convnext_tiny": {
        "batch_size": 24, "epochs": 60, "lr_backbone": 5e-5, "lr_head": 1e-3,
        "image_size": 224, "weight_decay": 0.05, "display_name": "ConvNeXt-Tiny",
    },
    "vit_b16": {
        "batch_size": 16, "epochs": 40, "lr_backbone": 3e-5, "lr_head": 1e-3,
        "image_size": 224, "weight_decay": 0.05, "display_name": "ViT-B/16",
    },
}

SUPPORTED_MODELS: tuple[str, ...] = tuple(HEAD_PREFIXES)


@dataclass
class ModelInfo:
    name: str
    display_name: str
    params_m: float
    head_param_names: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 主干构建
# --------------------------------------------------------------------------
def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_timm(name: str, timm_name: str, num_classes: int, pretrained: bool,
                drop_rate: float = 0.0, drop_path_rate: float = 0.0) -> nn.Module:
    import timm

    return timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    *,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """按名字构建模型（已替换为 ``num_classes`` 分类头）。"""
    if name not in HEAD_PREFIXES:
        raise ValueError(f"不支持的模型 {name!r}，可选：{', '.join(SUPPORTED_MODELS)}")

    if name == "resnet50":
        return _build_resnet50(num_classes, pretrained)
    if name == "efficientnet_b0":
        return _build_timm("efficientnet_b0", "tf_efficientnet_b0", num_classes,
                           pretrained, drop_rate, drop_path_rate)
    if name == "convnext_tiny":
        return _build_timm("convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k", num_classes,
                           pretrained, drop_rate, drop_path_rate)
    if name == "vit_b16":
        return _build_timm("vit_b16", "vit_base_patch16_224.augreg_in21k_ft_in1k",
                           num_classes, pretrained, drop_rate, drop_path_rate)
    raise ValueError(name)  # pragma: no cover


# --------------------------------------------------------------------------
# 参数分组 / 冻结
# --------------------------------------------------------------------------
def head_param_names(model: nn.Module, name: str) -> list[str]:
    prefixes = HEAD_PREFIXES[name]
    return [p for p, _ in model.named_parameters() if p.startswith(prefixes)]


def get_param_groups(
    model: nn.Module,
    name: str,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> list[dict]:
    """分层学习率参数组：主干用 ``lr_backbone``，新分类头用 ``lr_head``。

    偏置项与归一化层不做 weight decay（常规做法，能稳定微调）。
    """
    prefixes = HEAD_PREFIXES[name]
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = pname.startswith(prefixes)
        is_no_decay = pname.endswith(".bias") or "norm" in pname.lower() or "bn" in pname.lower()
        bucket = (
            (no_decay_head if is_no_decay else decay_head) if is_head
            else (no_decay_backbone if is_no_decay else decay_backbone)
        )
        bucket.append(param)

    groups: list[dict] = []
    for params, lr, wd, tag in (
        (decay_backbone, lr_backbone, weight_decay, "backbone/decay"),
        (no_decay_backbone, lr_backbone, 0.0, "backbone/no_decay"),
        (decay_head, lr_head, weight_decay, "head/decay"),
        (no_decay_head, lr_head, 0.0, "head/no_decay"),
    ):
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd, "group_name": tag})
    return groups


def freeze_backbone(model: nn.Module, name: str) -> int:
    """冻结主干，仅训练分类头。返回被冻结的参数张量数。"""
    prefixes = HEAD_PREFIXES[name]
    frozen = 0
    for pname, param in model.named_parameters():
        if not pname.startswith(prefixes):
            param.requires_grad = False
            frozen += 1
    return frozen


def unfreeze_backbone(model: nn.Module) -> None:
    """解冻全部参数。"""
    for param in model.parameters():
        param.requires_grad = True


# --------------------------------------------------------------------------
# 统计信息
# --------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_info(model: nn.Module, name: str) -> ModelInfo:
    total, _ = count_parameters(model)
    return ModelInfo(
        name=name,
        display_name=str(DEFAULTS[name]["display_name"]),
        params_m=round(total / 1e6, 2),
        head_param_names=head_param_names(model, name),
    )


# --------------------------------------------------------------------------
# 自检入口：python -m ml.models [--no-pretrained] [--only NAME]
# --------------------------------------------------------------------------
def _self_check(pretrained: bool = True, only: str | None = None) -> int:
    print(f"=== 模型构建自检（pretrained={pretrained}）===")
    names = [only] if only else list(SUPPORTED_MODELS)
    ok = True
    for name in names:
        try:
            model = build_model(name, pretrained=pretrained)
            info = model_info(model, name)
            default = DEFAULTS[name]
            frozen = freeze_backbone(model, name)
            _, trainable_after_freeze = count_parameters(model)
            unfreeze_backbone(model)
            _, trainable = count_parameters(model)

            # 前向验证：随机输入应输出 [B, 102]
            model.eval()
            with torch.no_grad():
                x = torch.randn(2, 3, int(default["image_size"]), int(default["image_size"]))
                y = model(x)
            assert y.shape == (2, NUM_CLASSES), f"输出形状 {tuple(y.shape)} 不符"

            groups = get_param_groups(
                model, name, float(default["lr_backbone"]), float(default["lr_head"]),
                float(default["weight_decay"]),
            )
            gdesc = ", ".join(f"{g['group_name']}={len(g['params'])}张" for g in groups)
            print(
                f"✓ {name:16s} {info.display_name:14s} 参数 {info.params_m:6.2f}M  "
                f"可训练 {trainable / 1e6:6.2f}M  冻结主干后 {trainable_after_freeze / 1e6:5.2f}M  "
                f"输出 {tuple(y.shape)}"
            )
            print(f"  分类头参数：{len(info.head_param_names)} 项（如 {info.head_param_names[:2]}）")
            print(f"  参数组：{gdesc}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"✗ {name:16s} 失败：{type(exc).__name__}: {exc}")
    print("=== 自检" + ("通过 ===" if ok else "存在失败 ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="模型构建自检")
    ap.add_argument("--no-pretrained", action="store_true", help="不加载预训练权重（离线快测）")
    ap.add_argument("--only", default=None, help="只测某个模型")
    a = ap.parse_args()
    raise SystemExit(_self_check(pretrained=not a.no_pretrained, only=a.only))
