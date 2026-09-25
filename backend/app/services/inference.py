"""推理服务：模型单例 + 与训练**完全一致**的预处理。

设计要点
--------
1. **模型单例**：由 ``app.main`` 的 lifespan 启动时加载一次并常驻显存，
   请求内只做前向，绝不在请求里重新 ``load_state_dict``。
2. **预处理唯一实现**：直接调用 ``ml.dataset.build_inference_transform``，
   与训练/评估用的是同一份代码 —— 这是"报告 99%、线上 70%"这类事故的根本防线
   （docs/03 §3.3 铁律）。
3. **GPU 串行化**：用 ``threading.Lock`` 串行化前向，避免并发请求争抢显存（NFR-03）。
4. **加载失败不阻断启动**：记录 ``load_error``，识别接口返回明确错误码 5001，
   服务本身仍可用于百科/模型对比等只读功能（NFR-04）。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from app.core.config import settings
from app.core.errors import APIError, ErrorCode
from app.core.logging import get_logger

log = get_logger("inference")


class InferenceService:
    """进程内唯一的推理服务实例。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transform = None

        self.model = None
        self.model_name: str | None = None
        self.model_display: str | None = None
        self.model_pretrained: str | None = None
        self.image_size: int = 224
        self.device: torch.device | None = None
        self.class_names: list[str] = []

        self.is_loaded: bool = False
        self.load_error: str | None = None

        # 复用的输入缓冲：保持显存地址稳定（见 _input_tensor 的说明）
        self._input_buffer: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def _metrics_entry(self, name: str) -> dict | None:
        path: Path = settings.metrics_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("读取 metrics.json 失败：%s", exc)
            return None
        for m in payload.get("models", []):
            if m.get("name") == name:
                return m
        return None

    def available_models(self) -> list[str]:
        """metrics.json 中记录的模型名列表。"""
        path = settings.metrics_path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [m.get("name") for m in payload.get("models", []) if m.get("name")]

    def _resolve_weights(self, name: str, entry: dict | None) -> Path:
        if entry and entry.get("weights_path"):
            return settings.path(entry["weights_path"])
        return settings.weights_path / f"{name}_best.pth"

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(self, model_name: str | None = None) -> bool:
        """加载（或切换）模型。失败时设置 ``load_error`` 并返回 False，不抛异常。"""
        from ml.dataset import build_inference_transform, get_class_names
        from ml.models import build_model

        name = model_name or settings.default_model
        try:
            entry = self._metrics_entry(name)
            weights = self._resolve_weights(name, entry)
            if not weights.exists():
                raise FileNotFoundError(
                    f"权重文件不存在：{weights}（可用模型：{self.available_models()}）"
                )

            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            ckpt_name = ckpt.get("model_name", name)
            num_classes = int(ckpt.get("num_classes", 102))
            image_size = int(ckpt.get("image_size", 224))

            model = build_model(ckpt_name, num_classes, pretrained=False)
            model.load_state_dict(ckpt["state_dict"])

            device = self._resolve_device()
            model.to(device).eval()

            self.model = model
            self.model_name = name
            self.model_display = (entry or {}).get("display_name", ckpt_name)
            self.model_pretrained = (entry or {}).get("pretrained", "")
            self.image_size = image_size
            self.device = device
            self.class_names = list(ckpt.get("class_names") or get_class_names())
            self._transform = build_inference_transform(image_size)
            self.is_loaded = True
            self.load_error = None

            log.info(
                "模型已加载：%s（%s）| 权重=%s | 设备=%s | 输入=%d | 类别=%d",
                self.model_name, self.model_display, weights.name, device, image_size,
                len(self.class_names),
            )
            self._warmup()
            return True
        except Exception as exc:  # noqa: BLE001
            self.is_loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("模型加载失败（%s）：%s", name, self.load_error)
            return False

    def _resolve_device(self) -> torch.device:
        from ml.utils import ensure_cache_env, resolve_device

        ensure_cache_env()
        return resolve_device(settings.device)

    def _warmup(self, runs: int = 5) -> None:
        """启动时预热若干次前向。

        **为什么必须做**：首次前向会触发 cuDNN 算法选择、显存分配与 kernel 编译，
        实测冷启动 **495.7 ms**，而稳态仅 **10.14 ms**（M2-6 实测）—— 相差近 50 倍。
        若不预热，答辩现场第一次上传图片会明显卡顿，甚至被误判为"系统很慢"。
        """
        if self.model is None or self._transform is None or self.device is None:
            return
        try:
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            # 关键：预热就用 _input_tensor 分配那块**将被后续请求复用**的缓冲区。
            # 若预热另用临时张量，首个真实请求仍会因新地址触发一次 cuDNN 重选
            # （实测首次 218 ms，之后才降到 13 ms）。
            buf = self._input_tensor(dummy)
            use_amp = self.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=use_amp):
                for _ in range(runs):
                    self.model(buf)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            log.info("模型预热完成（%d 次前向，缓冲区地址已固定，消除冷启动延迟）", runs)
        except Exception as exc:  # noqa: BLE001
            log.warning("模型预热失败（不影响功能，仅首次请求稍慢）：%s", exc)

    def ensure_model(self, model_name: str | None) -> None:
        """按需切换模型（并发下用锁保护，避免重复加载）。"""
        if not model_name or model_name == self.model_name:
            return
        allowed = self.available_models()
        if allowed and model_name not in allowed:
            raise APIError(
                ErrorCode.BAD_REQUEST,
                f"未知模型 {model_name!r}，可选：{'、'.join(allowed)}",
            )
        log.info("切换模型：%s → %s", self.model_name, model_name)
        self.load(model_name)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _input_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """把预处理结果原地拷进**固定缓冲区**后返回。

        **为什么不能每次新建张量**：训练侧开启了 ``torch.backends.cudnn.benchmark=True``，
        cuDNN 会按输入张量的**显存地址**缓存/重选卷积算法。若每个请求都新建张量，
        地址不断变化会反复触发算法重选 —— 实测表现为前几次请求稳定停在 ~67 ms，
        之后才降到 ~13 ms（见下方 predict 的注释与实测数据）。

        复用同一块缓冲区后地址恒定，第一次请求即达到稳态延迟。
        配合推理用的 ``threading.Lock`` 串行化，不存在并发写同一缓冲的风险。
        """
        if (
            self._input_buffer is None
            or self._input_buffer.shape != tensor.shape
            or self._input_buffer.device != tensor.device
        ):
            self._input_buffer = torch.empty_like(tensor)
        self._input_buffer.copy_(tensor)
        return self._input_buffer

    @torch.no_grad()
    def predict(self, image: Image.Image, *, topk: int = 5) -> tuple[list[dict], float]:
        """对单张 PIL 图片推理，返回 ``(Top-K 列表, 耗时毫秒)``。

        Top-K 项为 ``{"class_id", "name_en", "confidence"}``；中文名由调用方补。
        """
        if not self.is_loaded or self.model is None:
            raise APIError(
                ErrorCode.MODEL_NOT_LOADED,
                f"模型未加载，无法识别（{self.load_error or '未初始化'}）",
                http_status=503,
            )

        try:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        except Exception as exc:  # noqa: BLE001
            raise APIError(ErrorCode.BROKEN_IMAGE, "图片无法预处理，请更换图片") from exc

        # 拷进固定缓冲区：地址稳定 → cuDNN 不反复重选算法（见 _input_tensor 说明）
        tensor = self._input_tensor(tensor.to(self.device, non_blocking=True))
        use_amp = self.device.type == "cuda"

        # 计时只覆盖**推理本身**：先 synchronize 冲掉上面那次异步 H2D 拷贝，
        # 否则末尾的 synchronize 会把拷贝耗时也算进 latency_ms（实测虚高到 55—65 ms）。
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        started = time.perf_counter()
        with self._lock:  # 串行化 GPU 前向
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                logits = self.model(tensor)
            probs = torch.softmax(logits.float(), dim=1)
            if self.device.type == "cuda":
                # CUDA 算子是异步的：不等 GPU 真正算完就停表，测到的只是 kernel
                # 启动耗时，会**低估**真实推理延迟。作为报告中的 NFR-01 指标必须同步。
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        k = min(topk, probs.size(1))
        confidences, indices = probs.topk(k, dim=1)
        confidences = confidences[0].tolist()
        indices = indices[0].tolist()

        results = []
        for cid, conf in zip(indices, confidences):
            results.append(
                {
                    "class_id": int(cid),
                    "name_en": self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}",
                    "confidence": round(float(conf), 4),
                }
            )
        return results, round(latency_ms, 2)


# --------------------------------------------------------------------------
# 单例
# --------------------------------------------------------------------------
_service: InferenceService | None = None
_service_lock = threading.Lock()


def get_engine_service() -> InferenceService:
    """取进程内唯一实例（首次调用时创建，不自动加载模型）。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service
