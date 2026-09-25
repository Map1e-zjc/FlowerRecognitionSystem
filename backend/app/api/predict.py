"""识别接口（核心，FR-01）。

处理链：认证 → 校验文件 → 解码/纠正方向 → 存盘 → 推理 → 关联中文名 → 落库 → 返回 Top-5。
每一步的失败都返回明确的中文业务码，不抛 500 堆栈（NFR-04）。
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.errors import APIError, Envelope, ErrorCode, ok
from app.core.logging import get_logger
from app.db.session import get_db
from app.models import ModelMetric, RecognitionRecord, User
from app.schemas.predict import PredictOut, PredictionItem
from app.services.inference import get_engine_service
from app.services.knowledge import flowers_map

log = get_logger("predict")
router = APIRouter()

# JPEG/PNG/WEBP 为主；MPO 是部分手机拍出的多帧 JPEG，按 JPEG 处理
_ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}


# --------------------------------------------------------------------------
# 上传校验
# --------------------------------------------------------------------------
def _check_suffix(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    allowed = settings.allowed_extension_set
    if ext not in allowed:
        raise APIError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            f"不支持的文件类型 {ext or '（无扩展名）'}，仅支持 "
            + "、".join(sorted(allowed)),
        )
    return ext


async def _read_limited(file: UploadFile) -> bytes:
    """读取上传内容，超过上限立即中止（不把超大文件读进内存）。"""
    limit = settings.max_upload_bytes
    data = await file.read(limit + 1)
    if not data:
        raise APIError(ErrorCode.UNSUPPORTED_FILE_TYPE, "上传内容为空，请选择一张图片")
    if len(data) > limit:
        raise APIError(
            ErrorCode.FILE_TOO_LARGE,
            f"图片大小超过 {settings.max_upload_mb} MB 限制",
            http_status=413,
        )
    return data


def _decode_image(data: bytes) -> Image.Image:
    """校验可解码性并做 EXIF 方向纠正。"""
    try:
        with Image.open(BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()  # 校验文件完整性
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise APIError(ErrorCode.BROKEN_IMAGE, "图片已损坏或不是有效图片") from exc

    if fmt not in _ALLOWED_PIL_FORMATS:
        raise APIError(
            ErrorCode.UNSUPPORTED_FILE_TYPE, f"不支持的图片格式 {fmt or '未知'}"
        )

    try:
        img = Image.open(BytesIO(data))
        # verify() 后对象不可用，这里重新打开；exif_transpose 按 EXIF 摆正方向
        img = ImageOps.exif_transpose(img) or img
        return img.convert("RGB")
    except (OSError, ValueError) as exc:
        raise APIError(ErrorCode.BROKEN_IMAGE, "图片解码失败，请更换图片") from exc


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=Envelope[PredictOut],
    summary="上传花卉图片识别（返回 Top-5）",
    responses={
        400: {"description": "文件类型不支持 / 图片损坏"},
        401: {"description": "未登录"},
        413: {"description": "文件超过大小限制"},
        503: {"description": "模型未加载"},
    },
)
async def predict(
    file: UploadFile = File(..., description="花卉图片，JPG/PNG/WEBP，≤5 MB"),
    model_name: str | None = Form(None, description="可选：指定模型，默认使用最优模型"),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ext = _check_suffix(file.filename)
    data = await _read_limited(file)
    image = _decode_image(data)

    service = get_engine_service()
    service.ensure_model(model_name)
    if not service.is_loaded:
        raise APIError(
            ErrorCode.MODEL_NOT_LOADED,
            f"模型未加载，无法识别（{service.load_error or '未初始化'}）",
            http_status=503,
        )

    # 存盘：按用户分目录 + UUID 文件名，杜绝路径穿越与重名（NFR-05）
    user_dir = settings.upload_path / str(current.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{ext}"
    (user_dir / stored_name).write_bytes(data)
    relative_path = f"{current.id}/{stored_name}"

    raw_predictions, latency_ms = service.predict(image)

    # 关联中文百科信息（一次查询拿全 5 条）
    flower_map = flowers_map(db, [p["class_id"] for p in raw_predictions])
    predictions: list[dict] = []
    for item in raw_predictions:
        flower = flower_map.get(item["class_id"])
        predictions.append(
            {
                "class_id": item["class_id"],
                "name_en": (flower.name_en if flower else item["name_en"]),
                "name_cn": (flower.name_cn if flower else "未知"),
                "confidence": item["confidence"],
            }
        )

    top1 = predictions[0]
    record = RecognitionRecord(
        user_id=current.id,
        image_path=relative_path,
        predicted_class_id=top1["class_id"],
        confidence=top1["confidence"],
        top5_json=json.dumps(predictions, ensure_ascii=False),
        model_name=service.model_name or "",
        latency_ms=latency_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    log.info(
        "识别完成：user=%s record=%s 模型=%s Top-1=%s(%.2f%%) 耗时=%.1fms",
        current.username, record.id, service.model_name,
        top1["name_cn"], top1["confidence"] * 100, latency_ms,
    )

    # 展示信息以数据库为准（pretrained 等字段来自 init_db 的种子，metrics.json 里没有）
    metric = db.scalar(select(ModelMetric).where(ModelMetric.name == service.model_name))
    payload = {
        "record_id": record.id,
        "model_name": service.model_name or "",
        "model_display_name": (metric.display_name if metric else None)
        or service.model_display or "",
        "model_pretrained": (metric.pretrained if metric else "") or "",
        "latency_ms": latency_ms,
        "image_url": f"/uploads/{relative_path}",
        "predictions": [PredictionItem(**p).model_dump() for p in predictions],
    }
    return ok(payload)
