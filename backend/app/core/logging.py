"""日志配置：控制台 + 轮转文件，统一 UTF-8 与中文友好格式。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT, settings

_LOG_DIR = PROJECT_ROOT / "backend" / "logs"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int | None = None) -> logging.Logger:
    """初始化根日志（幂等，可重复调用）。"""
    global _configured
    logger = logging.getLogger("flower")
    if _configured:
        return logger

    lvl = level if level is not None else (logging.DEBUG if settings.debug else logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(lvl)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_DIR / "backend.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.setLevel(lvl)
    logger.propagate = False

    # 让 uvicorn 的访问日志走同一套格式
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.addHandler(console)
        uv.propagate = False

    _configured = True
    logger.info("日志已初始化：level=%s，文件=%s", logging.getLevelName(lvl), _LOG_DIR / "backend.log")
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """取子 logger（自动挂到 ``flower`` 下）。"""
    setup_logging()
    return logging.getLogger(f"flower.{name}" if name else "flower")
