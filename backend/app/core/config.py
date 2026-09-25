"""应用配置。

设计要点
--------
1. 所有可变项走 ``.env``（见项目根 ``.env.example``），代码内不硬编码。
2. **所有相对路径统一以项目根目录为基准解析**，与启动时的 CWD 无关 ——
   这样无论用 ``uvicorn --app-dir backend app.main:app`` 还是从 ``backend/`` 启动，
   权重、数据库、上传目录、百科数据都能正确找到。
3. 把项目根加入 ``sys.path``，保证后端能 ``import ml.*``（推理复用训练侧预处理）。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py  →  parents[3] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 让 ``import ml.*`` 在任何启动方式下都可用（推理需要复用 ml/dataset.py 的预处理）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings(BaseSettings):
    """从 ``.env`` 读取的全局配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "AI 花卉识别系统"
    app_env: str = "dev"
    debug: bool = True
    api_prefix: str = "/api/v1"

    # ---------- 安全 ----------
    secret_key: str = "dev-only-please-change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 720

    # ---------- 数据库 ----------
    database_url: str = "sqlite:///./flowers.db"

    # ---------- 上传 ----------
    upload_dir: str = "backend/app/uploads"
    max_upload_mb: int = 5
    allowed_extensions: str = ".jpg,.jpeg,.png,.webp"

    # ---------- 模型 ----------
    default_model: str = "vit_b16"
    weights_dir: str = "ml/outputs/checkpoints"
    metrics_json: str = "ml/outputs/metrics.json"
    device: str = "auto"

    # ---------- CORS ----------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------- 百科 ----------
    flowers_json: str = "backend/app/data/flowers.json"

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        out: set[str] = set()
        for ext in self.allowed_extensions.split(","):
            ext = ext.strip().lower()
            if not ext:
                continue
            out.add(ext if ext.startswith(".") else f".{ext}")
        return out

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def path(self, value: str | Path) -> Path:
        """把相对路径解析为相对**项目根**的绝对路径。"""
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def upload_path(self) -> Path:
        return self.path(self.upload_dir)

    @property
    def weights_path(self) -> Path:
        return self.path(self.weights_dir)

    @property
    def metrics_path(self) -> Path:
        return self.path(self.metrics_json)

    @property
    def flowers_path(self) -> Path:
        return self.path(self.flowers_json)

    @property
    def sqlalchemy_url(self) -> str:
        """把 ``sqlite:///./x.db`` 之类的相对路径改写为绝对路径。"""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            rel = url[len(prefix):]
            return f"{prefix}{self.path(rel).as_posix()}"
        return url

    def describe(self) -> dict[str, str]:
        """启动时打印用（不含密钥明文）。"""
        return {
            "app_env": self.app_env,
            "debug": str(self.debug),
            "api_prefix": self.api_prefix,
            "database": self.sqlalchemy_url,
            "upload_dir": str(self.upload_path),
            "weights_dir": str(self.weights_path),
            "metrics_json": str(self.metrics_path),
            "flowers_json": str(self.flowers_path),
            "default_model": self.default_model,
            "device": self.device,
            "max_upload_mb": str(self.max_upload_mb),
            "secret_key": "***已设置***" if self.secret_key else "***未设置***",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
