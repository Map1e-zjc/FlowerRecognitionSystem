"""FastAPI 应用装配：CORS、异常处理、路由挂载、静态资源、健康检查。

启动方式（在**项目根目录**执行）：

    python -m uvicorn --app-dir backend app.main:app --reload --port 8000

``--app-dir backend`` 让 ``app.*`` 可导入，同时保持 CWD 在项目根，
使 ``ml.*`` 与相对数据路径都能正确解析（见 app/core/config.py）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.core.config import PROJECT_ROOT, settings
from app.core.errors import ok, register_exception_handlers
from app.core.logging import get_logger, setup_logging

log = get_logger("main")

STATIC_DIR = PROJECT_ROOT / "backend" / "app" / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    log.info("=" * 74)
    log.info("%s 启动中…", settings.app_name)
    for key, value in settings.describe().items():
        log.info("  %-14s = %s", key, value)

    # 建表（幂等）。种子数据用 python -m app.db.init_db 导入
    # 注意：必须先 import app.models，否则 Base.metadata 为空、一张表都建不出来
    from app import models  # noqa: F401  仅为注册 ORM 模型到 metadata
    from app.db.session import Base, engine

    Base.metadata.create_all(bind=engine)
    log.info("  数据表          = %s", ", ".join(sorted(Base.metadata.tables)) or "（无）")

    # 预加载推理模型：失败也不阻止服务启动（NFR-04）
    try:
        from app.services.inference import get_engine_service

        get_engine_service().load()
    except ImportError:
        log.warning("  推理服务         = 尚未实现（M3-4），/predict 暂不可用")
    except Exception as exc:  # noqa: BLE001
        log.error("  模型加载失败     = %s: %s（服务仍会启动，识别接口将返回明确错误）",
                  type(exc).__name__, exc)
    log.info("=" * 74)
    yield
    log.info("%s 正在关闭…", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="基于深度迁移学习的花卉识别系统后端 API。上传图片返回 Top-5 候选种类与置信度。",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    log.info("静态资源目录已挂载：%s → /static", STATIC_DIR)

# 用户上传的图片（识别结果缩略图）。目录可能在首次上传时才创建，故先建好再挂载
settings.upload_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.upload_path)), name="uploads")
log.info("上传目录已挂载：%s → /uploads", settings.upload_path)

# 训练产出的图表（混淆矩阵、训练曲线、对比柱状图、错误案例），供"模型对比"页展示
FIGURES_DIR = PROJECT_ROOT / "ml" / "outputs" / "figures"
if FIGURES_DIR.is_dir():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
    log.info("图表目录已挂载：%s → /figures", FIGURES_DIR)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/healthz", tags=["系统"], summary="健康检查")
def healthz() -> dict:
    """报告数据库、GPU、模型加载与百科数据状态。"""
    info: dict = {
        "app": settings.app_name,
        "env": settings.app_env,
        "api_prefix": settings.api_prefix,
    }

    # --- 数据库与百科 ---
    try:
        from sqlalchemy import func, select

        from app.db.session import SessionLocal
        from app.models import Flower, ModelMetric

        with SessionLocal() as db:
            info["flowers"] = db.scalar(select(func.count()).select_from(Flower)) or 0
            info["model_metrics"] = db.scalar(select(func.count()).select_from(ModelMetric)) or 0
        info["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["database"] = f"error: {type(exc).__name__}: {exc}"
        info["flowers"] = 0
        info["model_metrics"] = 0

    # --- GPU ---
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if info["gpu_available"] else None
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"import 失败：{type(exc).__name__}"
        info["gpu_available"] = False
        info["gpu_name"] = None

    # --- 模型 ---
    info["model_loaded"] = False
    info["model_name"] = None
    info["model_error"] = None
    try:
        from app.services.inference import get_engine_service

        svc = get_engine_service()
        info["model_loaded"] = svc.is_loaded
        info["model_name"] = svc.model_name
        info["model_error"] = svc.load_error
    except ImportError:
        info["model_error"] = "推理服务尚未实现（M3-4）"
    except Exception as exc:  # noqa: BLE001
        info["model_error"] = f"{type(exc).__name__}: {exc}"

    return ok(info)
