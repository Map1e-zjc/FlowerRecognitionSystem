"""pytest 全局夹具。

关键点：**必须在导入 ``app`` 之前**改写环境变量，指向独立的临时数据库与上传目录，
否则测试会污染开发用的 ``flowers.db`` 与 ``uploads/``。
``app.core.config`` 在导入时就把 ``.env`` 读进 ``settings`` 单例，所以顺序不能反。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1) 先改环境变量，再导入 app
# ---------------------------------------------------------------------------
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="flowers_test_"))
_TMP_DB = _TMP_ROOT / "test.db"
_TMP_UPLOADS = _TMP_ROOT / "uploads"

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_UPLOADS)
os.environ["APP_ENV"] = "test"
os.environ["DEBUG"] = "false"
os.environ.setdefault("PYTHONUTF8", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "flowers-102" / "jpg" / "image_00001.jpg"


@pytest.fixture(scope="session", autouse=True)
def _prepare_database():
    """建表 + 导入种子（花卉百科与模型指标），测试结束后清理临时目录。"""
    from app.db.init_db import create_tables, seed_flowers, seed_model_metrics
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        create_tables(drop=True)
        seed_flowers(db)
        seed_model_metrics(db)
    yield
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def client():
    """带 lifespan 的 TestClient（会加载模型，耗时约 15—25 秒，只做一次）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def api():
    """API 前缀。"""
    from app.core.config import settings

    return settings.api_prefix


@pytest.fixture(scope="session")
def model_ready(client, api) -> bool:
    """模型是否成功加载（未加载时跳过依赖推理的用例，而不是让整套测试失败）。"""
    data = client.get("/healthz").json().get("data", {})
    return bool(data.get("model_loaded"))


@pytest.fixture
def user_token(client, api):
    """每个用例注册一个独立用户并返回 (headers, username)。"""
    import random
    import string

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    username = f"t_{suffix}"
    password = "Pytest12345"
    r = client.post(
        f"{api}/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    token = client.post(
        f"{api}/auth/login", json={"username": username, "password": password}
    ).json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}, username


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"缺少测试图片：{SAMPLE_IMAGE}（请先准备数据集）")
    return SAMPLE_IMAGE.read_bytes()
