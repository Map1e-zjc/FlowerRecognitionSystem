"""统一错误码、异常类型、响应封装与全局异常处理。

响应契约（docs/02 §4）
----------------------
成功：``{"code": 0, "message": "ok", "data": {...}}``
失败：``{"code": <业务码>, "message": "<中文提示>", "data": null}``
HTTP 状态码与业务码同时使用；未捕获异常绝不把堆栈返回前端（NFR-04）。
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

log = get_logger("errors")

T = TypeVar("T")

# Starlette 新版把 422 常量改名了，这里做兼容（否则会产生 DeprecationWarning）
_HTTP_422: int = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or 422


# --------------------------------------------------------------------------
# 业务错误码
# --------------------------------------------------------------------------
class ErrorCode:
    """业务码。4xxx = 客户端问题，5xxx = 服务端问题。"""

    OK = 0
    BAD_REQUEST = 4000
    UNSUPPORTED_FILE_TYPE = 4001
    FILE_TOO_LARGE = 4002
    BROKEN_IMAGE = 4003
    UNAUTHORIZED = 4010
    BAD_CREDENTIALS = 4011
    TOKEN_EXPIRED = 4012
    FORBIDDEN = 4030
    NOT_FOUND = 4040
    CONFLICT = 4090
    VALIDATION_ERROR = 4220
    INTERNAL_ERROR = 5000
    MODEL_NOT_LOADED = 5001
    INFERENCE_FAILED = 5002
    KNOWLEDGE_MISSING = 5003


# --------------------------------------------------------------------------
# 响应封装
# --------------------------------------------------------------------------
class Envelope(BaseModel, Generic[T]):
    """统一响应体，供 FastAPI ``response_model`` 使用（让 /docs 显示真实结构）。"""

    code: int = ErrorCode.OK
    message: str = "ok"
    data: T | None = None


def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    return {"code": ErrorCode.OK, "message": message, "data": data}


def fail(code: int, message: str) -> dict[str, Any]:
    return {"code": code, "message": message, "data": None}


class APIError(Exception):
    """业务异常：抛出后由全局处理器转成统一响应。"""

    def __init__(
        self,
        code: int,
        message: str,
        http_status: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


# --------------------------------------------------------------------------
# 全局异常处理
# --------------------------------------------------------------------------
_HTTP_TO_CODE = {
    status.HTTP_400_BAD_REQUEST: ErrorCode.BAD_REQUEST,
    status.HTTP_401_UNAUTHORIZED: ErrorCode.UNAUTHORIZED,
    status.HTTP_403_FORBIDDEN: ErrorCode.FORBIDDEN,
    status.HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND,
    status.HTTP_409_CONFLICT: ErrorCode.CONFLICT,
    _HTTP_422: ErrorCode.VALIDATION_ERROR,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=fail(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details: list[str] = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            details.append(f"{loc or '请求体'}: {err.get('msg', '格式错误')}")
        message = "请求参数校验失败" + ("（" + "；".join(details) + "）" if details else "")
        return JSONResponse(
            status_code=_HTTP_422,
            content=fail(ErrorCode.VALIDATION_ERROR, message),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_TO_CODE.get(exc.status_code, ErrorCode.BAD_REQUEST)
        raw = exc.detail if isinstance(exc.detail, str) else "请求失败"
        # 把常见的英文默认提示换成中文，避免前端出现英文
        message = {
            "Not Found": "接口不存在",
            "Method Not Allowed": "请求方法不被支持",
            "Unauthorized": "未登录或登录状态已失效",
            "Forbidden": "没有权限访问该资源",
        }.get(raw, raw)
        return JSONResponse(status_code=exc.status_code, content=fail(code, message))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("未捕获异常：%s %s → %s: %s",
                      request.method, request.url.path, type(exc).__name__, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=fail(ErrorCode.INTERNAL_ERROR, "服务器内部错误，请稍后重试或查看后端日志"),
        )
