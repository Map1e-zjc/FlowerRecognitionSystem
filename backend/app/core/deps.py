"""FastAPI 依赖注入：当前用户解析。

按需求（docs/01 FR-04）：百科与模型对比**无需登录**；识别与历史记录**必须登录**。
因此同时提供"必须登录"与"可选登录"两个依赖。
"""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.errors import APIError, ErrorCode
from app.core.security import TokenError, decode_access_token
from app.db.session import get_db
from app.models import User

# auto_error=False：未携带令牌时返回 None，由我们给出统一中文提示
_bearer = HTTPBearer(auto_error=False, description="Bearer JWT，格式：`Bearer <token>`")


def _user_from_credentials(
    credentials: HTTPAuthorizationCredentials | None, db: Session
) -> User:
    if credentials is None or not credentials.credentials:
        raise APIError(
            ErrorCode.UNAUTHORIZED, "请先登录后再使用该功能",
            http_status=401,
        )
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        code = ErrorCode.TOKEN_EXPIRED if exc.expired else ErrorCode.UNAUTHORIZED
        raise APIError(code, exc.message, http_status=401) from exc

    uid = payload.get("uid")
    if not isinstance(uid, int):
        raise APIError(ErrorCode.UNAUTHORIZED, "登录凭证无效，请重新登录", http_status=401)

    user = db.get(User, uid)
    if user is None:
        # 用户被删除但令牌仍在有效期内
        raise APIError(ErrorCode.UNAUTHORIZED, "账号不存在或已被删除", http_status=401)
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """必须登录；否则 401 中文提示。"""
    return _user_from_credentials(credentials, db)


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User | None:
    """可选登录：未携带令牌返回 None，携带但无效仍报 401（避免静默降级）。"""
    if credentials is None or not credentials.credentials:
        return None
    return _user_from_credentials(credentials, db)
