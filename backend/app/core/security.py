"""安全工具：密码哈希与 JWT 签发/校验。

* 密码用 **bcrypt** 哈希，数据库不存明文（NFR-05）。
* JWT 用 HS256，载荷含 ``sub``（用户名）、``uid``（用户 ID）、``exp``。
* bcrypt 固定 4.0.1：passlib 1.7.4 会读取 ``bcrypt.__about__.__version__``，
  该属性在 bcrypt ≥ 4.1 被移除，会导致版本探测报错（见 docs/05 §4.2）。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 密码强度：≥8 位，且同时包含字母与数字
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")


# --------------------------------------------------------------------------
# 密码
# --------------------------------------------------------------------------
def hash_password(raw: str) -> str:
    """生成 bcrypt 哈希（含盐）。"""
    # bcrypt 只处理前 72 字节，超长密码先截断避免报错
    return _pwd_context.hash(raw[:72])


def verify_password(raw: str, hashed: str) -> bool:
    """校验明文与哈希是否匹配（任何异常都视为不匹配）。"""
    try:
        return _pwd_context.verify(raw[:72], hashed)
    except (ValueError, TypeError):
        return False


def check_password_strength(raw: str) -> tuple[bool, str]:
    """返回 ``(是否合格, 中文原因)``。"""
    if len(raw) < 8:
        return False, "密码长度至少 8 位"
    if len(raw) > 64:
        return False, "密码长度不能超过 64 位"
    if not _HAS_LETTER.search(raw):
        return False, "密码必须包含字母"
    if not _HAS_DIGIT.search(raw):
        return False, "密码必须包含数字"
    return True, ""


# --------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------
def create_access_token(*, user_id: int, username: str, expires_minutes: int | None = None):
    """签发访问令牌，返回 ``(token, 过期时间)``。"""
    minutes = expires_minutes or settings.access_token_expire_minutes
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=minutes)
    payload = {
        "sub": username,
        "uid": user_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token, expire


class TokenError(Exception):
    """JWT 校验失败（区分过期与其他错误，便于返回不同提示）。"""

    def __init__(self, message: str, *, expired: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.expired = expired


def decode_access_token(token: str) -> dict:
    """解析并校验令牌。失败时抛 :class:`TokenError`。"""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        text = str(exc).lower()
        if "expire" in text or "exp" in text:
            raise TokenError("登录状态已过期，请重新登录", expired=True) from exc
        raise TokenError("登录凭证无效，请重新登录") from exc
