"""用户与认证相关的请求/响应模型。"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 用正则而非 EmailStr，避免为此引入 email-validator 依赖
EMAIL_PATTERN = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
USERNAME_PATTERN = r"^[A-Za-z0-9_\u4e00-\u9fa5]+$"


class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, description="用户名，3—20 位")
    email: str = Field(..., max_length=120, pattern=EMAIL_PATTERN, description="邮箱")
    password: str = Field(..., min_length=8, max_length=64, description="密码，≥8 位且含字母与数字")

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        v = v.strip()
        if not re.match(USERNAME_PATTERN, v):
            raise ValueError("用户名只能包含中文、字母、数字与下划线")
        return v

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        from app.core.security import check_password_strength

        ok, reason = check_password_strength(v)
        if not ok:
            raise ValueError(reason)
        return v


class LoginIn(BaseModel):
    # 允许用"用户名或邮箱"登录，字段名沿用契约里的 username 以保持前端一致
    username: str = Field(..., min_length=1, max_length=120, description="用户名或邮箱")
    password: str = Field(..., min_length=1, max_length=64)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="有效期（秒）")
    user: UserOut
