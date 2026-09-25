"""认证接口：注册 / 登录 / 当前用户。

契约见 docs/02 §4.1。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.errors import APIError, Envelope, ErrorCode, ok
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models import User
from app.schemas.user import LoginIn, RegisterIn, TokenOut, UserOut

log = get_logger("auth")
router = APIRouter()


@router.post(
    "/register",
    response_model=Envelope[UserOut],
    summary="注册新用户",
    responses={409: {"description": "用户名或邮箱已被占用"}},
)
def register(payload: RegisterIn, db: Session = Depends(get_db)) -> dict:
    exists = db.scalar(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.email)
        )
    )
    if exists is not None:
        if exists.username == payload.username:
            raise APIError(ErrorCode.CONFLICT, "该用户名已被注册", http_status=409)
        raise APIError(ErrorCode.CONFLICT, "该邮箱已被注册", http_status=409)

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log.info("新用户注册：id=%s username=%s", user.id, user.username)
    return ok(UserOut.model_validate(user).model_dump(mode="json"), message="注册成功")


@router.post(
    "/login",
    response_model=Envelope[TokenOut],
    summary="登录并获取访问令牌",
    responses={401: {"description": "用户名或密码错误"}},
)
def login(payload: LoginIn, db: Session = Depends(get_db)) -> dict:
    user = db.scalar(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.username)
        )
    )
    # 用户不存在与密码错误返回同一提示，避免用户名枚举
    if user is None or not verify_password(payload.password, user.password_hash):
        log.info("登录失败：identifier=%s", payload.username)
        raise APIError(ErrorCode.BAD_CREDENTIALS, "用户名或密码错误", http_status=401)

    token, expire = create_access_token(user_id=user.id, username=user.username)
    expires_in = settings.access_token_expire_minutes * 60
    log.info("登录成功：id=%s username=%s 有效期至 %s", user.id, user.username, expire.isoformat())
    return ok(
        {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "user": UserOut.model_validate(user).model_dump(mode="json"),
        },
        message="登录成功",
    )


@router.get("/me", response_model=Envelope[UserOut], summary="获取当前登录用户")
def me(current: User = Depends(get_current_user)) -> dict:
    return ok(UserOut.model_validate(current).model_dump(mode="json"))
