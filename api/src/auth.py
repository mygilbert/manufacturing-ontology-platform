"""인증 / 주체 식별

Action 계층은 "누가"를 요구한다. 인증이 없으면 Principal을 만들 수 없고,
따라서 액션도 수행할 수 없다. 그것이 의도된 동작이다.

토큰은 JWT(HS256)를 사용한다. 클레임::

    {
      "sub": "hong.gildong",      # 사용자 ID (필수)
      "roles": ["process_engineer"],
      "name": "홍길동",
      "exp": 1790000000
    }

개발 편의를 위해 ``AUTH_DEV_MODE=true``이면 토큰 없이도 고정 주체로
동작한다. 운영에서는 반드시 꺼야 하며, 켜져 있으면 기동 시 경고가 남는다.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from common.actions import Principal
from config import settings

logger = logging.getLogger(__name__)

# 개발 모드: 토큰 없이 고정 주체로 동작 (운영 금지)
AUTH_DEV_MODE = os.getenv("AUTH_DEV_MODE", "false").lower() in ("1", "true", "yes")
DEV_PRINCIPAL = Principal(
    user_id="dev.user",
    roles=("ontology_admin", "process_engineer", "equipment_engineer",
           "shift_supervisor", "quality_engineer", "operator"),
    display_name="개발 모드 사용자",
)

# auto_error=False: 토큰이 없을 때 우리가 직접 메시지를 통제한다.
_bearer = HTTPBearer(auto_error=False)


def create_access_token(
    user_id: str,
    roles: Optional[List[str]] = None,
    display_name: str = "",
    expires_minutes: Optional[int] = None,
) -> str:
    """개발/테스트용 토큰 발급.

    운영에서는 사내 SSO가 발급한 토큰을 검증만 하는 것이 보통이다.
    """
    from jose import jwt

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.jwt_expire_minutes
    )
    payload = {
        "sub": user_id,
        "roles": roles or [],
        "name": display_name,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def principal_from_token(token: str) -> Principal:
    """JWT를 검증해 Principal을 만든다."""
    from jose import JWTError, jwt

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise _unauthorized(f"토큰이 유효하지 않습니다: {exc}") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("토큰에 sub(사용자 ID) 클레임이 없습니다")

    return Principal(
        user_id=user_id,
        roles=tuple(payload.get("roles") or ()),
        display_name=payload.get("name", ""),
        is_agent=bool(payload.get("is_agent", False)),
        on_behalf_of=payload.get("on_behalf_of"),
    )


async def get_current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Principal:
    """현재 요청의 주체를 반환한다. 액션 엔드포인트의 필수 의존성."""
    if credentials is None or not credentials.credentials:
        if AUTH_DEV_MODE:
            return DEV_PRINCIPAL
        raise _unauthorized("인증 토큰이 필요합니다")
    return principal_from_token(credentials.credentials)


async def get_optional_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[Principal]:
    """조회 전용 엔드포인트에서 쓰는 느슨한 버전 (없어도 통과)."""
    if credentials is None or not credentials.credentials:
        return DEV_PRINCIPAL if AUTH_DEV_MODE else None
    return principal_from_token(credentials.credentials)


def warn_if_dev_mode() -> None:
    if AUTH_DEV_MODE:
        logger.warning(
            "AUTH_DEV_MODE가 켜져 있습니다. 모든 요청이 고정 주체(%s)로 처리되며 "
            "감사 기록의 '누가'가 무의미해집니다. 운영에서는 반드시 끄십시오.",
            DEV_PRINCIPAL.user_id,
        )
