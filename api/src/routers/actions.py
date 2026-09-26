"""Action API

온톨로지 상태를 바꾸는 유일한 공개 경로.

  GET  /api/actions                      액션 타입 목록
  GET  /api/actions/{action_type}        액션 타입 상세 (에이전트 도구 정의로도 사용)
  POST /api/actions/{action_type}        액션 수행
  POST /api/actions/{action_type}/simulate   쓰기 없이 예상 결과만
  GET  /api/actions/audit/recent         최근 감사 기록
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import get_current_principal, get_optional_principal
from common.actions import (
    ActionExecutor,
    ActionRegistry,
    ActionRequest,
    InMemoryAuditSink,
    PermissionDenied,
    Principal,
    UnknownActionType,
    ValidationFailed,
)
from services.action_handlers import register_handlers

logger = logging.getLogger(__name__)

router = APIRouter()


# --- 실행기 초기화 (모듈 단위 싱글톤) ---
# TODO: PostgresAuditSink 로 교체. 현재 InMemory 라 프로세스 재시작 시 기록이 사라진다.
_registry = register_handlers(ActionRegistry.from_schema_dir())
_audit_sink = InMemoryAuditSink()
_executor = ActionExecutor(_registry, _audit_sink)


def get_executor() -> ActionExecutor:
    return _executor


# --- 요청/응답 모델 ---

class ActionInvocation(BaseModel):
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = Field(
        None, description="수행 사유. reasonRequired 인 액션은 필수"
    )
    client_request_id: Optional[str] = None


# --- 엔드포인트 ---

@router.get("", summary="액션 타입 목록")
async def list_action_types(
    principal: Optional[Principal] = Depends(get_optional_principal),
    executor: ActionExecutor = Depends(get_executor),
) -> Dict[str, Any]:
    """정의된 액션 타입을 나열한다.

    인증된 주체가 있으면 수행 가능 여부(``permitted``)를 함께 표시한다.
    """
    implemented = set(executor.registry.implemented())
    items = []
    for definition in executor.registry.list_definitions():
        entry = definition.to_public_dict()
        entry["implemented"] = definition.action_type in implemented
        entry["permitted"] = definition.permits(principal) if principal else None
        items.append(entry)
    return {"total": len(items), "actions": items}


@router.get("/audit/recent", summary="최근 감사 기록")
async def recent_audit(
    limit: int = Query(50, ge=1, le=500),
    action_type: Optional[str] = None,
    principal: Principal = Depends(get_current_principal),
    executor: ActionExecutor = Depends(get_executor),
) -> Dict[str, Any]:
    """최근 액션 감사 기록을 조회한다 (거부된 시도 포함)."""
    sink = executor.audit_sink
    records = getattr(sink, "records", [])
    if action_type:
        records = [r for r in records if r.action_type == action_type]
    recent = list(reversed(records))[:limit]
    return {"total": len(recent), "records": [r.to_dict() for r in recent]}


@router.get("/{action_type}", summary="액션 타입 상세")
async def get_action_type(
    action_type: str,
    principal: Optional[Principal] = Depends(get_optional_principal),
    executor: ActionExecutor = Depends(get_executor),
) -> Dict[str, Any]:
    try:
        definition = executor.registry.get(action_type)
    except UnknownActionType as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    entry = definition.to_public_dict()
    entry["implemented"] = executor.registry.get_handler(action_type) is not None
    entry["permitted"] = definition.permits(principal) if principal else None
    return entry


def _run(
    action_type: str,
    body: ActionInvocation,
    principal: Principal,
    executor: ActionExecutor,
    dry_run: bool,
) -> Dict[str, Any]:
    request = ActionRequest(
        action_type=action_type,
        parameters=body.parameters,
        reason=body.reason,
        dry_run=dry_run,
        client_request_id=body.client_request_id,
    )
    try:
        result = executor.execute(request, principal)
    except UnknownActionType as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PermissionDenied as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValidationFailed as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "파라미터 검증 실패", "errors": exc.errors},
        ) from exc
    except Exception as exc:
        logger.exception("액션 수행 중 오류: %s", action_type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    return result.to_dict()


@router.post("/{action_type}/simulate", summary="액션 시뮬레이션 (쓰기 없음)")
async def simulate_action(
    action_type: str,
    body: ActionInvocation,
    principal: Principal = Depends(get_current_principal),
    executor: ActionExecutor = Depends(get_executor),
) -> Dict[str, Any]:
    """검증과 권한 확인을 모두 거치되 **아무것도 쓰지 않고** 예상 변경만 반환한다."""
    return _run(action_type, body, principal, executor, dry_run=True)


@router.post("/{action_type}", summary="액션 수행")
async def execute_action(
    action_type: str,
    body: ActionInvocation,
    principal: Principal = Depends(get_current_principal),
    executor: ActionExecutor = Depends(get_executor),
) -> Dict[str, Any]:
    """액션을 수행한다. 인증 필수이며 모든 시도가 감사에 기록된다."""
    return _run(action_type, body, principal, executor, dry_run=False)
