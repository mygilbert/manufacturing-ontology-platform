"""액션 실행기

수행 순서는 고정이다. 순서를 바꾸면 통제가 새는 곳이 생긴다.

    1. 액션 타입 조회       (없으면 UnknownActionType)
    2. 권한 확인            (없으면 PermissionDenied - 감사 기록 남김)
    3. 파라미터 검증        (실패 시 ValidationFailed - 감사 기록 남김)
    4. 사유(reason) 확인    (reasonRequired인 경우)
    5. 핸들러 수행          (dry_run이면 쓰기 없이 예상 변경만)
    6. 감사 기록            (성공/실패 무관하게 항상)

거부된 시도도 감사에 남긴다. "누가 무엇을 하려다 막혔는가"가
"누가 무엇을 했는가"만큼 중요하기 때문이다.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .audit import AuditRecord, AuditSink, InMemoryAuditSink, utcnow
from .registry import ActionRegistry
from .types import (
    ActionError,
    ActionRequest,
    ActionResult,
    ActionTypeDef,
    PermissionDenied,
    Principal,
    UnknownActionType,
    ValidationFailed,
)

logger = logging.getLogger(__name__)

__all__ = ["ActionExecutor"]


class ActionExecutor:
    """액션 수행의 단일 관문

    서비스 레이어가 그래프/DB에 직접 쓰는 대신 이 경로를 통과하게 하면
    권한·감사·시뮬레이션이 한 곳에서 보장된다.
    """

    def __init__(self, registry: ActionRegistry, audit_sink: Optional[AuditSink] = None):
        self.registry = registry
        self.audit_sink = audit_sink or InMemoryAuditSink()

    # --- 공개 API ---

    def execute(self, request: ActionRequest, principal: Principal) -> ActionResult:
        """액션을 수행한다.

        검증/권한 실패는 예외로 전파되며, 그 전에 감사 기록이 남는다.
        """
        definition = self.registry.get(request.action_type)

        try:
            self._authorize(definition, principal)
            params = self._validate(definition, request)
            handler = self._require_handler(definition)
        except ActionError as exc:
            self._audit(request, principal, succeeded=False, changes=[], error=str(exc))
            raise

        try:
            outcome = handler(params, principal, request.dry_run) or {}
        except Exception as exc:
            logger.exception("액션 수행 실패: %s", request.action_type)
            self._audit(request, principal, succeeded=False, changes=[], error=str(exc))
            raise

        changes = list(outcome.get("changes") or [])
        message = outcome.get("message", "")

        audit_id = self._audit(
            request, principal, succeeded=True, changes=changes, error=None
        )

        return ActionResult(
            action_type=request.action_type,
            succeeded=True,
            dry_run=request.dry_run,
            principal_id=principal.user_id,
            executed_at=utcnow(),
            changes=changes,
            message=message,
            audit_id=audit_id,
        )

    def simulate(self, request: ActionRequest, principal: Principal) -> ActionResult:
        """쓰기 없이 예상 결과만 확인한다 (dry_run 강제)."""
        simulated = ActionRequest(
            action_type=request.action_type,
            parameters=dict(request.parameters),
            reason=request.reason,
            dry_run=True,
            client_request_id=request.client_request_id,
        )
        return self.execute(simulated, principal)

    # --- 내부 단계 ---

    def _authorize(self, definition: ActionTypeDef, principal: Principal) -> None:
        if not definition.permits(principal):
            raise PermissionDenied(
                f"'{definition.action_type}' 수행 권한이 없습니다. "
                f"필요 역할: {', '.join(definition.allowed_roles)}"
            )

    def _validate(self, definition: ActionTypeDef, request: ActionRequest) -> Dict[str, Any]:
        errors: Dict[str, str] = {}
        validated: Dict[str, Any] = {}

        for name, param in definition.parameters.items():
            try:
                validated[name] = param.validate(request.parameters.get(name))
            except ValueError as exc:
                errors[name] = str(exc)

        unknown = set(request.parameters) - set(definition.parameters)
        for name in sorted(unknown):
            errors[name] = "정의되지 않은 파라미터입니다"

        if definition.reason_required and not (request.reason or "").strip():
            errors["reason"] = "이 액션은 수행 사유가 필요합니다"

        if errors:
            raise ValidationFailed(errors)

        return validated

    def _require_handler(self, definition: ActionTypeDef):
        handler = self.registry.get_handler(definition.action_type)
        if handler is None:
            raise UnknownActionType(
                f"'{definition.action_type}'은 선언됐으나 아직 구현되지 않았습니다"
            )
        return handler

    def _audit(
        self,
        request: ActionRequest,
        principal: Principal,
        *,
        succeeded: bool,
        changes: list,
        error: Optional[str],
    ) -> str:
        record = AuditRecord(
            audit_id=AuditRecord.new_id(),
            action_type=request.action_type,
            principal_id=principal.user_id,
            principal_roles=list(principal.roles),
            parameters=dict(request.parameters),
            reason=request.reason,
            dry_run=request.dry_run,
            succeeded=succeeded,
            changes=changes,
            error=error,
            occurred_at=utcnow(),
            on_behalf_of=principal.on_behalf_of,
            is_agent=principal.is_agent,
            client_request_id=request.client_request_id,
        )
        try:
            self.audit_sink.write(record)
        except Exception:
            # 감사 기록 실패가 액션 결과를 뒤집지는 않되, 반드시 로그로 남긴다.
            logger.exception("감사 기록 실패: %s (%s)", request.action_type, record.audit_id)
        return record.audit_id
