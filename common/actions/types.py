"""Action 계층 기본 타입

팔란티어 Foundry 온톨로지의 Kinetic Layer에 해당하는 구조.

핵심 원칙: **상태 변경은 오직 Action Type을 통해서만 일어난다.**
서비스가 그래프/DB에 직접 쓰면 누가 무엇을 왜 바꿨는지 알 수 없고,
권한도 감사도 시뮬레이션도 불가능해진다.

구성 요소
  - ActionTypeDef : 액션의 계약 (파라미터, 권한, 검증 규칙, 감사 요건)
  - Principal     : 액션을 수행하는 주체 (누가)
  - ActionRequest : 수행 요청 (무엇을, 왜)
  - ActionResult  : 수행 결과 (무엇이 바뀌었나)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ActionError",
    "ValidationFailed",
    "PermissionDenied",
    "UnknownActionType",
    "ParameterDef",
    "ActionTypeDef",
    "Principal",
    "ActionRequest",
    "ActionResult",
]


class ActionError(Exception):
    """Action 수행 중 발생한 오류의 기반 클래스."""


class ValidationFailed(ActionError):
    """파라미터 검증 실패.

    어떤 필드가 왜 실패했는지를 모두 모아서 전달한다.
    하나씩 알려주면 호출자가 여러 번 왕복해야 한다.
    """

    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(f"파라미터 검증 실패 - {detail}")


class PermissionDenied(ActionError):
    """수행 권한 없음."""


class UnknownActionType(ActionError):
    """등록되지 않은 액션 타입."""


_SCALAR_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "datetime": (datetime, date, str),
}


@dataclass(frozen=True)
class ParameterDef:
    """액션 파라미터 정의"""
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    values: Optional[Sequence[str]] = None      # enum 허용값
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None
    default: Any = None

    def validate(self, value: Any) -> Any:
        """값을 검증하고 정규화해 반환한다.

        검증 실패 시 사람이 읽을 수 있는 사유 문자열을 담아
        :class:`ValueError`를 던진다. 호출자(Executor)가 이를 모아
        :class:`ValidationFailed`로 만든다.
        """
        if value is None:
            if self.required:
                raise ValueError("필수 항목입니다")
            return self.default

        expected = _SCALAR_TYPES.get(self.type)
        if expected is None:
            raise ValueError(f"알 수 없는 파라미터 타입입니다: {self.type}")

        # bool은 int의 하위 타입이므로 숫자 검사보다 먼저 걸러야 한다.
        if self.type != "boolean" and isinstance(value, bool):
            raise ValueError(f"{self.type} 타입이어야 합니다")
        if not isinstance(value, expected):
            raise ValueError(f"{self.type} 타입이어야 합니다 (받은 값: {type(value).__name__})")

        if self.values is not None and value not in self.values:
            raise ValueError(f"허용되지 않은 값입니다. 가능: {', '.join(map(str, self.values))}")

        if self.type in ("integer", "number"):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{self.minimum} 이상이어야 합니다")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{self.maximum} 이하여야 합니다")

        if self.pattern is not None and isinstance(value, str):
            if not re.match(self.pattern, value):
                raise ValueError(f"형식이 올바르지 않습니다 (기대: {self.pattern})")

        return value


@dataclass(frozen=True)
class ActionTypeDef:
    """액션 타입 정의 (ontology/schemas/actions/*.yaml 에서 로드)"""
    action_type: str
    version: str = "1.0.0"
    description: str = ""
    applies_to: Sequence[str] = field(default_factory=tuple)
    parameters: Dict[str, ParameterDef] = field(default_factory=dict)
    allowed_roles: Sequence[str] = field(default_factory=tuple)
    reason_required: bool = False
    side_effects: Sequence[str] = field(default_factory=tuple)
    transitions: Optional[Dict[str, Sequence[str]]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def permits(self, principal: "Principal") -> bool:
        """이 주체가 액션을 수행할 수 있는지.

        allowed_roles가 비어 있으면 인증된 주체 전원에게 허용한다.
        """
        if not self.allowed_roles:
            return True
        return bool(set(self.allowed_roles) & set(principal.roles))

    def to_public_dict(self) -> Dict[str, Any]:
        """API로 노출할 형태 (에이전트의 도구 정의로도 쓰인다)."""
        return {
            "action_type": self.action_type,
            "version": self.version,
            "description": self.description,
            "applies_to": list(self.applies_to),
            "reason_required": self.reason_required,
            "allowed_roles": list(self.allowed_roles),
            "side_effects": list(self.side_effects),
            "parameters": {
                name: {
                    "type": p.type,
                    "required": p.required,
                    "description": p.description,
                    **({"values": list(p.values)} if p.values else {}),
                }
                for name, p in self.parameters.items()
            },
        }


@dataclass(frozen=True)
class Principal:
    """액션을 수행하는 주체

    감사 기록의 "누가"에 해당한다. 인증이 없으면 이 값을 만들 수 없고,
    따라서 액션도 수행할 수 없다. 그것이 의도된 동작이다.
    """
    user_id: str
    roles: Sequence[str] = field(default_factory=tuple)
    display_name: str = ""
    # 에이전트가 사람을 대신해 수행한 경우 사람의 승인 여부를 기록한다.
    on_behalf_of: Optional[str] = None
    is_agent: bool = False

    def __post_init__(self):
        if not self.user_id:
            raise ValueError("Principal.user_id는 비어 있을 수 없습니다")


@dataclass(frozen=True)
class ActionRequest:
    """액션 수행 요청"""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    dry_run: bool = False
    client_request_id: Optional[str] = None


@dataclass
class ActionResult:
    """액션 수행 결과"""
    action_type: str
    succeeded: bool
    dry_run: bool
    principal_id: str
    executed_at: datetime
    changes: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""
    audit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "succeeded": self.succeeded,
            "dry_run": self.dry_run,
            "principal_id": self.principal_id,
            "executed_at": self.executed_at.isoformat(),
            "changes": self.changes,
            "message": self.message,
            "audit_id": self.audit_id,
        }
