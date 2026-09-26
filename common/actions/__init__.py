"""Action 계층 (Kinetic Layer)

온톨로지의 상태를 바꾸는 유일한 경로.
자세한 설계 배경은 docs/08_액션_계층.md 참조.
"""
from .audit import AuditRecord, AuditSink, InMemoryAuditSink, PostgresAuditSink, utcnow
from .executor import ActionExecutor
from .registry import ActionRegistry, load_action_types
from .types import (
    ActionError,
    ActionRequest,
    ActionResult,
    ActionTypeDef,
    ParameterDef,
    PermissionDenied,
    Principal,
    UnknownActionType,
    ValidationFailed,
)

__all__ = [
    "ActionError",
    "ActionExecutor",
    "ActionRegistry",
    "ActionRequest",
    "ActionResult",
    "ActionTypeDef",
    "AuditRecord",
    "AuditSink",
    "InMemoryAuditSink",
    "ParameterDef",
    "PermissionDenied",
    "PostgresAuditSink",
    "Principal",
    "UnknownActionType",
    "ValidationFailed",
    "load_action_types",
    "utcnow",
]
