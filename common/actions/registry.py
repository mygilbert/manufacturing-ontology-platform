"""액션 타입 레지스트리

``ontology/schemas/actions/*.yaml``에 선언된 액션 타입을 읽어들이고,
각 타입에 핸들러(실제 수행 로직)를 연결한다.

정의(YAML)와 구현(핸들러)을 분리하는 이유는 온톨로지 스키마가
도메인 담당자가 읽고 고칠 수 있는 형태로 남아야 하기 때문이다.
핸들러가 없는 액션 타입은 "선언됐지만 아직 구현되지 않음" 상태로
목록에는 보이되 수행은 거부된다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .types import ActionTypeDef, ParameterDef, UnknownActionType

logger = logging.getLogger(__name__)

__all__ = ["ActionRegistry", "load_action_types", "DEFAULT_SCHEMA_DIR"]

# 리포지토리 루트 기준 액션 스키마 위치
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "ontology" / "schemas" / "actions"


def _parse_parameters(raw: Optional[dict]) -> Dict[str, ParameterDef]:
    params: Dict[str, ParameterDef] = {}
    for name, spec in (raw or {}).items():
        spec = spec or {}
        params[name] = ParameterDef(
            name=name,
            type=spec.get("type", "string"),
            required=bool(spec.get("required", False)),
            description=spec.get("description", ""),
            values=tuple(spec["values"]) if spec.get("values") else None,
            minimum=spec.get("minimum"),
            maximum=spec.get("maximum"),
            pattern=spec.get("pattern"),
            default=spec.get("default"),
        )
    return params


def _parse_action_type(data: dict) -> ActionTypeDef:
    permissions = data.get("permissions") or {}
    audit = data.get("audit") or {}
    transitions = data.get("transitions")
    return ActionTypeDef(
        action_type=data["actionType"],
        version=str(data.get("version", "1.0.0")),
        description=data.get("description", ""),
        applies_to=tuple(data.get("appliesTo") or ()),
        parameters=_parse_parameters(data.get("parameters")),
        allowed_roles=tuple(permissions.get("roles") or ()),
        reason_required=bool(audit.get("reasonRequired", False)),
        side_effects=tuple(data.get("sideEffects") or ()),
        transitions={k: tuple(v) for k, v in transitions.items()} if transitions else None,
        raw=data,
    )


def load_action_types(schema_dir: Optional[Path] = None) -> Dict[str, ActionTypeDef]:
    """YAML 디렉토리에서 액션 타입 정의를 로드한다."""
    import yaml  # 지연 임포트: 레지스트리를 쓰지 않는 경로에 의존성을 강요하지 않는다

    directory = Path(schema_dir or DEFAULT_SCHEMA_DIR)
    if not directory.is_dir():
        logger.warning("액션 스키마 디렉토리를 찾을 수 없습니다: %s", directory)
        return {}

    defs: Dict[str, ActionTypeDef] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("액션 스키마 파싱 실패 %s: %s", path.name, exc)
            continue
        if not data or "actionType" not in data:
            logger.warning("actionType이 없는 스키마를 건너뜁니다: %s", path.name)
            continue
        action_def = _parse_action_type(data)
        if action_def.action_type in defs:
            logger.warning("중복된 액션 타입: %s (%s)", action_def.action_type, path.name)
        defs[action_def.action_type] = action_def

    logger.info("액션 타입 %d개 로드 완료", len(defs))
    return defs


class ActionRegistry:
    """액션 타입 정의 + 핸들러 보관소"""

    def __init__(self, definitions: Optional[Dict[str, ActionTypeDef]] = None):
        self._definitions: Dict[str, ActionTypeDef] = dict(definitions or {})
        self._handlers: Dict[str, Callable] = {}

    # --- 정의 ---

    @classmethod
    def from_schema_dir(cls, schema_dir: Optional[Path] = None) -> "ActionRegistry":
        return cls(load_action_types(schema_dir))

    def add_definition(self, definition: ActionTypeDef) -> None:
        self._definitions[definition.action_type] = definition

    def get(self, action_type: str) -> ActionTypeDef:
        try:
            return self._definitions[action_type]
        except KeyError:
            raise UnknownActionType(f"등록되지 않은 액션 타입입니다: {action_type}") from None

    def list_definitions(self) -> List[ActionTypeDef]:
        return [self._definitions[k] for k in sorted(self._definitions)]

    def __contains__(self, action_type: str) -> bool:
        return action_type in self._definitions

    # --- 핸들러 ---

    def handler(self, action_type: str) -> Callable:
        """데코레이터로 핸들러를 등록한다.

        핸들러 시그니처::

            def handle(params: dict, principal: Principal, dry_run: bool) -> dict

        반환값은 ``{"changes": [...], "message": "..."}`` 형태.
        ``dry_run=True``이면 **어떤 쓰기도 하지 않고** 예상 변경만 돌려줘야 한다.
        """
        def decorator(func: Callable) -> Callable:
            self.register_handler(action_type, func)
            return func
        return decorator

    def register_handler(self, action_type: str, func: Callable) -> None:
        if action_type not in self._definitions:
            raise UnknownActionType(
                f"핸들러를 등록하려면 액션 타입 정의가 먼저 있어야 합니다: {action_type}"
            )
        self._handlers[action_type] = func

    def get_handler(self, action_type: str) -> Optional[Callable]:
        return self._handlers.get(action_type)

    def implemented(self) -> Iterable[str]:
        return self._handlers.keys()
