"""Cypher 쿼리 안전 조립 유틸리티

Apache AGE의 ``cypher()`` 함수는 쿼리 본문을 달러 인용 문자열로 받기 때문에
일반적인 SQL 바인드 파라미터(``$1``)를 쿼리 본문 안에서 쓸 수 없다.
그래서 값을 문자열로 조립할 수밖에 없는데, 이때 이스케이프를 하지 않으면
두 가지 문제가 동시에 발생한다.

1. **기능 버그**: Lot ID나 알람 메시지에 작은따옴표가 하나만 있어도 쿼리가 깨진다.
   합성 데이터에서는 드러나지 않고 실제 데이터에서 바로 터진다.
2. **인젝션**: 외부 입력으로 그래프 질의를 조작할 수 있다.

이 모듈은 값(리터럴)과 식별자(레이블/속성명/관계타입)를 구분해서 처리한다.

- 값은 이스케이프 후 인용한다  -> :func:`cy_str`, :func:`cy_num`, :func:`cy_value`
- 식별자는 이스케이프가 불가능하므로 허용 문자만 통과시킨다 -> :func:`cy_ident`,
  :func:`cy_label`, :func:`cy_rel_type`

사용 예::

    query = f'''
    SELECT * FROM cypher('manufacturing', $$
        MATCH (e:{cy_label("Equipment")} {{equipment_id: {cy_str(equipment_id)}}})
        RETURN e
    $$) as (equipment agtype);
    '''
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

__all__ = [
    "CypherSafetyError",
    "cy_str",
    "cy_num",
    "cy_int",
    "cy_bool",
    "cy_value",
    "cy_ident",
    "cy_label",
    "cy_rel_type",
    "cy_rel_types",
    "cy_props",
]


class CypherSafetyError(ValueError):
    """Cypher 조각으로 쓸 수 없는 값이 들어온 경우."""


# 식별자(레이블, 속성명, 관계 타입) 허용 패턴.
# AGE/Cypher에서 백틱 인용 없이 안전하게 쓸 수 있는 형태로만 제한한다.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# 리터럴에서 제거 대상인 제어 문자 (탭/개행 제외).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 문자열 리터럴 이스케이프 대상.
# 역슬래시를 먼저 치환해야 이중 이스케이프가 되지 않는다.
_ESCAPES = (
    ("\\", "\\\\"),
    ("'", "\\'"),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
)

# 달러 인용($$)으로 감싼 쿼리 본문을 조기 종료시킬 수 있는 시퀀스.
_DOLLAR_QUOTE = "$$"

MAX_STR_LEN = 4096


def cy_str(value: Any, *, max_len: int = MAX_STR_LEN) -> str:
    """값을 Cypher 문자열 리터럴로 변환한다 (따옴표 포함).

    ``None``은 Cypher의 ``null``로 변환된다.

    >>> cy_str("LOT-001")
    "'LOT-001'"
    >>> cy_str("O'Brien")
    "'O\\\\'Brien'"
    """
    if value is None:
        return "null"

    if isinstance(value, (datetime, date)):
        text = value.isoformat()
    elif isinstance(value, bool):
        # bool은 int의 하위 타입이므로 숫자보다 먼저 확인해야 한다.
        return cy_bool(value)
    elif isinstance(value, (int, float)):
        text = str(value)
    else:
        text = str(value)

    if len(text) > max_len:
        raise CypherSafetyError(
            f"문자열이 최대 길이 {max_len}자를 초과했습니다 (실제 {len(text)}자)"
        )

    text = _CONTROL_RE.sub("", text)

    if _DOLLAR_QUOTE in text:
        # $$ 는 AGE 쿼리 본문의 종료 구분자다. 이스케이프할 방법이 없으므로 거부한다.
        raise CypherSafetyError("문자열에 '$$' 시퀀스를 포함할 수 없습니다")

    for target, replacement in _ESCAPES:
        text = text.replace(target, replacement)

    return f"'{text}'"


def cy_num(value: Any) -> str:
    """값을 Cypher 숫자 리터럴로 변환한다."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        raise CypherSafetyError("bool은 숫자로 쓸 수 없습니다. cy_bool을 사용하세요")
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise CypherSafetyError(f"숫자로 변환할 수 없습니다: {value!r}") from exc
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise CypherSafetyError("NaN/Inf는 Cypher 리터럴로 쓸 수 없습니다")
    return repr(value)


def cy_int(value: Any, *, minimum: int = 0, maximum: int = 100_000) -> str:
    """LIMIT/SKIP/가변 길이 경로 깊이처럼 정수만 허용되는 자리에 사용한다.

    범위를 벗어나면 예외 대신 경계값으로 자른다. 페이지네이션 파라미터에
    이상한 값이 들어왔다고 조회 자체를 실패시키는 것보다 안전한 기본값으로
    동작하는 편이 낫기 때문이다.
    """
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CypherSafetyError(f"정수로 변환할 수 없습니다: {value!r}") from exc
    return str(max(minimum, min(maximum, number)))


def cy_bool(value: Any) -> str:
    """값을 Cypher 불리언 리터럴로 변환한다."""
    return "true" if bool(value) else "false"


def cy_value(value: Any) -> str:
    """타입에 따라 적절한 Cypher 리터럴로 변환한다."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return cy_bool(value)
    if isinstance(value, (int, float)):
        return cy_num(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cy_value(item) for item in value) + "]"
    return cy_str(value)


def cy_ident(name: Any) -> str:
    """속성명 등 식별자를 검증한다.

    식별자는 이스케이프로 무해하게 만들 수 없으므로 허용 문자만 통과시킨다.
    """
    text = str(name)
    if not _IDENT_RE.match(text):
        raise CypherSafetyError(
            f"식별자로 사용할 수 없습니다: {text!r} "
            "(영문자/밑줄로 시작하고 영숫자/밑줄만 허용, 최대 64자)"
        )
    return text


def cy_label(label: Any) -> str:
    """노드 레이블을 검증한다 (Equipment, Lot 등)."""
    return cy_ident(label)


def cy_rel_type(relation: Any) -> str:
    """관계 타입을 검증한다 (PROCESSED_AT 등)."""
    return cy_ident(relation)


def cy_rel_types(relations: Optional[Iterable[Any]]) -> str:
    """관계 타입 목록을 ``TYPE_A|TYPE_B`` 형태로 조립한다.

    비어 있으면 빈 문자열을 반환하므로 호출부에서 필터 유무를 판단할 수 있다.
    """
    if not relations:
        return ""
    return "|".join(cy_rel_type(relation) for relation in relations)


def cy_props(properties: Optional[Dict[str, Any]], *, skip_none: bool = True) -> str:
    """딕셔너리를 Cypher 속성 맵 ``{k: v, ...}`` 으로 조립한다.

    키는 식별자로 검증하고 값은 타입별 리터럴로 변환한다.
    빈 딕셔너리이거나 남는 항목이 없으면 빈 문자열을 반환한다.
    """
    if not properties:
        return ""

    parts = []
    for key, value in properties.items():
        if skip_none and value is None:
            continue
        parts.append(f"{cy_ident(key)}: {cy_value(value)}")

    if not parts:
        return ""
    return "{" + ", ".join(parts) + "}"
