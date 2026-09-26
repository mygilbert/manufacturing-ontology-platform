"""채점기

타겟 실행 결과를 받아 합격/불합격을 판정한다.

결과는 두 형태 중 하나다.
  - SQL 타겟   : rows = [dict, ...], columns = [str, ...]
  - Action 타겟: action = {"succeeded": bool, "changes": [...], "message": str}

채점기는 작고 명시적으로 유지한다. 판정 근거가 리포트에 그대로 남아야
"왜 떨어졌는지"를 다시 실행하지 않고 알 수 있다.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .types import GraderResult

__all__ = ["run_grader", "GRADERS", "register"]

GRADERS: Dict[str, Callable[..., GraderResult]] = {}


def register(name: str):
    def deco(fn):
        GRADERS[name] = fn
        return fn
    return deco


def run_grader(spec_type: str, config: Dict[str, Any], outcome: Dict[str, Any]) -> GraderResult:
    fn = GRADERS.get(spec_type)
    if fn is None:
        return GraderResult(spec_type, False, f"알 수 없는 채점기: {spec_type}")
    try:
        return fn(config, outcome)
    except Exception as exc:  # 채점기 자체의 오류도 불합격으로 남긴다
        return GraderResult(spec_type, False, f"채점 중 오류: {exc}")


def _rows(outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
    return outcome.get("rows") or []


# ----------------------------------------------------------------------
# 행 기반 채점기 (SQL 타겟)
# ----------------------------------------------------------------------

@register("not_empty")
def _not_empty(cfg, outcome) -> GraderResult:
    n = len(_rows(outcome))
    return GraderResult("not_empty", n > 0, f"{n}행")


@register("no_rows")
def _no_rows(cfg, outcome) -> GraderResult:
    """결과가 없어야 통과. 데이터 품질 점검용.

    예: '전극 Lot 에 붙지 않은 측정값이 있는가' -> 없어야 정상.
    """
    n = len(_rows(outcome))
    return GraderResult("no_rows", n == 0, f"{n}행 (0이어야 통과)")


@register("row_count")
def _row_count(cfg, outcome) -> GraderResult:
    n = len(_rows(outcome))
    lo, hi = cfg.get("min"), cfg.get("max")
    ok = (lo is None or n >= lo) and (hi is None or n <= hi)
    bound = f"min={lo} max={hi}" if (lo is not None or hi is not None) else "제한 없음"
    return GraderResult("row_count", ok, f"{n}행 ({bound})")


@register("columns_present")
def _columns_present(cfg, outcome) -> GraderResult:
    want = set(cfg.get("columns") or [])
    have = set(outcome.get("columns") or [])
    missing = sorted(want - have)
    return GraderResult("columns_present", not missing,
                        "모두 존재" if not missing else f"누락: {', '.join(missing)}")


@register("top_row_equals")
def _top_row_equals(cfg, outcome) -> GraderResult:
    """1순위 행의 특정 컬럼이 기대값인지.

    역추적의 핵심 판정. "불량 셀에서 출발해 원인 구간을 1순위로 지목하는가"
    """
    rows = _rows(outcome)
    if not rows:
        return GraderResult("top_row_equals", False, "결과 없음")
    col, want = cfg["column"], cfg["value"]
    got = rows[0].get(col)
    return GraderResult("top_row_equals", str(got) == str(want),
                        f"{col}[0] = {got!r} (기대 {want!r})")


@register("value_between")
def _value_between(cfg, outcome) -> GraderResult:
    """첫 행의 수치 컬럼이 범위 안인지."""
    rows = _rows(outcome)
    if not rows:
        return GraderResult("value_between", False, "결과 없음")
    col = cfg["column"]
    raw = rows[0].get(col)
    if raw is None:
        return GraderResult("value_between", False, f"{col} 이 NULL")
    v = float(raw)
    lo, hi = cfg.get("min"), cfg.get("max")
    ok = (lo is None or v >= lo) and (hi is None or v <= hi)
    return GraderResult("value_between", ok, f"{col} = {v:g} (min={lo} max={hi})")


@register("all_rows_match")
def _all_rows_match(cfg, outcome) -> GraderResult:
    """모든 행의 컬럼이 허용값 집합 안에 있는지."""
    rows = _rows(outcome)
    if not rows:
        return GraderResult("all_rows_match", False, "결과 없음")
    col = cfg["column"]
    allowed = {str(v) for v in cfg["allowed"]}
    bad = sorted({str(r.get(col)) for r in rows} - allowed)
    return GraderResult("all_rows_match", not bad,
                        "모두 일치" if not bad else f"허용되지 않은 값: {', '.join(bad)}")


@register("column_values_include")
def _column_values_include(cfg, outcome) -> GraderResult:
    """결과 어딘가에 기대값들이 전부 나타나는지."""
    rows = _rows(outcome)
    col = cfg["column"]
    want = {str(v) for v in cfg["values"]}
    have = {str(r.get(col)) for r in rows}
    missing = sorted(want - have)
    return GraderResult("column_values_include", not missing,
                        "모두 포함" if not missing else f"누락: {', '.join(missing)}")


# ----------------------------------------------------------------------
# Action 타겟 채점기
# ----------------------------------------------------------------------

@register("action_succeeded")
def _action_succeeded(cfg, outcome) -> GraderResult:
    a = outcome.get("action") or {}
    ok = bool(a.get("succeeded"))
    return GraderResult("action_succeeded", ok, a.get("message", "") or ("성공" if ok else "실패"))


@register("action_rejected")
def _action_rejected(cfg, outcome) -> GraderResult:
    """거부돼야 통과. 권한/검증 통제가 실제로 막는지 확인한다."""
    a = outcome.get("action") or {}
    err = a.get("error") or ""
    want = cfg.get("error_contains")
    ok = not a.get("succeeded")
    if ok and want:
        ok = want in err
    return GraderResult("action_rejected", ok,
                        f"거부됨: {err}" if err else "거부되지 않음")


@register("changes_include")
def _changes_include(cfg, outcome) -> GraderResult:
    """변경 목록에 기대한 필드/값이 있는지."""
    a = outcome.get("action") or {}
    changes = a.get("changes") or []
    field = cfg["field"]
    want_after = cfg.get("after")
    for c in changes:
        if c.get("field") != field:
            continue
        if want_after is None or str(c.get("after")) == str(want_after):
            return GraderResult("changes_include", True,
                                f"{field} -> {c.get('after')!r}")
    fields = ", ".join(sorted({str(c.get("field")) for c in changes})) or "없음"
    return GraderResult("changes_include", False,
                        f"{field}={want_after!r} 없음 (있는 필드: {fields})")


@register("dry_run_wrote_nothing")
def _dry_run_wrote_nothing(cfg, outcome) -> GraderResult:
    """시뮬레이션이 실제로 쓰지 않았는지.

    Action 타겟은 항상 dry_run 으로 실행된다. 결과에 dry_run 표시가
    남아 있어야 평가가 상태를 오염시키지 않았음을 보장할 수 있다.
    """
    a = outcome.get("action") or {}
    ok = a.get("dry_run") is True
    return GraderResult("dry_run_wrote_nothing", ok,
                        "시뮬레이션으로 실행됨" if ok else "dry_run 표시 없음")
