"""평가 스위트 실행기

타겟 종류
  sql               DuckDB 로 랜딩 Parquet 에 질의
  action            Action 을 **항상 dry_run 으로** 수행 (상태를 바꾸지 않는다)
  not_implemented   아직 답할 수 없는 케이스. 분모에는 들어간다

AIP Evals 가 온톨로지 편집 케이스를 시뮬레이션에서 실행하는 것과 같은 방식이다.
평가하느라 실제 상태가 바뀌면 그 평가는 한 번밖에 못 쓴다.
"""
from __future__ import annotations

import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .graders import run_grader
from .types import (
    CaseResult,
    GraderResult,
    Priority,
    Status,
    SuiteResult,
    TestCase,
)

__all__ = ["EvalRunner", "load_suite", "CANONICAL_TABLES"]

# 표준 테이블 이름. 사이트 데이터도 매핑을 거쳐 이 이름의 뷰로 올라온다.
CANONICAL_TABLES = [
    "rolls", "web_measurements", "roll_position_map", "electrode_lots",
    "cells", "supplies_cell", "cell_tests",
    "modules", "module_cells", "packs", "pack_modules",
]


def load_suite(path: pathlib.Path) -> Dict[str, Any]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "cases" not in data:
        raise ValueError(f"{path}: cases 가 없습니다")
    return data


class EvalRunner:
    """스위트 하나를 실행한다.

    DuckDB 연결과 Action 실행기를 지연 생성해, SQL 만 쓰는 스위트가
    액션 의존성을 끌어오지 않도록 한다.
    """

    def __init__(self, landing: pathlib.Path, action_executor=None):
        self.landing = pathlib.Path(landing)
        self._con = None
        self._executor = action_executor

    # --- 자원 ---

    @property
    def con(self):
        if self._con is None:
            import duckdb
            self._con = duckdb.connect()
            self._register_views(self._con)
        return self._con

    def _register_views(self, con) -> None:
        """랜딩 Parquet 를 표준 이름의 뷰로 올린다.

        없는 테이블은 건너뛴다. 그 테이블을 쓰는 케이스는 ERROR 로 남아
        "무엇이 없어서 못 하는지"가 리포트에 드러난다.
        """
        for t in CANONICAL_TABLES:
            pattern = str(self.landing / t / "**" / "*.parquet")
            try:
                con.execute(
                    f"CREATE OR REPLACE VIEW {t} AS "
                    f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)"
                )
            except Exception:
                continue

    @property
    def executor(self):
        if self._executor is None:
            import sys
            repo = pathlib.Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(repo / "api" / "src"))
            from common.actions import ActionExecutor, ActionRegistry, InMemoryAuditSink
            from services.action_handlers import register_handlers

            registry = register_handlers(ActionRegistry.from_schema_dir())
            self._executor = ActionExecutor(registry, InMemoryAuditSink())
        return self._executor

    # --- 실행 ---

    def run(self, suite: Dict[str, Any], only: Optional[List[str]] = None) -> SuiteResult:
        started = time.perf_counter()
        result = SuiteResult(
            suite=suite.get("suite", "unnamed"),
            version=str(suite.get("version", "0")),
            dataset=str(self.landing),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        for raw in suite["cases"]:
            case = TestCase.from_dict(raw)
            if only and case.id not in only:
                continue
            result.results.append(self._run_case(case))

        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    def _run_case(self, case: TestCase) -> CaseResult:
        t0 = time.perf_counter()
        kind = case.target.type

        if kind == "not_implemented":
            return CaseResult(
                case=case, status=Status.NOT_IMPLEMENTED,
                duration_ms=(time.perf_counter() - t0) * 1000,
                evidence=case.target.config.get("note") or case.note,
            )

        try:
            if kind == "sql":
                outcome = self._run_sql(case)
            elif kind == "action":
                outcome = self._run_action(case)
            else:
                raise ValueError(f"알 수 없는 타겟 종류: {kind}")
        except Exception as exc:
            return CaseResult(
                case=case, status=Status.ERROR,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        graders: List[GraderResult] = [
            run_grader(g.type, g.config, outcome) for g in case.graders
        ]
        # 채점기가 하나도 없으면 실행만으로 통과시키지 않는다.
        # 무엇을 검증하는지 적지 않은 케이스는 통과해도 의미가 없다.
        if not graders:
            graders = [GraderResult("no_grader", False, "채점기가 선언되지 않았습니다")]

        status = Status.PASSED if all(g.passed for g in graders) else Status.FAILED
        return CaseResult(
            case=case, status=status, graders=graders,
            duration_ms=(time.perf_counter() - t0) * 1000,
            rows=len(outcome.get("rows") or []),
            evidence=outcome.get("evidence"),
        )

    # --- 타겟별 실행 ---

    def _run_sql(self, case: TestCase) -> Dict[str, Any]:
        query = case.target.config["query"]
        rel = self.con.execute(query)
        columns = [d[0] for d in (rel.description or [])]
        rows = [dict(zip(columns, r)) for r in rel.fetchall()]

        evidence = None
        if rows:
            first = rows[0]
            evidence = ", ".join(f"{k}={first[k]}" for k in list(first)[:4])
        return {"rows": rows, "columns": columns, "evidence": evidence}

    def _run_action(self, case: TestCase) -> Dict[str, Any]:
        from common.actions import (
            ActionError,
            ActionRequest,
            Principal,
        )

        cfg = case.target.config
        principal = Principal(
            user_id=cfg.get("principal_id", "eval.runner"),
            roles=tuple(cfg.get("principal_roles") or ()),
        )
        request = ActionRequest(
            action_type=cfg["action_type"],
            parameters=cfg.get("parameters") or {},
            reason=cfg.get("reason"),
            dry_run=True,   # 평가는 절대 실제 상태를 바꾸지 않는다
        )

        try:
            res = self.executor.execute(request, principal)
        except ActionError as exc:
            # 거부도 정상적인 평가 결과다 (action_rejected 채점기가 받는다)
            return {"action": {"succeeded": False, "error": str(exc),
                               "dry_run": True, "changes": []},
                    "rows": [], "columns": [],
                    "evidence": f"거부: {exc}"}

        return {
            "action": {
                "succeeded": res.succeeded,
                "dry_run": res.dry_run,
                "changes": res.changes,
                "message": res.message,
                "error": None,
            },
            "rows": [], "columns": [],
            "evidence": res.message,
        }
