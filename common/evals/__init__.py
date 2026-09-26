"""평가 프레임 — 역량질문이 실제로 답해지는지 숫자로 잰다

docs/05_역량질문.md 가 테스트 케이스의 원천이고,
docs/11_AIP_구성_분석.md 의 6요소 중 5번에 해당한다.
"""
from .graders import GRADERS, register, run_grader
from .report import print_console, write_reports
from .runner import CANONICAL_TABLES, EvalRunner, load_suite
from .types import (
    CaseResult,
    GraderResult,
    GraderSpec,
    Priority,
    Status,
    SuiteResult,
    TargetSpec,
    TestCase,
)

__all__ = [
    "CANONICAL_TABLES", "CaseResult", "EvalRunner", "GRADERS", "GraderResult",
    "GraderSpec", "Priority", "Status", "SuiteResult", "TargetSpec", "TestCase",
    "load_suite", "print_console", "register", "run_grader", "write_reports",
]
