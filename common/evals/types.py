"""평가 프레임 기본 타입

역량질문(docs/05)이 실제로 답해지는지를 숫자로 재는 장치.

[설계 원칙]

1. **"답 못 함"과 "틀림"을 구분한다.**
   NOT_IMPLEMENTED 는 실패가 아니라 아직 구현되지 않은 것이다.
   둘을 뭉치면 진척을 볼 수 없다. 40개 중 0개에서 시작해
   n개로 올라가는 궤적이 이 프레임의 존재 이유다.

2. **온톨로지를 바꾸는 케이스는 시뮬레이션에서 실행한다.**
   평가하느라 실제 상태를 바꾸면 안 된다. Action 타겟은 항상
   dry_run=True 로 수행된다 (common/actions/executor.py).

3. **정의는 YAML, 실행은 코드.**
   테스트 케이스는 도메인 담당자가 읽고 고칠 수 있어야 한다.
   evals/suites/*.yaml 참조.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "Status",
    "Priority",
    "GraderSpec",
    "TargetSpec",
    "TestCase",
    "GraderResult",
    "CaseResult",
    "SuiteResult",
]


class Status(str, Enum):
    """케이스 실행 결과

    PASSED / FAILED 만으로는 진척을 읽을 수 없다.
    "아직 못 하는 것"과 "하는데 틀린 것"은 다른 문제이고 대응도 다르다.
    """
    PASSED = "PASSED"                    # 모든 채점 통과
    FAILED = "FAILED"                    # 실행됐으나 채점 불합격
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"  # 아직 답할 수 없음 (분모에는 포함)
    ERROR = "ERROR"                      # 실행 중 예외
    SKIPPED = "SKIPPED"                  # 조건 미충족으로 건너뜀

    @property
    def is_pass(self) -> bool:
        return self is Status.PASSED

    @property
    def counts_in_denominator(self) -> bool:
        """합격률 분모에 포함되는가.

        SKIPPED 만 제외한다. NOT_IMPLEMENTED 는 반드시 포함해야
        "아직 못 한다"가 점수에 드러난다.
        """
        return self is not Status.SKIPPED


class Priority(str, Enum):
    P0 = "P0"   # 없으면 프로젝트 의미 없음
    P1 = "P1"   # 중요
    P2 = "P2"   # 있으면 좋음


@dataclass(frozen=True)
class GraderSpec:
    """채점기 선언 (YAML 의 graders 항목 하나)"""
    type: str
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraderSpec":
        d = dict(d)
        t = d.pop("type")
        return cls(type=t, config=d)


@dataclass(frozen=True)
class TargetSpec:
    """평가 대상 선언

    type 별 필요한 설정
      sql              query
      action           action_type, parameters, reason, principal_roles
      not_implemented  note (왜 아직 못 하는지)
    """
    type: str
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetSpec":
        d = dict(d)
        t = d.pop("type")
        return cls(type=t, config=d)


@dataclass(frozen=True)
class TestCase:
    """테스트 케이스 하나 = 역량질문 하나"""

    # pytest 가 이름만 보고 수집하려 드는 것을 막는다 (도메인 용어를 지키기 위해)
    __test__ = False

    id: str                       # 역량질문 ID (A1, B2, ...)
    question: str                 # 사람이 읽는 질문
    target: TargetSpec
    graders: Sequence[GraderSpec] = ()
    priority: Priority = Priority.P1
    category: str = "misc"
    note: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestCase":
        return cls(
            id=d["id"],
            question=d["question"],
            target=TargetSpec.from_dict(d["target"]),
            graders=tuple(GraderSpec.from_dict(g) for g in (d.get("graders") or [])),
            priority=Priority(d.get("priority", "P1")),
            category=d.get("category", "misc"),
            note=d.get("note", ""),
        )


@dataclass
class GraderResult:
    grader: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: TestCase
    status: Status
    graders: List[GraderResult] = field(default_factory=list)
    duration_ms: float = 0.0
    rows: int = 0
    error: Optional[str] = None
    evidence: Optional[str] = None   # 첫 행 요약 등, 리포트에 남길 근거

    @property
    def failed_graders(self) -> List[GraderResult]:
        return [g for g in self.graders if not g.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "priority": self.case.priority.value,
            "category": self.case.category,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "rows": self.rows,
            "error": self.error,
            "evidence": self.evidence,
            "graders": [
                {"grader": g.grader, "passed": g.passed, "detail": g.detail}
                for g in self.graders
            ],
        }


@dataclass
class SuiteResult:
    suite: str
    version: str
    dataset: str
    results: List[CaseResult] = field(default_factory=list)
    started_at: str = ""
    duration_ms: float = 0.0

    # --- 집계 ---

    @property
    def scored(self) -> List[CaseResult]:
        return [r for r in self.results if r.status.counts_in_denominator]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status.is_pass)

    @property
    def total(self) -> int:
        return len(self.scored)

    @property
    def pass_rate(self) -> float:
        return (100.0 * self.passed / self.total) if self.total else 0.0

    def by_status(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.results:
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out

    def by_priority(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.scored:
            b = out.setdefault(r.case.priority.value, {"passed": 0, "total": 0})
            b["total"] += 1
            if r.status.is_pass:
                b["passed"] += 1
        return dict(sorted(out.items()))

    def by_category(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.scored:
            b = out.setdefault(r.case.category, {"passed": 0, "total": 0})
            b["total"] += 1
            if r.status.is_pass:
                b["passed"] += 1
        return dict(sorted(out.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "version": self.version,
            "dataset": self.dataset,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 1),
            "by_status": self.by_status(),
            "by_priority": self.by_priority(),
            "by_category": self.by_category(),
            "cases": [r.to_dict() for r in self.results],
        }
