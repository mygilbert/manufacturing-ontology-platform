"""평가 결과 리포트

콘솔 요약과 Markdown/JSON 산출물.

리포트는 두 질문에 답해야 한다.
  1. 지금 몇 개에 답하는가 (n/40)
  2. 못 하는 것은 아직 못 하는 것인가, 하는데 틀린 것인가
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from .types import Status, SuiteResult

__all__ = ["print_console", "write_reports"]

MARK = {
    Status.PASSED: "PASS",
    Status.FAILED: "FAIL",
    Status.NOT_IMPLEMENTED: "미구현",
    Status.ERROR: "오류",
    Status.SKIPPED: "건너뜀",
}


def _bar(passed: int, total: int, width: int = 24) -> str:
    if not total:
        return "-" * width
    filled = round(width * passed / total)
    return "#" * filled + "." * (width - filled)


def print_console(res: SuiteResult) -> None:
    print()
    print("=" * 74)
    print(f"{res.suite} v{res.version}   데이터: {res.dataset}")
    print("=" * 74)

    for r in res.results:
        mark = MARK[r.status]
        line = f"  {mark:<6} {r.case.id:<5} {r.case.priority.value}  {r.case.question[:44]}"
        print(line)
        if r.status is Status.FAILED:
            for g in r.failed_graders:
                print(f"         └ {g.grader}: {g.detail}")
        elif r.status is Status.ERROR:
            print(f"         └ {r.error}")
        elif r.status is Status.NOT_IMPLEMENTED and r.evidence:
            print(f"         └ {r.evidence}")

    print()
    print("-" * 74)
    print(f"  합격률  {res.passed}/{res.total}  ({res.pass_rate:.1f}%)  "
          f"[{_bar(res.passed, res.total)}]")

    counts = res.by_status()
    detail = "  ".join(f"{MARK[Status(k)]} {v}" for k, v in sorted(counts.items()))
    print(f"  상태    {detail}")

    print()
    print("  우선순위별")
    for p, b in res.by_priority().items():
        rate = 100 * b["passed"] / b["total"] if b["total"] else 0
        print(f"    {p}  {b['passed']:>2}/{b['total']:<3} ({rate:5.1f}%)  "
              f"[{_bar(b['passed'], b['total'], 18)}]")

    print()
    print("  분류별")
    for c, b in res.by_category().items():
        rate = 100 * b["passed"] / b["total"] if b["total"] else 0
        print(f"    {c:<16} {b['passed']:>2}/{b['total']:<3} ({rate:5.1f}%)")
    print("-" * 74)


def write_reports(res: SuiteResult, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{res.suite}.json").write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    L.append(f"# 평가 결과 — {res.suite}\n")
    L.append(f"- 스위트 버전 `{res.version}`")
    L.append(f"- 데이터 `{res.dataset}`")
    L.append(f"- 실행 {res.started_at} ({res.duration_ms:.0f} ms)")
    L.append("")
    L.append(f"## 합격률 {res.passed}/{res.total} ({res.pass_rate:.1f}%)\n")

    counts = res.by_status()
    L.append("| 상태 | 개수 | 의미 |")
    L.append("|---|---|---|")
    meaning = {
        "PASSED": "답한다",
        "FAILED": "답하는데 기준 미달",
        "NOT_IMPLEMENTED": "아직 답할 수 없다",
        "ERROR": "실행 중 오류",
        "SKIPPED": "건너뜀 (분모 제외)",
    }
    for k in ["PASSED", "FAILED", "NOT_IMPLEMENTED", "ERROR", "SKIPPED"]:
        if k in counts:
            L.append(f"| {MARK[Status(k)]} | {counts[k]} | {meaning[k]} |")
    L.append("")

    L.append("### 우선순위별\n")
    L.append("| 우선순위 | 합격 | 전체 | 비율 |")
    L.append("|---|---|---|---|")
    for p, b in res.by_priority().items():
        rate = 100 * b["passed"] / b["total"] if b["total"] else 0
        L.append(f"| {p} | {b['passed']} | {b['total']} | {rate:.1f}% |")
    L.append("")

    L.append("### 분류별\n")
    L.append("| 분류 | 합격 | 전체 | 비율 |")
    L.append("|---|---|---|---|")
    for c, b in res.by_category().items():
        rate = 100 * b["passed"] / b["total"] if b["total"] else 0
        L.append(f"| {c} | {b['passed']} | {b['total']} | {rate:.1f}% |")
    L.append("")

    L.append("## 케이스별\n")
    L.append("| ID | 상태 | 우선 | 역량질문 | 근거 / 사유 |")
    L.append("|---|---|---|---|---|")
    for r in res.results:
        note = ""
        if r.status is Status.FAILED:
            note = "; ".join(f"{g.grader}: {g.detail}" for g in r.failed_graders)
        elif r.status is Status.ERROR:
            note = (r.error or "")[:110]
        elif r.evidence:
            note = r.evidence[:110]
        note = note.replace("|", "\\|")
        q = r.case.question.replace("|", "\\|")
        L.append(f"| `{r.case.id}` | {MARK[r.status]} | {r.case.priority.value} "
                 f"| {q} | {note} |")
    L.append("")
    L.append("> 온톨로지를 바꾸는 케이스는 모두 시뮬레이션(dry_run)으로 실행됐다. "
             "평가가 실제 상태를 바꾸지 않는다.")

    path = out_dir / f"{res.suite}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
