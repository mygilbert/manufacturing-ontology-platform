#!/usr/bin/env python3
"""평가 스위트 실행 — 역량질문이 실제로 답해지는지 숫자로 잰다

docs/05_역량질문.md 가 원천이고, docs/11_AIP_구성_분석.md 의 6요소 중 5번이다.

[읽는 법]
  PASS     지금 답한다
  FAIL     답하는데 기준에 못 미친다
  미구현    아직 답할 수 없다 — 실패가 아니라 남은 과제 (분모에는 포함)

둘을 뭉치면 진척을 볼 수 없다. 합격 수가 올라가는 궤적이 목적이다.

[온톨로지 편집]
  Action 타겟은 항상 dry_run 으로 실행된다. 평가가 실제 상태를 바꾸지 않는다.
  AIP Evals 가 온톨로지 편집을 시뮬레이션에서 실행하는 것과 같은 방식이다.

사용법
  pip install duckdb pyarrow pyyaml
  python scripts/generate_battery_genealogy.py     # 합성 데이터 (없으면)
  python scripts/run_evals.py                      # 전체 실행
  python scripts/run_evals.py --case B2 --case B3  # 일부만
  python scripts/run_evals.py --landing /site/data # 실데이터로
  python scripts/run_evals.py --fail-under 60      # CI 용 임계값
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.evals import EvalRunner, load_suite, print_console, write_reports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="evals/suites/competency_battery.yaml")
    ap.add_argument("--landing", default="landing", help="데이터 랜딩 폴더")
    ap.add_argument("--out", default="evals/results", help="리포트 산출 폴더")
    ap.add_argument("--case", action="append", dest="cases",
                    help="특정 케이스만 실행 (반복 지정 가능)")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="합격률이 이 값 미만이면 비정상 종료 (CI 용)")
    ap.add_argument("--quiet", action="store_true", help="콘솔 출력 생략")
    args = ap.parse_args()

    suite_path = pathlib.Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = REPO / suite_path
    if not suite_path.exists():
        print(f"스위트를 찾을 수 없습니다: {suite_path}", file=sys.stderr)
        return 2

    landing = pathlib.Path(args.landing)
    if not landing.is_absolute():
        landing = REPO / landing
    if not landing.is_dir():
        print(f"데이터 폴더가 없습니다: {landing}\n"
              f"  먼저 실행하십시오: python scripts/generate_battery_genealogy.py",
              file=sys.stderr)
        return 2

    suite = load_suite(suite_path)
    runner = EvalRunner(landing)
    result = runner.run(suite, only=args.cases)

    if not args.quiet:
        print_console(result)

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    path = write_reports(result, out_dir)
    # 리포 밖 경로도 허용한다 (임시 폴더로 뽑아보는 경우가 있다)
    try:
        shown = path.relative_to(REPO)
    except ValueError:
        shown = path
    print(f"\n리포트: {shown}")

    if args.fail_under is not None and result.pass_rate < args.fail_under:
        print(f"\n합격률 {result.pass_rate:.1f}% < 임계값 {args.fail_under}%",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
