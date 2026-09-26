#!/usr/bin/env python3
"""반출 경계 검사 — 사내 고유 정보가 커밋에 섞이지 않게 막는다

익명화는 커밋할 때마다 판단이 필요하고, 수백 번 판단하면 몇 번은 틀린다.
이 검사는 그 판단을 기계에 맡긴다.

검사 항목
  1. config/site/ 의 실제 파일이 스테이징됐는가 (예시/README 제외)
  2. 사이트 고유 패턴이 코어 파일에 들어갔는가
     패턴 목록은 config/site/leak_patterns.txt 에 둔다 (이 파일 자체도 추적 제외)
  3. 자격증명/접속정보로 보이는 문자열
  4. 데이터 파일(.parquet/.csv/.xlsx)이 스테이징됐는가

사용
  python scripts/check_leaks.py --staged     # pre-commit 훅이 부르는 형태
  python scripts/check_leaks.py --all        # 작업 트리 전체 점검

훅 설치
  git config core.hooksPath .githooks
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PATTERN_FILE = REPO / "config" / "site" / "leak_patterns.txt"

# config/site/ 에서 커밋이 허용되는 것
SITE_ALLOWED = re.compile(
    r"^config/site/(README\.md|(?:[^/]+/)?[^/]*\.example\.(?:yaml|yml|txt))$"
)

# 자격증명으로 보이는 것 (예시/플레이스홀더는 제외)
CREDENTIAL_PATTERNS = [
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*[\"']([^\"'{<$][^\"']{3,})[\"']", "비밀번호"),
    (r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*[\"']([^\"'{<$][^\"']{8,})[\"']", "API 키"),
    (r"(?i)jdbc:[a-z]+://[^\s\"'<]+", "JDBC 접속 문자열"),
    (r"(?i)(oracle|sqlserver|mysql|postgres)://[^\s\"'<]*:[^\s\"'<]*@", "자격증명 포함 DSN"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b", "IP:포트"),
]

# 이 값들은 예시/합성/템플릿이므로 자격증명 검사에서 제외한다.
# 특히 중괄호는 f-string/환경변수 치환 자리표시자를 뜻하므로 실제 값이 아니다.
#   f"jdbc:postgresql://{self.host}:{self.port}/{self.database}"  <- 템플릿, 문제 없음
CREDENTIAL_ALLOWLIST = re.compile(
    r"(?i)(\{|\$\{|<[^>]+>|your-super-secret|change-in-production|example|"
    r"localhost|127\.0\.0\.1|0\.0\.0\.0|ontology123|timescale123|redis123|dev\.user)"
)

DATA_SUFFIXES = {".parquet", ".csv", ".xlsx", ".xls", ".dat"}
# 빈 양식은 코어 자산이므로 데이터 파일 검사에서 제외한다.
# (채워진 파일은 config/site/knowledge/ 로 간다 — docs/09_배포_경계.md)
DATA_EXEMPT = re.compile(r"(^|/)templates?/|\.example\.[a-z]+$|(^|/)fixtures?/")
SKIP_DIRS = {".git", "node_modules", "__pycache__", "landing", "profiles_out",
             "venv", ".venv", "dist"}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".yaml", ".yml", ".json", ".md",
                 ".sql", ".sh", ".txt", ".html", ".toml", ".ini", ".cfg"}


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return [f for f in out.stdout.splitlines() if f.strip()]


def all_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=False)
    return [f for f in out.stdout.splitlines() if f.strip()]


def load_site_patterns() -> list[tuple[re.Pattern, str]]:
    """사이트 고유 패턴 목록.

    이 파일 자체가 사내 정보(설비 ID 접두어, 테이블명 등)이므로
    config/site/ 안에 두어 추적에서 제외한다.
    """
    if not PATTERN_FILE.exists():
        return []
    pats = []
    for i, raw in enumerate(PATTERN_FILE.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            pats.append((re.compile(line, re.I), f"{PATTERN_FILE.name}:{i}"))
        except re.error as exc:
            print(f"  경고: 잘못된 정규식 {PATTERN_FILE.name}:{i} — {exc}", file=sys.stderr)
    return pats


def check(files: list[str]) -> list[str]:
    problems: list[str] = []
    site_pats = load_site_patterns()

    for f in files:
        p = pathlib.Path(f)
        if any(part in SKIP_DIRS for part in p.parts):
            continue

        # 1. config/site/ 실제 파일
        if f.startswith("config/site/") and not SITE_ALLOWED.match(f):
            problems.append(
                f"{f}\n    사이트 오버레이 파일은 커밋할 수 없습니다. "
                f"예시는 *.example.yaml 로 만드십시오."
            )
            continue

        # 2. 데이터 파일
        if p.suffix.lower() in DATA_SUFFIXES and not DATA_EXEMPT.search(f):
            problems.append(
                f"{f}\n    데이터 파일은 커밋하지 않습니다. "
                f"config/site/ 또는 외부 저장소로 옮기십시오."
            )
            continue

        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue

        full = REPO / f
        if not full.exists():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # 3. 사이트 고유 패턴
        for pat, origin in site_pats:
            m = pat.search(text)
            if m:
                line_no = text[:m.start()].count("\n") + 1
                problems.append(
                    f"{f}:{line_no}\n    사이트 고유 패턴이 발견됐습니다 ({origin}). "
                    f"코어에서 제거하고 config/site/ 로 옮기십시오."
                )
                break

        # 4. 자격증명
        for pat, label in CREDENTIAL_PATTERNS:
            for m in re.finditer(pat, text):
                if CREDENTIAL_ALLOWLIST.search(m.group(0)):
                    continue
                line_no = text[:m.start()].count("\n") + 1
                problems.append(f"{f}:{line_no}\n    {label}로 보이는 값이 있습니다.")
                break

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true", help="스테이징된 파일만 (기본)")
    g.add_argument("--all", action="store_true", help="추적 중인 전체 파일")
    args = ap.parse_args()

    audit = args.all
    files = all_files() if audit else staged_files()
    if not files:
        return 0

    problems = check(files)

    if not problems:
        print(f"반출 경계 검사 통과 ({len(files)}개 파일)")
        return 0

    if audit:
        # 감사 모드: 이미 추적 중인 파일까지 훑는다. 보고만 하고 실패시키지 않는다.
        print(f"반출 경계 감사 — {len(problems)}건 (추적 파일 {len(files)}개)\n")
        for p in problems:
            print(f"  {p}\n")
        print("커밋을 막는 것은 --staged 검사(pre-commit 훅)입니다.")
        return 0

    print("\n반출 경계 검사 실패\n", file=sys.stderr)
    for p in problems:
        print(f"  {p}\n", file=sys.stderr)
    print("의도한 커밋이라면 --no-verify 로 우회할 수 있으나, "
          "그 전에 무엇을 내보내는지 확인하십시오.\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
