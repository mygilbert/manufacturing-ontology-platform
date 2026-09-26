#!/usr/bin/env python3
"""사내 데이터 프로파일러 — 비식별 요약만 산출한다

실데이터를 밖으로 내보내지 않고도 온톨로지 매핑을 설계할 수 있게 한다.
반출 승인을 받기도 훨씬 수월하다.

**산출물에 실제 값은 들어가지 않는다.** 기본값 기준으로 나가는 것은
컬럼 이름, 타입, 널 비율, 고유값 개수, 시각 범위, 수치 요약통계뿐이다.
(`--sample-values N` 으로 명시할 때만 값 예시가 포함되며, 마스킹 대상 컬럼은
그 경우에도 제외된다.)

두 가지 모드가 함께 돈다.

  1. 탐색     랜딩 폴더를 훑어 어떤 테이블에 어떤 컬럼이 있는지 정리
  2. 준비도   온톨로지 계보를 만들 수 있는 상태인지 진단
              - 롤 위치(position_m)가 있는가
              - 전극 Lot 에 롤 구간이 기록되는가       <- 가장 중요
              - 셀 하나에 양극/음극 Lot 이 둘 다 붙는가
              - 시각 컬럼의 시간대/정밀도는 어떤가

사용법
  pip install duckdb pyarrow
  python scripts/profile_site_data.py --landing /path/to/landing
  python scripts/profile_site_data.py --landing /data --mapping config/site/mappings/battery.yaml

산출물
  profiles_out/site_profile.md    사람이 읽는 보고서 (검토 후 공유 가능)
  profiles_out/site_profile.json  기계가 읽는 요약
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime

try:
    import duckdb
except ImportError:
    sys.exit("duckdb 가 필요합니다:  pip install duckdb pyarrow")

# 컬럼 이름만 보고도 민감할 수 있는 것은 기본 마스킹한다.
DEFAULT_MASK_PATTERNS = [
    r"operator", r"worker", r"employee", r"user", r"작업자", r"사번",
    r"customer", r"client", r"고객",
    r"password", r"secret", r"token",
]

# 계보 준비도 진단에서 찾는 개념과, 소스 컬럼명 후보 패턴
CONCEPT_HINTS = {
    "roll_id":          [r"roll", r"롤", r"coil"],
    "position_m":       [r"position", r"pos_?m", r"length", r"meter", r"위치", r"거리"],
    "lane_no":          [r"lane", r"레인", r"track"],
    "electrode_lot_id": [r"e(lec)?t?_?lot", r"electrode.*lot", r"전극.*lot", r"lot.*전극"],
    "cell_id":          [r"cell", r"셀"],
    "module_id":        [r"module", r"모듈"],
    "pack_id":          [r"pack", r"팩"],
    "electrode_role":   [r"role", r"polarity", r"cathode", r"anode", r"극성", r"양극", r"음극"],
    "line_speed":       [r"speed", r"속도", r"mpm"],
    "usl":              [r"usl", r"upper.*limit", r"상한"],
    "lsl":              [r"lsl", r"lower.*limit", r"하한"],
}


def masked(col: str, extra: list[str]) -> bool:
    pats = DEFAULT_MASK_PATTERNS + [re.escape(e) for e in extra if e and not e.startswith("<")]
    return any(re.search(p, col, re.I) for p in pats)


def find_tables(landing: pathlib.Path) -> dict[str, str]:
    """랜딩 폴더에서 Parquet 를 가진 테이블(폴더)을 찾는다.

    파티션 폴더(dt=..., hour=...)는 테이블로 세지 않고 상위로 접는다.
    """
    tables: dict[str, str] = {}
    for p in sorted(landing.rglob("*.parquet")):
        rel = p.relative_to(landing)
        parts = [s for s in rel.parts[:-1] if "=" not in s]
        name = parts[0] if parts else "_root"
        tables.setdefault(name, str(landing / name / "**" / "*.parquet"))
    if not tables:
        for p in sorted(landing.glob("*.parquet")):
            tables[p.stem] = str(p)
    return tables


def profile_table(con, name: str, pattern: str, mask_extra: list[str],
                  sample_values: int) -> dict:
    """테이블 하나의 비식별 프로파일."""
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW t_{name} AS "
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true, union_by_name=true)"
        )
    except Exception as exc:
        return {"table": name, "error": str(exc)}

    cols = con.execute(f"DESCRIBE t_{name}").df()
    n = con.execute(f"SELECT COUNT(*) c FROM t_{name}").df().c[0]

    out = {"table": name, "rows": int(n), "columns": []}
    for _, r in cols.iterrows():
        col, typ = r.column_name, str(r.column_type)
        q = f'"{col}"'
        info: dict = {"name": col, "type": typ}
        try:
            stats = con.execute(f"""
                SELECT COUNT({q}) nn, COUNT(DISTINCT {q}) nd FROM t_{name}
            """).df()
            nn, nd = int(stats.nn[0]), int(stats.nd[0])
            info["null_pct"] = round(100 * (1 - nn / n), 2) if n else None
            info["distinct"] = nd
            info["is_unique"] = bool(n and nd == n)
        except Exception:
            pass

        is_masked = masked(col, mask_extra)
        info["masked"] = is_masked

        low = typ.upper()
        try:
            if any(k in low for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "HUGEINT")):
                s = con.execute(f"""
                    SELECT MIN({q}) mn, MAX({q}) mx, AVG({q}) av,
                           STDDEV_SAMP({q}) sd,
                           QUANTILE_CONT({q}, 0.5) med FROM t_{name}
                """).df()
                info["numeric"] = {
                    "min": None if s.mn[0] is None else float(s.mn[0]),
                    "max": None if s.mx[0] is None else float(s.mx[0]),
                    "mean": None if s.av[0] is None else round(float(s.av[0]), 4),
                    "median": None if s.med[0] is None else round(float(s.med[0]), 4),
                    "sd": None if s.sd[0] is None else round(float(s.sd[0]), 4),
                }
            elif "TIMESTAMP" in low or "DATE" in low:
                s = con.execute(f"SELECT MIN({q}) mn, MAX({q}) mx FROM t_{name}").df()
                info["time_range"] = [str(s.mn[0]), str(s.mx[0])]
                info["tz_aware"] = "WITH TIME ZONE" in low or "TZ" in low
            elif sample_values and not is_masked:
                s = con.execute(f"""
                    SELECT {q} v, COUNT(*) c FROM t_{name}
                    WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {sample_values}
                """).df()
                info["top_values"] = [str(v) for v in s.v.tolist()]
        except Exception:
            pass

        out["columns"].append(info)
    return out


def guess_concepts(profiles: list[dict]) -> dict[str, list[str]]:
    """컬럼명에서 온톨로지 개념 후보를 추정한다 (사람이 확인해야 한다)."""
    found: dict[str, list[str]] = {k: [] for k in CONCEPT_HINTS}
    for t in profiles:
        for c in t.get("columns", []):
            for concept, pats in CONCEPT_HINTS.items():
                if any(re.search(p, c["name"], re.I) for p in pats):
                    found[concept].append(f'{t["table"]}.{c["name"]}')
    return found


def readiness(profiles: list[dict], concepts: dict) -> list[dict]:
    """계보를 만들 수 있는 상태인지 진단한다."""
    checks = []

    def add(key, ok, detail, why):
        checks.append({"check": key, "status": ok, "detail": detail, "why": why})

    add("롤 식별자", "OK" if concepts["roll_id"] else "없음",
        ", ".join(concepts["roll_id"][:4]) or "후보 컬럼을 찾지 못함",
        "연속 좌표계의 기준이다. 없으면 전극 공정 추적이 성립하지 않는다.")

    pos = concepts["position_m"]
    add("롤 내 위치 (position_m)", "OK" if pos else "없음",
        ", ".join(pos[:4]) or "후보 컬럼을 찾지 못함",
        "★ 가장 중요. 없으면 계보가 롤 단위에서 끊기고 구간 단위 원인 분석이 불가능하다. "
        "라인 속도가 있으면 파생 가능하다.")

    add("레인 (lane_no)", "OK" if concepts["lane_no"] else "없음",
        ", ".join(concepts["lane_no"][:4]) or "후보 컬럼을 찾지 못함",
        "레인을 구분하지 않으면 레인별 두께 편차가 평균에 묻힌다.")

    elot = concepts["electrode_lot_id"]
    add("전극 Lot", "OK" if elot else "없음",
        ", ".join(elot[:4]) or "후보 컬럼을 찾지 못함",
        "★ 연속 좌표계와 이산 좌표계를 잇는 전환점이다.")

    both = bool(elot and pos)
    add("전극 Lot ↔ 롤 위치 연결", "OK" if both else "확인 필요",
        "전극 Lot 테이블에 위치 컬럼이 함께 있는지 직접 확인할 것",
        "★★ 이 연결이 이 프로젝트의 성패다. 같은 테이블에 Lot ID 와 "
        "롤 구간(시작/끝 m)이 함께 있어야 한다.")

    add("셀 → 모듈 → 팩", "OK" if (concepts["cell_id"] and concepts["module_id"]) else "부분",
        f'셀 {len(concepts["cell_id"])}건 · 모듈 {len(concepts["module_id"])}건 '
        f'· 팩 {len(concepts["pack_id"])}건 후보',
        "이산 좌표계의 계보. 배터리 여권의 셀 단위 추적에도 필요하다.")

    add("양극/음극 구분", "OK" if concepts["electrode_role"] else "없음",
        ", ".join(concepts["electrode_role"][:4]) or "후보 컬럼을 찾지 못함",
        "셀에는 양극 Lot 과 음극 Lot 이 각각 들어간다. 구분이 없으면 "
        "한쪽 전극만 보고 원인을 판단하게 된다.")

    add("규격 (USL/LSL)", "OK" if (concepts["usl"] or concepts["lsl"]) else "없음",
        ", ".join((concepts["usl"] + concepts["lsl"])[:4]) or "후보 컬럼을 찾지 못함",
        "측정값에 규격이 함께 오지 않으면 별도 스펙 마스터와 조인해야 한다. "
        "레시피/스텝별 버전 관리가 흔한 난관이다.")

    add("라인 속도", "OK" if concepts["line_speed"] else "없음",
        ", ".join(concepts["line_speed"][:4]) or "후보 컬럼을 찾지 못함",
        "position_m 이 없을 때 위치를 파생시키는 유일한 입력이다.")

    # 시각 컬럼의 시간대 일관성
    tz = {}
    for t in profiles:
        for c in t.get("columns", []):
            if "time_range" in c:
                tz.setdefault(bool(c.get("tz_aware")), []).append(f'{t["table"]}.{c["name"]}')
    add("타임스탬프 시간대", "OK" if len(tz) <= 1 else "불일치",
        " / ".join(f'{"tz있음" if k else "tz없음"}: {len(v)}개' for k, v in tz.items()) or "시각 컬럼 없음",
        "FDC 와 MES 의 시간대 표기가 다르면 구간 조인이 조용히 어긋난다.")

    return checks


def write_report(out_dir: pathlib.Path, landing: pathlib.Path,
                 profiles: list[dict], concepts: dict, checks: list[dict],
                 sample_values: int) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "site_profile.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(),
                    "landing": str(landing), "tables": profiles,
                    "concepts": concepts, "readiness": checks},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("# 사내 데이터 프로파일 (비식별)\n")
    L.append(f"- 생성: {datetime.now():%Y-%m-%d %H:%M}")
    L.append(f"- 대상: `{landing}`")
    L.append(f"- 테이블 {len(profiles)}개")
    L.append("")
    L.append("> 이 보고서에는 실제 측정값·식별자가 들어 있지 않다. "
             "컬럼 구조와 통계 요약만 담는다."
             + ("" if sample_values == 0 else
                f" (단, `--sample-values {sample_values}` 로 값 예시가 포함돼 있다. "
                "공유 전 반드시 확인할 것.)"))
    L.append("")

    L.append("## 온톨로지 준비도\n")
    L.append("| 항목 | 상태 | 근거 |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f'| {c["check"]} | **{c["status"]}** | {c["detail"]} |')
    L.append("")
    L.append("### 항목별 의미\n")
    for c in checks:
        L.append(f'- **{c["check"]}** — {c["why"]}')
    L.append("")

    L.append("## 개념 후보 (컬럼명 추정 — 사람이 확인 필요)\n")
    L.append("| 온톨로지 개념 | 후보 컬럼 |")
    L.append("|---|---|")
    for k, v in concepts.items():
        L.append(f'| `{k}` | {", ".join(f"`{x}`" for x in v[:6]) or "—"} |')
    L.append("")

    L.append("## 테이블별 구조\n")
    for t in profiles:
        if "error" in t:
            L.append(f'### {t["table"]}\n\n읽기 실패: `{t["error"]}`\n')
            continue
        L.append(f'### {t["table"]}  ({t["rows"]:,}행)\n')
        L.append("| 컬럼 | 타입 | 널% | 고유값 | 요약 |")
        L.append("|---|---|---|---|---|")
        for c in t["columns"]:
            summary = ""
            if "numeric" in c and c["numeric"]["min"] is not None:
                nmr = c["numeric"]
                summary = (f'min {nmr["min"]:g} · med {nmr["median"]:g} · '
                           f'max {nmr["max"]:g} · sd {nmr["sd"] if nmr["sd"] is not None else "—"}')
            elif "time_range" in c:
                summary = f'{c["time_range"][0]} ~ {c["time_range"][1]}'
                summary += " (tz있음)" if c.get("tz_aware") else " (tz없음)"
            elif "top_values" in c:
                summary = ", ".join(f"`{v}`" for v in c["top_values"])
            elif c.get("masked"):
                summary = "_마스킹됨_"
            uniq = " (고유)" if c.get("is_unique") else ""
            L.append(f'| `{c["name"]}` | {c["type"]} | {c.get("null_pct", "—")} '
                     f'| {c.get("distinct", "—"):,}{uniq} | {summary} |'
                     if isinstance(c.get("distinct"), int) else
                     f'| `{c["name"]}` | {c["type"]} | {c.get("null_pct", "—")} | — | {summary} |')
        L.append("")

    path = out_dir / "site_profile.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landing", required=True, help="Parquet 랜딩 폴더")
    ap.add_argument("--mapping", help="config/site/mappings/*.yaml (있으면 마스킹 설정을 읽는다)")
    ap.add_argument("--out", default="profiles_out", help="산출 폴더 (기본: profiles_out)")
    ap.add_argument("--sample-values", type=int, default=0,
                    help="문자열 컬럼의 상위 값 N개 포함 (기본 0 = 값 미포함)")
    args = ap.parse_args()

    landing = pathlib.Path(args.landing).expanduser()
    if not landing.is_dir():
        sys.exit(f"랜딩 폴더를 찾을 수 없습니다: {landing}")

    mask_extra: list[str] = []
    if args.mapping:
        import yaml
        cfg = yaml.safe_load(pathlib.Path(args.mapping).read_text(encoding="utf-8")) or {}
        mask_extra = (cfg.get("profiling") or {}).get("mask_columns") or []

    tables = find_tables(landing)
    if not tables:
        sys.exit(f"{landing} 에서 Parquet 파일을 찾지 못했습니다.")

    print(f"테이블 {len(tables)}개 발견")
    con = duckdb.connect()
    profiles = []
    for name, pattern in tables.items():
        print(f"  프로파일링: {name}")
        profiles.append(profile_table(con, re.sub(r"\W", "_", name), pattern,
                                      mask_extra, args.sample_values))
        profiles[-1]["table"] = name

    concepts = guess_concepts(profiles)
    checks = readiness(profiles, concepts)

    out_dir = pathlib.Path(args.out)
    path = write_report(out_dir, landing, profiles, concepts, checks, args.sample_values)

    print(f"\n보고서: {path}")
    print("\n온톨로지 준비도")
    for c in checks:
        mark = "OK " if c["status"] == "OK" else "!! "
        print(f"  {mark}{c['check']:<28} {c['status']}")
    print(f"\n{out_dir}/ 는 .gitignore 대상입니다. 공유 전 내용을 직접 확인하십시오.")


if __name__ == "__main__":
    main()
