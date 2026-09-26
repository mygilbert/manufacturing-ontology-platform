#!/usr/bin/env python3
"""배터리 계보 질의 데모 — 설계가 실제로 작동하는지 확인한다

DB 구축 없이 Parquet 를 DuckDB 로 직접 질의한다.
실데이터가 오기 전에 조인 설계를 검증하는 것이 목적이며,
사내 반입 승인도 `pip install duckdb` 하나면 끝난다.

이 스크립트는 **이상이 어디 있는지 모르는 상태에서 시작한다.**
계보를 타고 되짚어 스스로 찾아내야 설계가 검증된 것이다.

  Q1  전극 Lot 의 롤 구간에 코팅 두께를 붙인다 (구간 조인)
  Q2  불량 셀에서 원인 구간을 역추적한다          <- 근본원인
  Q3  이상 구간에서 영향받은 셀/모듈/팩을 찾는다   <- 격리·리콜 범위
  Q4  추적 신뢰도별로 분리 집계한다               <- 오염 방지
  Q5  위치 신뢰도가 낮은 구간을 분리한다

사용법
  pip install duckdb
  python scripts/generate_battery_genealogy.py
  python scripts/demo_genealogy_queries.py [--landing landing]
"""
from __future__ import annotations

import argparse
import pathlib
import textwrap

import duckdb


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(con: duckdb.DuckDBPyConnection, sql: str, limit: int = 12) -> None:
    df = con.execute(sql).df()
    if len(df) > limit:
        print(df.head(limit).to_string(index=False))
        print(f"... 총 {len(df):,}행")
    else:
        print(df.to_string(index=False))


def register(con: duckdb.DuckDBPyConnection, landing: pathlib.Path) -> None:
    """랜딩 폴더의 Parquet 를 뷰로 등록한다.

    _SUCCESS 마커가 있는 폴더만 읽어 부분 쓰기 파일을 피한다.
    (실제 배치에서도 같은 규칙을 쓴다)
    """
    for table in [
        "rolls", "web_measurements", "roll_position_map", "electrode_lots",
        "cells", "supplies_cell", "cell_tests",
        "modules", "module_cells", "packs", "pack_modules",
    ]:
        pattern = str(landing / table / "**" / "*.parquet")
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landing", default="landing")
    args = ap.parse_args()

    landing = pathlib.Path(args.landing)
    if not landing.is_dir():
        raise SystemExit(
            f"{landing}/ 가 없습니다. 먼저 실행하세요:\n"
            f"  python scripts/generate_battery_genealogy.py"
        )

    con = duckdb.connect()
    register(con, landing)

    # ------------------------------------------------------------------
    banner("[0] 데이터 규모")
    show(con, """
        SELECT '롤' AS 대상, COUNT(*) AS 건수 FROM rolls
        UNION ALL SELECT '웹 측정값', COUNT(*) FROM web_measurements
        UNION ALL SELECT '전극 Lot', COUNT(*) FROM electrode_lots
        UNION ALL SELECT '셀', COUNT(*) FROM cells
        UNION ALL SELECT '계보 엣지', COUNT(*) FROM supplies_cell
        UNION ALL SELECT '모듈', COUNT(*) FROM modules
        UNION ALL SELECT '팩', COUNT(*) FROM packs
    """)

    # ------------------------------------------------------------------
    banner("[Q1] 구간 조인 — 전극 Lot 에 코팅 두께를 붙인다")
    print(textwrap.dedent("""
        연속 좌표계(roll_id, position_m)의 측정값을
        이산 단위(ElectrodeLot)에 연결하는 핵심 조인.

          ON  m.roll_id = l.roll_id
          AND m.lane_no = l.lane_no
          AND m.position_m >= l.position_start_m
          AND m.position_m <  l.position_end_m     -- 반열린 구간
    """).strip())

    con.execute("""
        CREATE OR REPLACE VIEW lot_thickness AS
        SELECT
            l.electrode_lot_id,
            l.roll_id,
            l.lane_no,
            l.position_start_m,
            l.position_end_m,
            l.traceability_level,
            COUNT(m.value)                                   AS n_meas,
            ROUND(AVG(m.value), 2)                           AS thickness_avg,
            ROUND(MAX(m.value), 2)                           AS thickness_max,
            ROUND(STDDEV_SAMP(m.value), 2)                   AS thickness_sd,
            SUM(CASE WHEN m.value > m.usl THEN 1 ELSE 0 END) AS over_usl,
            SUM(CASE WHEN m.position_confidence = 'LOW'
                     THEN 1 ELSE 0 END)                      AS low_conf_meas
        FROM electrode_lots l
        LEFT JOIN web_measurements m
               ON m.roll_id    = l.roll_id
              AND m.lane_no    = l.lane_no
              AND m.position_m >= l.position_start_m
              AND m.position_m <  l.position_end_m
             AND m.observed_property = 'coating_thickness'
        GROUP BY ALL
    """)

    print("\n-- 조인율 (붙지 않은 Lot 이 있으면 설계가 틀린 것) --")
    show(con, """
        SELECT
            COUNT(*)                                        AS lot_수,
            SUM(CASE WHEN n_meas = 0 THEN 1 ELSE 0 END)     AS 측정값_없는_lot,
            ROUND(100.0 * SUM(CASE WHEN n_meas > 0 THEN 1 ELSE 0 END)
                  / COUNT(*), 1)                            AS 조인율_pct,
            ROUND(AVG(n_meas), 1)                           AS lot당_평균_측정값
        FROM lot_thickness
    """)

    # ------------------------------------------------------------------
    banner("[Q2] 역추적 — 불량 셀에서 원인 구간을 찾는다  ★근본원인")
    print(textwrap.dedent("""
        시작점은 "용량이 낮은 셀"뿐이다. 어느 롤의 어느 구간이 문제인지는 모른다.

          Cell -> SUPPLIES_CELL(CATHODE) -> ElectrodeLot
               -> roll_id + [position_start_m, position_end_m)
               -> web_measurements 구간 집계
    """).strip())

    print("\n-- 용량 하위 셀이 어느 양극 Lot 에서 나왔나 --")
    show(con, """
        WITH low_cells AS (
            SELECT t.cell_id, t.value AS capacity
            FROM cell_tests t
            WHERE t.test_type = 'EOL_CAPACITY'
            ORDER BY t.value ASC
            LIMIT 60                      -- 하위 60개만 본다
        )
        SELECT
            s.electrode_lot_id,
            lt.roll_id,
            lt.lane_no,
            lt.position_start_m  AS 시작_m,
            lt.position_end_m    AS 끝_m,
            COUNT(*)             AS 불량셀_수,
            ROUND(AVG(c.capacity), 2) AS 평균용량_Ah,
            lt.thickness_avg     AS 두께평균_um,
            lt.over_usl          AS USL초과_측정점
        FROM low_cells c
        JOIN supplies_cell s ON s.cell_id = c.cell_id AND s.electrode_role = 'CATHODE'
        JOIN lot_thickness lt ON lt.electrode_lot_id = s.electrode_lot_id
        GROUP BY ALL
        HAVING COUNT(*) >= 3
        ORDER BY 불량셀_수 DESC, 평균용량_Ah ASC
    """)

    print("\n-- 지목된 구간의 실제 두께 프로파일 (5m 단위) --")
    show(con, """
        WITH suspect AS (
            SELECT lt.roll_id, lt.lane_no,
                   MIN(lt.position_start_m) AS lo,
                   MAX(lt.position_end_m)   AS hi
            FROM cell_tests t
            JOIN supplies_cell s ON s.cell_id = t.cell_id AND s.electrode_role='CATHODE'
            JOIN lot_thickness lt ON lt.electrode_lot_id = s.electrode_lot_id
            WHERE t.test_type='EOL_CAPACITY'
            GROUP BY lt.roll_id, lt.lane_no, lt.electrode_lot_id
            HAVING AVG(t.value) < (SELECT AVG(value) - 1.5*STDDEV_SAMP(value)
                                   FROM cell_tests WHERE test_type='EOL_CAPACITY')
        )
        SELECT
            FLOOR(m.position_m / 5) * 5      AS 위치_m,
            ROUND(AVG(m.value), 2)           AS 두께_um,
            CASE WHEN AVG(m.value) > MAX(m.usl) THEN '>>> USL 초과' ELSE '' END AS 판정
        FROM web_measurements m
        JOIN suspect s ON s.roll_id = m.roll_id AND s.lane_no = m.lane_no
        WHERE m.observed_property = 'coating_thickness'
          AND m.position_m >= s.lo - 20 AND m.position_m < s.hi + 20
        GROUP BY 1
        ORDER BY 1
    """, limit=30)

    # ------------------------------------------------------------------
    banner("[Q3] 순추적 — 이상 구간의 영향 범위  ★격리·리콜")
    print(textwrap.dedent("""
        반대 방향. USL 을 넘은 구간이 들어간 셀/모듈/팩을 전부 찾는다.
        배터리 여권의 추적성 요구와 직결되는 질의다.

        구간 겹침 조건:
          rel.position_start_m < :end  AND  rel.position_end_m > :start
    """).strip())

    # USL 초과 "지점"의 MIN/MAX 를 구간으로 삼으면 안 된다.
    # 노이즈로 인한 산발적 초과가 롤 전체에 흩어져 있어 구간이 롤 전체로 부풀고,
    # 그 결과 영향 범위가 과대 산정되어 격리 판단이 무의미해진다.
    # 연속 구간(island)을 찾아 최소 길이 이상인 것만 이상 구간으로 본다.
    con.execute("""
        CREATE OR REPLACE VIEW bad_region AS
        WITH flagged AS (
            SELECT roll_id, lane_no, position_m,
                   CASE WHEN value > usl THEN 1 ELSE 0 END AS over
            FROM web_measurements
            WHERE observed_property = 'coating_thickness'
        ),
        gaps AS (          -- 직전 초과 지점과의 간격으로 구간을 끊는다
            SELECT roll_id, lane_no, position_m,
                   position_m - LAG(position_m) OVER (
                       PARTITION BY roll_id, lane_no ORDER BY position_m
                   ) AS gap
            FROM flagged WHERE over = 1
        ),
        islands AS (
            SELECT roll_id, lane_no, position_m,
                   SUM(CASE WHEN gap IS NULL OR gap > 2.0 THEN 1 ELSE 0 END)
                       OVER (PARTITION BY roll_id, lane_no
                             ORDER BY position_m ROWS UNBOUNDED PRECEDING) AS island_id
            FROM gaps
        )
        SELECT roll_id, lane_no, island_id,
               MIN(position_m) AS start_m,
               MAX(position_m) AS end_m,
               COUNT(*)        AS n_points,
               MAX(position_m) - MIN(position_m) AS length_m
        FROM islands
        GROUP BY ALL
        HAVING COUNT(*) >= 4 AND MAX(position_m) - MIN(position_m) >= 2.0
    """)

    print("\n-- 데이터에서 찾아낸 이상 구간 (연속 구간만) --")
    show(con, """
        SELECT roll_id, lane_no,
               start_m AS 시작_m, end_m AS 끝_m,
               ROUND(length_m, 1) AS 길이_m, n_points AS 초과_측정점
        FROM bad_region ORDER BY length_m DESC
    """)

    print("\n-- 영향 범위 --")
    show(con, """
        WITH affected_lots AS (
            SELECT DISTINCT l.electrode_lot_id
            FROM electrode_lots l
            JOIN bad_region b
              ON b.roll_id = l.roll_id
             AND b.lane_no = l.lane_no
             AND l.position_start_m < b.end_m      -- 구간 겹침
             AND l.position_end_m   > b.start_m
        )
        SELECT
            (SELECT COUNT(*) FROM bad_region)                       AS 이상구간_수,
            (SELECT COUNT(*) FROM affected_lots)                    AS 영향_전극Lot,
            COUNT(DISTINCT s.cell_id)                               AS 영향_셀,
            COUNT(DISTINCT mc.module_id)                            AS 영향_모듈,
            COUNT(DISTINCT pm.pack_id)                              AS 영향_팩,
            ROUND(100.0 * COUNT(DISTINCT s.cell_id)
                  / (SELECT COUNT(*) FROM cells), 1)                AS 셀_비율_pct
        FROM affected_lots a
        JOIN supplies_cell s  ON s.electrode_lot_id = a.electrode_lot_id
        LEFT JOIN module_cells mc ON mc.cell_id = s.cell_id
        LEFT JOIN pack_modules pm ON pm.module_id = mc.module_id
    """)

    print("\n-- 영향받은 셀 vs 나머지: 용량 차이가 실제로 나는가 --")
    show(con, """
        WITH affected AS (
            SELECT DISTINCT s.cell_id
            FROM electrode_lots l
            JOIN bad_region b ON b.roll_id=l.roll_id AND b.lane_no=l.lane_no
                             AND l.position_start_m < b.end_m AND l.position_end_m > b.start_m
            JOIN supplies_cell s ON s.electrode_lot_id = l.electrode_lot_id
        ),
        grp AS (
            SELECT CASE WHEN a.cell_id IS NULL THEN '나머지 셀' ELSE '영향받은 셀' END AS 구분,
                   t.value
            FROM cell_tests t
            LEFT JOIN affected a ON a.cell_id = t.cell_id
            WHERE t.test_type='EOL_CAPACITY'
        )
        SELECT 구분,
               COUNT(*)                       AS 셀_수,
               ROUND(AVG(value), 3)           AS 평균용량_Ah,
               ROUND(STDDEV_SAMP(value), 3)   AS 표준편차,
               ROUND(MIN(value), 2)           AS 최소
        FROM grp GROUP BY 1 ORDER BY 1
    """)

    print("\n-- 효과 크기 (Cohen's d) --")
    show(con, """
        WITH affected AS (
            SELECT DISTINCT s.cell_id
            FROM electrode_lots l
            JOIN bad_region b ON b.roll_id=l.roll_id AND b.lane_no=l.lane_no
                             AND l.position_start_m < b.end_m AND l.position_end_m > b.start_m
            JOIN supplies_cell s ON s.electrode_lot_id = l.electrode_lot_id
        ),
        g AS (
            SELECT (a.cell_id IS NOT NULL) AS hit, t.value
            FROM cell_tests t LEFT JOIN affected a ON a.cell_id = t.cell_id
            WHERE t.test_type='EOL_CAPACITY'
        ),
        st AS (
            SELECT
                AVG(CASE WHEN hit THEN value END)              AS m1,
                AVG(CASE WHEN NOT hit THEN value END)          AS m0,
                STDDEV_SAMP(CASE WHEN hit THEN value END)      AS s1,
                STDDEV_SAMP(CASE WHEN NOT hit THEN value END)  AS s0
            FROM g
        )
        SELECT ROUND(m0 - m1, 3) AS 용량차이_Ah,
               ROUND((m0 - m1) / SQRT((s1*s1 + s0*s0)/2), 2) AS cohens_d,
               CASE WHEN ABS((m0-m1)/SQRT((s1*s1+s0*s0)/2)) >= 0.8 THEN '큼 (large)'
                    WHEN ABS((m0-m1)/SQRT((s1*s1+s0*s0)/2)) >= 0.5 THEN '중간 (medium)'
                    WHEN ABS((m0-m1)/SQRT((s1*s1+s0*s0)/2)) >= 0.2 THEN '작음 (small)'
                    ELSE '무시할 수준' END AS 효과크기
        FROM st
    """)

    # ------------------------------------------------------------------
    banner("[Q4] 추적 신뢰도별 분리 — 섞으면 분석이 오염된다")
    print(textwrap.dedent("""
        traceability_level 을 구분하지 않으면 추정 계보와 실측 계보가 섞인다.
        INFERRED 계보로 "이 구간이 원인"이라고 단정하면 현장 신뢰를 잃는다.
    """).strip())

    print("\n-- 전극 Lot 기준 --")
    show(con, """
        SELECT traceability_level AS 추적수준, COUNT(*) AS lot_수,
               CASE traceability_level
                   WHEN 'MEASURED'  THEN '신뢰 가능'
                   WHEN 'INFERRED'  THEN '참고만. 단정 금지'
                   WHEN 'ROLL_ONLY' THEN '구간 분석 불가'
                   ELSE '제외' END AS 분석에서
        FROM electrode_lots GROUP BY ALL ORDER BY lot_수 DESC
    """)

    # 셀 하나에는 양극/음극 Lot 이 각각 붙는다. 단순 집계하면 셀이 중복 계수된다.
    # 계보의 신뢰도는 둘 중 **낮은 쪽**을 따른다.
    print("\n-- 셀 기준 (양극/음극 중 낮은 쪽을 따름) --")
    show(con, """
        WITH cell_level AS (
            SELECT s.cell_id,
                   MAX(CASE l.traceability_level      -- 숫자가 클수록 나쁨
                           WHEN 'MEASURED'  THEN 0
                           WHEN 'INFERRED'  THEN 1
                           WHEN 'ROLL_ONLY' THEN 2
                           ELSE 3 END) AS worst
            FROM supplies_cell s
            JOIN electrode_lots l ON l.electrode_lot_id = s.electrode_lot_id
            GROUP BY s.cell_id
        )
        SELECT
            CASE worst WHEN 0 THEN 'MEASURED' WHEN 1 THEN 'INFERRED'
                       WHEN 2 THEN 'ROLL_ONLY' ELSE 'UNKNOWN' END AS 유효_추적수준,
            COUNT(*) AS 셀_수,
            ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM cells), 1) AS 비율_pct
        FROM cell_level GROUP BY 1 ORDER BY 셀_수 DESC
    """)

    print("\n-- 계보 완결성 (genealogy_complete) --")
    show(con, """
        SELECT
            genealogy_complete AS 계보완결,
            COUNT(*)           AS 셀_수,
            ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM cells), 1) AS 비율_pct
        FROM cells GROUP BY 1 ORDER BY 1 DESC
    """)

    # ------------------------------------------------------------------
    banner("[Q5] 위치 신뢰도 — 시간→위치 변환이 깨지는 구간")
    print(textwrap.dedent("""
        position_m = ∫ line_speed dt 는 정지/가감속 구간에서 깨진다.
        LOW 구간을 그대로 근본원인 분석에 쓰면 엉뚱한 구간을 지목하게 된다.
    """).strip())

    show(con, """
        SELECT
            segment_type        AS 구간유형,
            confidence          AS 신뢰도,
            COUNT(*)            AS 구간_수,
            ROUND(AVG(position_end_m - position_start_m), 1) AS 평균길이_m,
            ROUND(AVG(avg_speed_mpm), 1) AS 평균속도_mpm
        FROM roll_position_map
        GROUP BY ALL ORDER BY 구간유형
    """)

    print("\n-- 위치 신뢰도가 낮은 측정값을 포함한 Lot --")
    show(con, """
        SELECT
            electrode_lot_id, roll_id, lane_no,
            position_start_m AS 시작_m, n_meas AS 측정점,
            low_conf_meas    AS 저신뢰_측정점,
            ROUND(100.0*low_conf_meas/NULLIF(n_meas,0), 1) AS 저신뢰_비율_pct
        FROM lot_thickness
        WHERE low_conf_meas > 0
        ORDER BY low_conf_meas DESC
    """, limit=8)

    banner("정리")
    print(textwrap.dedent("""
      - Q1 구간 조인이 성립하면 연속/이산 좌표계가 연결된 것이다
      - Q2 는 "용량 낮은 셀"만 알고 시작해 원인 구간을 스스로 지목했다
      - Q3 은 반대 방향으로 격리 범위를 산정했다 (여권 추적성과 동일 경로)
      - Q4/Q5 는 신뢰할 수 없는 계보를 분리해 결론의 오염을 막는다

      실데이터가 오면 바꿀 것은 테이블/컬럼명뿐이고 질의 구조는 그대로다.
    """).strip())


if __name__ == "__main__":
    main()
