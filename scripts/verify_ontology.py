#!/usr/bin/env python3
"""
Manufacturing Ontology 검증 스크립트

이 스크립트는 PostgreSQL + Apache AGE에 생성된 온톨로지 스키마와
샘플 데이터를 검증합니다.

사용법:
    python scripts/verify_ontology.py

환경 변수:
    POSTGRES_HOST: PostgreSQL 호스트 (기본: localhost)
    POSTGRES_PORT: PostgreSQL 포트 (기본: 5432)
    POSTGRES_USER: 사용자 (기본: ontology)
    POSTGRES_PASSWORD: 비밀번호 (기본: ontology123)
    POSTGRES_DB: 데이터베이스 (기본: manufacturing)
"""

import os
import sys
import json
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("psycopg2가 필요합니다. 설치하세요: pip install psycopg2-binary")
    sys.exit(1)


# 데이터베이스 연결 설정
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USER", "ontology"),
    "password": os.getenv("POSTGRES_PASSWORD", "ontology123"),
    "database": os.getenv("POSTGRES_DB", "manufacturing"),
}


def get_connection():
    """PostgreSQL 연결 생성"""
    return psycopg2.connect(**DB_CONFIG)


def execute_cypher(conn, query: str) -> list[dict[str, Any]]:
    """Cypher 쿼리 실행"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # AGE 로드
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        # Cypher 쿼리 실행
        full_query = f"SELECT * FROM cypher('manufacturing', $$ {query} $$) AS (result agtype)"
        cur.execute(full_query)
        results = cur.fetchall()

        # agtype을 Python 객체로 변환
        parsed_results = []
        for row in results:
            result = row["result"]
            if isinstance(result, str):
                try:
                    parsed_results.append(json.loads(result))
                except json.JSONDecodeError:
                    parsed_results.append(result)
            else:
                parsed_results.append(result)

        return parsed_results


def verify_vertex_labels(conn) -> bool:
    """Vertex 레이블 검증"""
    print("\n" + "=" * 60)
    print("1. Vertex Labels (Object Types) 검증")
    print("=" * 60)

    expected_labels = ["Equipment", "Process", "Lot", "Wafer", "Recipe", "Measurement", "Alarm"]
    success = True

    with conn.cursor() as cur:
        cur.execute("""
            SELECT name FROM ag_catalog.ag_label
            WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = 'manufacturing')
            AND kind = 'v'
        """)
        existing_labels = [row[0] for row in cur.fetchall()]

    for label in expected_labels:
        if label in existing_labels:
            print(f"  [OK] {label}")
        else:
            print(f"  [FAIL] {label} - 존재하지 않음")
            success = False

    return success


def verify_edge_labels(conn) -> bool:
    """Edge 레이블 검증"""
    print("\n" + "=" * 60)
    print("2. Edge Labels (Link Types) 검증")
    print("=" * 60)

    expected_labels = [
        "PROCESSED_AT", "MEASURED_BY", "BELONGS_TO", "GENERATES_ALARM",
        "USES_RECIPE", "FOLLOWS_ROUTE", "NEXT_STEP", "AFFECTS_LOT", "CONTAINS_WAFER"
    ]
    success = True

    with conn.cursor() as cur:
        cur.execute("""
            SELECT name FROM ag_catalog.ag_label
            WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = 'manufacturing')
            AND kind = 'e'
        """)
        existing_labels = [row[0] for row in cur.fetchall()]

    for label in expected_labels:
        if label in existing_labels:
            print(f"  [OK] {label}")
        else:
            print(f"  [FAIL] {label} - 존재하지 않음")
            success = False

    return success


def verify_sample_data(conn) -> bool:
    """샘플 데이터 검증"""
    print("\n" + "=" * 60)
    print("3. 샘플 데이터 검증")
    print("=" * 60)

    success = True

    # Equipment 수 확인
    results = execute_cypher(conn, "MATCH (n:Equipment) RETURN count(n) as count")
    count = results[0] if results else 0
    print(f"  Equipment: {count}개", end="")
    if count >= 5:
        print(" [OK]")
    else:
        print(" [FAIL] - 5개 이상 필요")
        success = False

    # Process 수 확인
    results = execute_cypher(conn, "MATCH (n:Process) RETURN count(n) as count")
    count = results[0] if results else 0
    print(f"  Process: {count}개", end="")
    if count >= 4:
        print(" [OK]")
    else:
        print(" [FAIL] - 4개 이상 필요")
        success = False

    # Lot 수 확인
    results = execute_cypher(conn, "MATCH (n:Lot) RETURN count(n) as count")
    count = results[0] if results else 0
    print(f"  Lot: {count}개", end="")
    if count >= 3:
        print(" [OK]")
    else:
        print(" [FAIL] - 3개 이상 필요")
        success = False

    # Wafer 수 확인
    results = execute_cypher(conn, "MATCH (n:Wafer) RETURN count(n) as count")
    count = results[0] if results else 0
    print(f"  Wafer: {count}개", end="")
    if count >= 3:
        print(" [OK]")
    else:
        print(" [FAIL] - 3개 이상 필요")
        success = False

    # Alarm 수 확인
    results = execute_cypher(conn, "MATCH (n:Alarm) RETURN count(n) as count")
    count = results[0] if results else 0
    print(f"  Alarm: {count}개", end="")
    if count >= 2:
        print(" [OK]")
    else:
        print(" [FAIL] - 2개 이상 필요")
        success = False

    return success


def verify_relationships(conn) -> bool:
    """관계 검증"""
    print("\n" + "=" * 60)
    print("4. 관계(Edge) 검증")
    print("=" * 60)

    success = True

    # PROCESSED_AT 관계 확인
    results = execute_cypher(conn, "MATCH ()-[r:PROCESSED_AT]->() RETURN count(r) as count")
    count = results[0] if results else 0
    print(f"  PROCESSED_AT: {count}개", end="")
    if count >= 2:
        print(" [OK]")
    else:
        print(" [FAIL]")
        success = False

    # BELONGS_TO 관계 확인
    results = execute_cypher(conn, "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count")
    count = results[0] if results else 0
    print(f"  BELONGS_TO: {count}개", end="")
    if count >= 3:
        print(" [OK]")
    else:
        print(" [FAIL]")
        success = False

    # GENERATES_ALARM 관계 확인
    results = execute_cypher(conn, "MATCH ()-[r:GENERATES_ALARM]->() RETURN count(r) as count")
    count = results[0] if results else 0
    print(f"  GENERATES_ALARM: {count}개", end="")
    if count >= 2:
        print(" [OK]")
    else:
        print(" [FAIL]")
        success = False

    # NEXT_STEP 관계 확인
    results = execute_cypher(conn, "MATCH ()-[r:NEXT_STEP]->() RETURN count(r) as count")
    count = results[0] if results else 0
    print(f"  NEXT_STEP: {count}개", end="")
    if count >= 3:
        print(" [OK]")
    else:
        print(" [FAIL]")
        success = False

    return success


def run_sample_queries(conn):
    """샘플 쿼리 실행"""
    print("\n" + "=" * 60)
    print("5. 샘플 Cypher 쿼리 실행")
    print("=" * 60)

    # 쿼리 1: 특정 설비에서 처리 중인 Lot
    print("\n[쿼리 1] EQP-ETCH-001에서 처리 중인 Lot:")
    results = execute_cypher(conn, """
        MATCH (l:Lot)-[r:PROCESSED_AT]->(e:Equipment {equipment_id: 'EQP-ETCH-001'})
        RETURN l.lot_id as lot_id, l.status as status, r.recipe_id as recipe
    """)
    for r in results:
        print(f"  - {r}")

    # 쿼리 2: 공정 경로 탐색
    print("\n[쿼리 2] ROUTE-DRAM-001 공정 경로:")
    results = execute_cypher(conn, """
        MATCH path = (p1:Process)-[:NEXT_STEP*]->(p2:Process)
        WHERE p1.route_id = 'ROUTE-DRAM-001'
        RETURN p1.name as from_step, p2.name as to_step
    """)
    for r in results:
        print(f"  - {r}")

    # 쿼리 3: 알람 영향 분석
    print("\n[쿼리 3] 알람이 영향을 미친 Lot:")
    results = execute_cypher(conn, """
        MATCH (a:Alarm)-[r:AFFECTS_LOT]->(l:Lot)
        RETURN a.alarm_code as alarm, l.lot_id as lot, r.impact_level as impact
    """)
    for r in results:
        print(f"  - {r}")

    # 쿼리 4: 웨이퍼 → Lot → 설비 추적
    print("\n[쿼리 4] 웨이퍼에서 설비까지 추적:")
    results = execute_cypher(conn, """
        MATCH (w:Wafer)-[:BELONGS_TO]->(l:Lot)-[:PROCESSED_AT]->(e:Equipment)
        RETURN w.wafer_id as wafer, l.lot_id as lot, e.equipment_id as equipment
    """)
    for r in results:
        print(f"  - {r}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("Manufacturing Ontology 검증")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

    try:
        conn = get_connection()
        print("  [OK] 데이터베이스 연결 성공")
    except Exception as e:
        print(f"  [FAIL] 데이터베이스 연결 실패: {e}")
        sys.exit(1)

    try:
        all_passed = True

        # 검증 실행
        all_passed &= verify_vertex_labels(conn)
        all_passed &= verify_edge_labels(conn)
        all_passed &= verify_sample_data(conn)
        all_passed &= verify_relationships(conn)

        # 샘플 쿼리 실행
        run_sample_queries(conn)

        # 결과 요약
        print("\n" + "=" * 60)
        print("검증 결과")
        print("=" * 60)
        if all_passed:
            print("  [SUCCESS] 모든 검증 통과!")
        else:
            print("  [FAILED] 일부 검증 실패")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
