#!/usr/bin/env python3
"""
CDC 파이프라인 테스트 스크립트

Kafka에 테스트 메시지를 발행하고 변환/적재 결과를 확인합니다.
"""
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic
import psycopg2

# 설정
KAFKA_BOOTSTRAP = "localhost:9092"
POSTGRES_DSN = "postgresql://ontology:ontology123@localhost:5432/manufacturing"
TIMESCALE_DSN = "postgresql://timescale:timescale123@localhost:5433/measurements"


def get_producer():
    """Kafka Producer 생성"""
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "acks": "all",
    })


def get_consumer(topics: list[str], group_id: str = "test-consumer"):
    """Kafka Consumer 생성"""
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe(topics)
    return consumer


def produce_fdc_measurement():
    """FDC 측정 데이터 테스트 메시지 발행"""
    producer = get_producer()

    measurement = {
        "VALUE_ID": f"FDC-{uuid.uuid4().hex[:8]}",
        "EQUIP_ID": "EQP-ETCH-001",
        "CHAMBER_ID": "CH1",
        "RECIPE_ID": "RCP-ETCH-OXIDE-V2.1",
        "LOT_ID": "LOT20231201001",
        "WAFER_ID": "LOT20231201001-01",
        "SLOT_NO": 1,
        "PARAM_ID": "CHAMBER_PRESSURE",
        "PARAM_NAME": "Chamber Pressure",
        "PARAM_VALUE": 98.5,
        "PARAM_UNIT": "mTorr",
        "COLLECT_TIME": datetime.utcnow().isoformat(),
        "USL": 100.0,
        "LSL": 90.0,
        "TARGET": 95.0,
        "source_system": "FDC",
    }

    producer.produce(
        topic="fdc.FDC_PARAM_VALUE",
        key=measurement["VALUE_ID"],
        value=json.dumps(measurement),
    )
    producer.flush()

    print(f"Produced FDC measurement: {measurement['VALUE_ID']}")
    return measurement


def produce_spc_measurement():
    """SPC 측정 데이터 테스트 메시지 발행"""
    producer = get_producer()

    measurement = {
        "MEAS_ID": f"SPC-{uuid.uuid4().hex[:8]}",
        "EQUIP_ID": "EQP-CVD-001",
        "PROCESS_ID": "PROC-CVD-001",
        "LOT_ID": "LOT20231201003",
        "WAFER_ID": "LOT20231201003-01",
        "ITEM_ID": "FILM_THICKNESS",
        "ITEM_NAME": "Film Thickness",
        "MEAS_VALUE": 1485.0,
        "UNIT": "A",
        "MEAS_TIME": datetime.utcnow().isoformat(),
        "USL": 1550.0,
        "LSL": 1450.0,
        "UCL": 1520.0,
        "LCL": 1480.0,
        "TARGET": 1500.0,
        "source_system": "SPC",
    }

    producer.produce(
        topic="spc.SPC_MEASUREMENT",
        key=measurement["MEAS_ID"],
        value=json.dumps(measurement),
    )
    producer.flush()

    print(f"Produced SPC measurement: {measurement['MEAS_ID']}")
    return measurement


def produce_alarm():
    """알람 테스트 메시지 발행"""
    producer = get_producer()

    alarm = {
        "ALARM_ID": f"ALM-{uuid.uuid4().hex[:8]}",
        "ALARM_CODE": "FDC-TEMP-HIGH",
        "ALARM_NAME": "Temperature High",
        "ALARM_LEVEL": "MAJOR",
        "EQUIP_ID": "EQP-ETCH-001",
        "CHAMBER_ID": "CH1",
        "LOT_ID": "LOT20231201001",
        "ALARM_TIME": datetime.utcnow().isoformat(),
        "ALARM_MSG": "Chamber temperature exceeded UCL: 350C > 340C",
        "ALARM_VALUE": 350.0,
        "THRESHOLD": 340.0,
        "source_system": "FDC",
    }

    producer.produce(
        topic="fdc.FDC_ALARM_HISTORY",
        key=alarm["ALARM_ID"],
        value=json.dumps(alarm),
    )
    producer.flush()

    print(f"Produced alarm: {alarm['ALARM_ID']}")
    return alarm


def consume_processed_messages(topic: str, timeout: int = 10):
    """처리된 메시지 확인"""
    consumer = get_consumer([topic])
    messages = []
    start_time = time.time()

    print(f"Consuming from {topic}...")

    while time.time() - start_time < timeout:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        value = json.loads(msg.value().decode("utf-8"))
        messages.append(value)
        print(f"Received: {json.dumps(value, indent=2)}")

    consumer.close()
    return messages


def verify_timescaledb():
    """TimescaleDB 데이터 확인"""
    print("\n" + "=" * 60)
    print("TimescaleDB 데이터 확인")
    print("=" * 60)

    try:
        conn = psycopg2.connect(TIMESCALE_DSN)
        cur = conn.cursor()

        # FDC 측정값
        cur.execute("""
            SELECT time, measurement_id, equipment_id, param_id, value, status
            FROM fdc_measurements
            ORDER BY time DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("\nFDC Measurements (최근 5개):")
        for row in rows:
            print(f"  {row}")

        # SPC 측정값
        cur.execute("""
            SELECT time, measurement_id, equipment_id, item_id, value, status, rule_violations
            FROM spc_measurements
            ORDER BY time DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("\nSPC Measurements (최근 5개):")
        for row in rows:
            print(f"  {row}")

        # 알람
        cur.execute("""
            SELECT time, alarm_id, alarm_code, severity, equipment_id, status
            FROM alarm_history
            ORDER BY time DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("\nAlarm History (최근 5개):")
        for row in rows:
            print(f"  {row}")

        conn.close()
    except Exception as e:
        print(f"TimescaleDB 연결 오류: {e}")


def verify_age():
    """Apache AGE 그래프 데이터 확인"""
    print("\n" + "=" * 60)
    print("Apache AGE 그래프 데이터 확인")
    print("=" * 60)

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()

        # AGE 로드
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        # Equipment 수
        cur.execute("""
            SELECT * FROM cypher('manufacturing', $$
                MATCH (n:Equipment) RETURN count(n) as count
            $$) AS (count agtype)
        """)
        print(f"\nEquipment 수: {cur.fetchone()[0]}")

        # Lot 수
        cur.execute("""
            SELECT * FROM cypher('manufacturing', $$
                MATCH (n:Lot) RETURN count(n) as count
            $$) AS (count agtype)
        """)
        print(f"Lot 수: {cur.fetchone()[0]}")

        # Alarm 수
        cur.execute("""
            SELECT * FROM cypher('manufacturing', $$
                MATCH (n:Alarm) RETURN count(n) as count
            $$) AS (count agtype)
        """)
        print(f"Alarm 수: {cur.fetchone()[0]}")

        # 최근 알람
        cur.execute("""
            SELECT * FROM cypher('manufacturing', $$
                MATCH (a:Alarm)
                RETURN a.alarm_id, a.alarm_code, a.severity
                ORDER BY a.occurred_at DESC
                LIMIT 5
            $$) AS (alarm_id agtype, code agtype, severity agtype)
        """)
        rows = cur.fetchall()
        print("\n최근 알람:")
        for row in rows:
            print(f"  {row}")

        conn.close()
    except Exception as e:
        print(f"AGE 연결 오류: {e}")


def run_full_test():
    """전체 테스트 실행"""
    print("=" * 60)
    print("CDC 파이프라인 통합 테스트")
    print("=" * 60)

    # 1. 테스트 데이터 발행
    print("\n[1] 테스트 데이터 발행")
    produce_fdc_measurement()
    produce_fdc_measurement()
    produce_spc_measurement()
    produce_alarm()

    # 2. 처리 대기
    print("\n[2] 변환 처리 대기 (5초)...")
    time.sleep(5)

    # 3. 처리된 메시지 확인
    print("\n[3] 처리된 메시지 확인")
    # 실제로는 transformer가 실행 중이어야 함

    # 4. DB 확인
    print("\n[4] 데이터베이스 확인")
    verify_timescaledb()
    verify_age()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fdc":
            produce_fdc_measurement()
        elif command == "spc":
            produce_spc_measurement()
        elif command == "alarm":
            produce_alarm()
        elif command == "verify":
            verify_timescaledb()
            verify_age()
        else:
            print(f"Unknown command: {command}")
    else:
        run_full_test()
