#!/usr/bin/env python3
"""
Flink 스트림 처리 잡 테스트 스크립트

사용법:
    python test_flink_jobs.py --job fdc     # FDC 데이터 테스트
    python test_flink_jobs.py --job spc     # SPC 데이터 테스트
    python test_flink_jobs.py --job all     # 모든 잡 테스트
"""
import argparse
import json
import random
import time
import uuid
from datetime import datetime
from typing import Dict, Any

from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic


# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# 토픽
TOPICS = {
    "fdc_measurement": "fdc.FDC_PARAM_VALUE",
    "fdc_alarm": "fdc.FDC_ALARM_HISTORY",
    "fdc_equipment": "fdc.FDC_EQUIPMENT_MASTER",
    "spc_measurement": "spc.SPC_MEASUREMENT",
    "enriched_measurement": "flink.measurements.enriched",
    "spc_analyzed": "flink.spc.analyzed",
    "cep_alerts": "flink.cep.alerts",
    "aggregated_stats": "flink.stats.aggregated",
}


def create_producer() -> KafkaProducer:
    """Kafka 프로듀서 생성"""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def create_consumer(topics: list, group_id: str = "test-consumer") -> KafkaConsumer:
    """Kafka 컨슈머 생성"""
    return KafkaConsumer(
        *topics,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id=group_id,
        consumer_timeout_ms=10000,
    )


def generate_fdc_measurement(
    equipment_id: str = "EQP001",
    param_id: str = "TEMP001",
    value: float = None,
    status: str = "NORMAL",
) -> Dict[str, Any]:
    """FDC 측정 데이터 생성"""
    if value is None:
        value = random.uniform(100, 200)

    return {
        "VALUE_ID": f"FDC-{uuid.uuid4().hex[:8]}",
        "EQUIP_ID": equipment_id,
        "PARAM_ID": param_id,
        "PARAM_NAME": "Temperature",
        "PARAM_VALUE": value,
        "PARAM_UNIT": "degC",
        "COLLECT_TIME": datetime.utcnow().isoformat(),
        "CHAMBER_ID": "CH01",
        "RECIPE_ID": "RCP001",
        "LOT_ID": "LOT20231201001",
        "WAFER_ID": "WAFER001",
        "USL": 200.0,
        "LSL": 100.0,
        "TARGET": 150.0,
    }


def generate_spc_measurement(
    equipment_id: str = "EQP001",
    item_id: str = "THICK001",
    value: float = None,
) -> Dict[str, Any]:
    """SPC 측정 데이터 생성"""
    if value is None:
        value = random.gauss(100, 5)

    return {
        "MEAS_ID": f"SPC-{uuid.uuid4().hex[:8]}",
        "EQUIP_ID": equipment_id,
        "PROCESS_ID": "ETCH001",
        "ITEM_ID": item_id,
        "ITEM_NAME": "Thickness",
        "MEAS_VALUE": value,
        "MEAS_TIME": datetime.utcnow().isoformat(),
        "UNIT": "nm",
        "LOT_ID": "LOT20231201001",
        "WAFER_ID": "WAFER001",
        "USL": 120.0,
        "LSL": 80.0,
        "UCL": 115.0,
        "LCL": 85.0,
        "TARGET": 100.0,
        "SUBGROUP_ID": "SG001",
        "SUBGROUP_SIZE": 5,
    }


def generate_fdc_alarm(
    equipment_id: str = "EQP001",
    alarm_code: str = "ALM001",
) -> Dict[str, Any]:
    """FDC 알람 데이터 생성"""
    return {
        "ALARM_ID": f"ALM-{uuid.uuid4().hex[:8]}",
        "EQUIP_ID": equipment_id,
        "ALARM_CODE": alarm_code,
        "ALARM_LEVEL": random.choice(["WARNING", "MAJOR", "CRITICAL"]),
        "ALARM_MSG": f"Test alarm from {equipment_id}",
        "ALARM_TIME": datetime.utcnow().isoformat(),
        "LOT_ID": "LOT20231201001",
    }


def test_fdc_enrichment(producer: KafkaProducer, count: int = 10):
    """FDC 보강 잡 테스트"""
    print("\n=== Testing FDC Enrichment Job ===")
    print(f"Sending {count} FDC measurements...")

    for i in range(count):
        # 정상/경고/알람 상태 혼합
        if i % 5 == 0:
            value = 205.0  # USL 초과
        elif i % 3 == 0:
            value = 95.0  # 경고 범위
        else:
            value = random.uniform(110, 190)  # 정상

        msg = generate_fdc_measurement(value=value)
        producer.send(TOPICS["fdc_measurement"], key=msg["EQUIP_ID"], value=msg)
        print(f"  Sent: VALUE_ID={msg['VALUE_ID']}, VALUE={msg['PARAM_VALUE']:.2f}")
        time.sleep(0.5)

    producer.flush()
    print("Done!")


def test_spc_control_chart(producer: KafkaProducer, count: int = 30):
    """SPC 관리도 잡 테스트"""
    print("\n=== Testing SPC Control Chart Job ===")
    print(f"Sending {count} SPC measurements...")

    # 다양한 패턴 생성
    patterns = {
        "normal": lambda: random.gauss(100, 3),
        "trend_up": lambda i: 100 + i * 0.5 + random.gauss(0, 1),
        "run_above": lambda: random.gauss(108, 2),
        "ooc": lambda: random.choice([75, 125]),
    }

    for i in range(count):
        # 패턴 선택
        if i < 10:
            value = patterns["normal"]()
        elif i < 20:
            value = patterns["trend_up"](i - 10)
        elif i < 25:
            value = patterns["run_above"]()
        else:
            value = patterns["ooc"]()

        msg = generate_spc_measurement(value=value)
        producer.send(TOPICS["spc_measurement"], key=msg["EQUIP_ID"], value=msg)
        print(f"  Sent: MEAS_ID={msg['MEAS_ID']}, VALUE={msg['MEAS_VALUE']:.2f}")
        time.sleep(0.3)

    producer.flush()
    print("Done!")


def test_cep_detection(producer: KafkaProducer):
    """CEP 이상 패턴 감지 테스트"""
    print("\n=== Testing CEP Anomaly Detection Job ===")

    # 테스트 1: 연속 임계값 초과
    print("\n[Test 1] Consecutive threshold violations (3 alarms in 3 min)")
    for i in range(4):
        msg = generate_fdc_measurement(value=210.0)  # USL 초과
        producer.send(TOPICS["fdc_measurement"], key=msg["EQUIP_ID"], value=msg)
        print(f"  Sent alarm {i+1}: VALUE={msg['PARAM_VALUE']:.2f}")
        time.sleep(1)

    # 테스트 2: 연속 알람
    print("\n[Test 2] Multiple alarms from same equipment")
    for i in range(6):
        msg = generate_fdc_alarm(alarm_code=f"ALM00{i+1}")
        producer.send(TOPICS["fdc_alarm"], key=msg["EQUIP_ID"], value=msg)
        print(f"  Sent alarm: CODE={msg['ALARM_CODE']}")
        time.sleep(0.5)

    # 테스트 3: 드리프트
    print("\n[Test 3] Value drift pattern")
    base_value = 150.0
    for i in range(15):
        # 점진적 상승 트렌드
        value = base_value + i * 2 + random.gauss(0, 1)
        msg = generate_fdc_measurement(value=value)
        producer.send(TOPICS["fdc_measurement"], key=msg["EQUIP_ID"], value=msg)
        print(f"  Sent: VALUE={msg['PARAM_VALUE']:.2f}")
        time.sleep(0.3)

    producer.flush()
    print("Done!")


def consume_results(topics: list, timeout: int = 30):
    """결과 토픽 소비"""
    print(f"\n=== Consuming results from {topics} (timeout: {timeout}s) ===")

    consumer = create_consumer(topics, group_id=f"test-{uuid.uuid4().hex[:6]}")

    start_time = time.time()
    count = 0

    try:
        while time.time() - start_time < timeout:
            for message in consumer:
                count += 1
                topic = message.topic
                value = message.value

                # 토픽별 출력 포맷
                if topic == TOPICS["enriched_measurement"]:
                    print(f"\n[ENRICHED] ID={value.get('measurement_id')}, "
                          f"STATUS={value.get('status')}, "
                          f"EQUIP={value.get('equipment_name', 'N/A')}")
                elif topic == TOPICS["spc_analyzed"]:
                    print(f"\n[SPC] ID={value.get('measurement_id')}, "
                          f"STATUS={value.get('status')}, "
                          f"RULES={value.get('rule_violations', [])}")
                elif topic == TOPICS["cep_alerts"]:
                    print(f"\n[CEP ALERT] ID={value.get('alert_id')}, "
                          f"PATTERN={value.get('pattern_id')}, "
                          f"SEVERITY={value.get('severity')}")
                    print(f"  MESSAGE: {value.get('message')}")
                elif topic == TOPICS["aggregated_stats"]:
                    print(f"\n[STATS] EQUIP={value.get('equipment_id')}, "
                          f"COUNT={value.get('count')}, "
                          f"AVG={value.get('avg_value', 0):.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    print(f"\nTotal messages received: {count}")


def main():
    parser = argparse.ArgumentParser(description="Flink Stream Processing Job Tester")
    parser.add_argument(
        "--job",
        choices=["fdc", "spc", "cep", "all"],
        default="all",
        help="Job to test (default: all)",
    )
    parser.add_argument(
        "--consume",
        action="store_true",
        help="Also consume and display results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Consumer timeout in seconds (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Flink Stream Processing Job Tester")
    print("=" * 60)

    producer = create_producer()

    try:
        if args.job == "fdc" or args.job == "all":
            test_fdc_enrichment(producer)

        if args.job == "spc" or args.job == "all":
            test_spc_control_chart(producer)

        if args.job == "cep" or args.job == "all":
            test_cep_detection(producer)

        if args.consume:
            result_topics = [
                TOPICS["enriched_measurement"],
                TOPICS["spc_analyzed"],
                TOPICS["cep_alerts"],
                TOPICS["aggregated_stats"],
            ]
            consume_results(result_topics, args.timeout)

    finally:
        producer.close()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
