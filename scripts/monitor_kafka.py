#!/usr/bin/env python3
"""
Kafka 모니터링 스크립트

토픽 상태, 컨슈머 그룹 랙, 메시지 처리량 등을 모니터링합니다.
"""
import json
import time
from datetime import datetime
from typing import Optional

from confluent_kafka import Consumer
from confluent_kafka.admin import AdminClient, ClusterMetadata

# 설정
KAFKA_BOOTSTRAP = "localhost:9092"

# 모니터링 대상 토픽
MONITORED_TOPICS = [
    # FDC
    "fdc.FDC_PARAM_VALUE",
    "fdc.measurements.processed",
    "fdc.FDC_ALARM_HISTORY",
    "fdc.alarms.enriched",
    # SPC
    "spc.SPC_MEASUREMENT",
    "spc.measurements.analyzed",
    # MES
    "mes.MES_LOT_MASTER",
    "mes.MES_WAFER_MASTER",
    "mes.MES_TRACK_IN_OUT",
    # 온톨로지
    "ontology.objects.equipment",
    "ontology.objects.lot",
    "ontology.objects.wafer",
    # 알람
    "alerts.realtime",
]

# 컨슈머 그룹
CONSUMER_GROUPS = [
    "ontology-transformer",
    "ontology-transformer-sink",
]


def get_admin_client():
    """AdminClient 생성"""
    return AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})


def get_cluster_metadata():
    """클러스터 메타데이터 조회"""
    admin = get_admin_client()
    return admin.list_topics(timeout=10)


def print_cluster_info():
    """클러스터 정보 출력"""
    print("=" * 70)
    print("Kafka 클러스터 정보")
    print("=" * 70)

    try:
        metadata = get_cluster_metadata()

        print(f"\n브로커 목록:")
        for broker_id, broker in metadata.brokers.items():
            print(f"  - ID: {broker_id}, Host: {broker.host}:{broker.port}")

        print(f"\n클러스터 ID: {metadata.cluster_id}")
        print(f"컨트롤러: Broker {metadata.controller_id}")
        print(f"토픽 수: {len(metadata.topics)}")

    except Exception as e:
        print(f"오류: {e}")


def print_topic_info():
    """토픽 정보 출력"""
    print("\n" + "=" * 70)
    print("토픽 정보")
    print("=" * 70)

    try:
        metadata = get_cluster_metadata()
        admin = get_admin_client()

        # 토픽별 오프셋 조회를 위한 Consumer
        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "monitor-temp",
        })

        print(f"\n{'토픽':<45} {'파티션':<8} {'메시지 수':<15} {'상태':<10}")
        print("-" * 80)

        for topic_name in sorted(metadata.topics.keys()):
            if topic_name.startswith("_"):  # 내부 토픽 제외
                continue

            topic = metadata.topics[topic_name]
            partition_count = len(topic.partitions)

            # 메시지 수 계산
            total_messages = 0
            try:
                for partition_id in topic.partitions:
                    low, high = consumer.get_watermark_offsets(
                        topic.partitions[partition_id],
                        timeout=5.0,
                    )
                    if high is not None and low is not None:
                        total_messages += high - low
            except Exception:
                total_messages = "N/A"

            # 모니터링 대상 여부
            status = "모니터링" if topic_name in MONITORED_TOPICS else ""

            print(f"{topic_name:<45} {partition_count:<8} {str(total_messages):<15} {status:<10}")

        consumer.close()

    except Exception as e:
        print(f"오류: {e}")


def print_consumer_lag():
    """컨슈머 그룹 랙 출력"""
    print("\n" + "=" * 70)
    print("컨슈머 그룹 랙")
    print("=" * 70)

    try:
        admin = get_admin_client()

        # 컨슈머 그룹 목록
        groups = admin.list_consumer_groups(timeout=10)

        for group in groups.valid:
            group_id = group.group_id
            if not any(g in group_id for g in CONSUMER_GROUPS):
                continue

            print(f"\n그룹: {group_id}")
            print(f"  상태: {group.state}")

            # 그룹 상세 정보
            try:
                desc = admin.describe_consumer_groups([group_id])
                for gid, future in desc.items():
                    group_desc = future.result()
                    print(f"  멤버 수: {len(group_desc.members)}")

                    for member in group_desc.members:
                        print(f"    - {member.member_id}: {member.client_id}")

            except Exception as e:
                print(f"  상세 정보 조회 실패: {e}")

            # 오프셋 정보
            try:
                offsets = admin.list_consumer_group_offsets([group_id])
                for gid, future in offsets.items():
                    offset_data = future.result()
                    print(f"  커밋된 오프셋:")
                    for tp, offset in list(offset_data.items())[:10]:  # 상위 10개만
                        print(f"    - {tp.topic}[{tp.partition}]: {offset.offset}")
            except Exception as e:
                print(f"  오프셋 조회 실패: {e}")

    except Exception as e:
        print(f"오류: {e}")


def print_connect_status():
    """Kafka Connect 상태 출력"""
    print("\n" + "=" * 70)
    print("Kafka Connect 상태")
    print("=" * 70)

    import requests

    connect_url = "http://localhost:8083"

    try:
        # 커넥터 목록
        response = requests.get(f"{connect_url}/connectors", timeout=5)
        connectors = response.json()

        print(f"\n커넥터 수: {len(connectors)}")

        for connector in connectors:
            # 커넥터 상태
            status_response = requests.get(
                f"{connect_url}/connectors/{connector}/status",
                timeout=5,
            )
            status = status_response.json()

            connector_state = status["connector"]["state"]
            tasks = status.get("tasks", [])
            task_states = [t["state"] for t in tasks]

            print(f"\n  {connector}:")
            print(f"    상태: {connector_state}")
            print(f"    태스크: {task_states}")

            # 에러가 있으면 표시
            for task in tasks:
                if task["state"] == "FAILED":
                    print(f"    에러: {task.get('trace', 'N/A')[:100]}...")

    except requests.exceptions.ConnectionError:
        print("  Kafka Connect에 연결할 수 없습니다.")
    except Exception as e:
        print(f"  오류: {e}")


def live_monitor(interval: int = 5):
    """실시간 모니터링"""
    print("실시간 모니터링 시작 (Ctrl+C로 종료)")

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": f"monitor-live-{int(time.time())}",
        "auto.offset.reset": "latest",
    })

    consumer.subscribe(MONITORED_TOPICS)

    message_counts = {}
    last_print = time.time()

    try:
        while True:
            msg = consumer.poll(0.1)

            if msg is not None and not msg.error():
                topic = msg.topic()
                message_counts[topic] = message_counts.get(topic, 0) + 1

            # 주기적 출력
            if time.time() - last_print >= interval:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 메시지 처리량 (최근 {interval}초):")
                for topic, count in sorted(message_counts.items()):
                    print(f"  {topic}: {count} msg ({count/interval:.1f} msg/s)")
                message_counts = {}
                last_print = time.time()

    except KeyboardInterrupt:
        print("\n모니터링 종료")
    finally:
        consumer.close()


def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "cluster":
            print_cluster_info()
        elif command == "topics":
            print_topic_info()
        elif command == "lag":
            print_consumer_lag()
        elif command == "connect":
            print_connect_status()
        elif command == "live":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            live_monitor(interval)
        else:
            print(f"Unknown command: {command}")
            print("사용법: python monitor_kafka.py [cluster|topics|lag|connect|live]")
    else:
        print_cluster_info()
        print_topic_info()
        print_consumer_lag()
        print_connect_status()


if __name__ == "__main__":
    main()
