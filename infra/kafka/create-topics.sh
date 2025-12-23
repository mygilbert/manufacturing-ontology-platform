#!/bin/bash
# ============================================================
# Kafka 토픽 생성 스크립트
# ============================================================
# 사용법: docker exec -it ontology-kafka bash /scripts/create-topics.sh
# ============================================================

KAFKA_BOOTSTRAP=${KAFKA_BOOTSTRAP:-localhost:9092}
REPLICATION_FACTOR=${REPLICATION_FACTOR:-1}
PARTITIONS=${PARTITIONS:-3}

echo "============================================================"
echo "Kafka 토픽 생성"
echo "Bootstrap: $KAFKA_BOOTSTRAP"
echo "Replication Factor: $REPLICATION_FACTOR"
echo "Partitions: $PARTITIONS"
echo "============================================================"

# 토픽 생성 함수
create_topic() {
    local topic=$1
    local partitions=${2:-$PARTITIONS}
    local retention_ms=${3:-604800000}  # 기본 7일
    local cleanup_policy=${4:-delete}

    echo "Creating topic: $topic (partitions=$partitions, retention=${retention_ms}ms)"

    kafka-topics --bootstrap-server $KAFKA_BOOTSTRAP \
        --create \
        --if-not-exists \
        --topic $topic \
        --partitions $partitions \
        --replication-factor $REPLICATION_FACTOR \
        --config retention.ms=$retention_ms \
        --config cleanup.policy=$cleanup_policy
}

# ============================================================
# FDC 토픽
# ============================================================
echo ""
echo "[FDC Topics]"

# CDC 소스 토픽 (Debezium에서 자동 생성되지만 설정 최적화)
create_topic "fdc.FDC_PARAM_VALUE" 6 86400000      # 파라미터 값 - 1일 보관
create_topic "fdc.FDC_ALARM_HISTORY" 3 604800000   # 알람 - 7일 보관
create_topic "fdc.FDC_EQUIPMENT_MASTER" 1 -1       # 장비 마스터 - 무제한 (compact)
create_topic "fdc.FDC_RECIPE_MASTER" 1 -1          # 레시피 마스터 - 무제한 (compact)

# 처리된 토픽
create_topic "fdc.measurements.processed" 6 172800000  # 처리된 측정값 - 2일
create_topic "fdc.alarms.enriched" 3 604800000         # 보강된 알람 - 7일

# ============================================================
# SPC 토픽
# ============================================================
echo ""
echo "[SPC Topics]"

create_topic "spc.SPC_MEASUREMENT" 6 86400000      # SPC 측정값 - 1일
create_topic "spc.SPC_ALARM_LOG" 3 604800000       # SPC 알람 - 7일
create_topic "spc.SPC_PROCESS_SPEC" 1 -1           # 공정 규격 - 무제한
create_topic "spc.SPC_CONTROL_CHART" 3 172800000   # 관리도 - 2일

# 처리된 토픽
create_topic "spc.measurements.analyzed" 6 172800000   # 분석된 측정값 - 2일
create_topic "spc.violations" 3 604800000              # 규칙 위반 - 7일

# ============================================================
# MES 토픽
# ============================================================
echo ""
echo "[MES Topics]"

create_topic "mes.MES_LOT_MASTER" 3 604800000      # Lot 마스터 - 7일
create_topic "mes.MES_WAFER_MASTER" 3 604800000    # Wafer 마스터 - 7일
create_topic "mes.MES_EQUIPMENT" 1 -1              # 설비 마스터 - 무제한
create_topic "mes.MES_TRACK_IN_OUT" 6 172800000    # Track In/Out - 2일
create_topic "mes.MES_PROCESS_MASTER" 1 -1         # 공정 마스터 - 무제한
create_topic "mes.MES_ROUTE_STEP" 1 -1             # 경로 - 무제한
create_topic "mes.MES_LOT_STEP" 6 172800000        # Lot 단계 - 2일

# 처리된 토픽
create_topic "mes.lots.status" 3 604800000         # Lot 상태 변경 - 7일
create_topic "mes.tracking.events" 6 172800000     # 트래킹 이벤트 - 2일

# ============================================================
# 통합 토픽 (온톨로지 적재용)
# ============================================================
echo ""
echo "[Ontology Integration Topics]"

create_topic "ontology.objects.equipment" 3 -1     # 설비 객체 - 무제한 (compact)
create_topic "ontology.objects.lot" 3 604800000    # Lot 객체 - 7일
create_topic "ontology.objects.wafer" 6 604800000  # Wafer 객체 - 7일
create_topic "ontology.objects.process" 1 -1       # 공정 객체 - 무제한
create_topic "ontology.objects.alarm" 3 604800000  # 알람 객체 - 7일
create_topic "ontology.objects.measurement" 6 86400000  # 측정 객체 - 1일

create_topic "ontology.links.processed_at" 6 604800000   # 처리 관계 - 7일
create_topic "ontology.links.belongs_to" 3 604800000     # 소속 관계 - 7일
create_topic "ontology.links.generates_alarm" 3 604800000 # 알람 생성 관계 - 7일

# ============================================================
# 알람/알림 토픽
# ============================================================
echo ""
echo "[Alert Topics]"

create_topic "alerts.realtime" 3 86400000          # 실시간 알람 - 1일
create_topic "alerts.aggregated" 1 604800000       # 집계 알람 - 7일
create_topic "alerts.notifications" 3 172800000    # 알림 - 2일

# ============================================================
# 분석 토픽
# ============================================================
echo ""
echo "[Analytics Topics]"

create_topic "analytics.anomaly.detected" 3 604800000   # 이상 감지 결과
create_topic "analytics.prediction.results" 3 604800000  # 예측 결과
create_topic "analytics.spc.statistics" 3 172800000      # SPC 통계

# ============================================================
# 시스템 토픽
# ============================================================
echo ""
echo "[System Topics]"

create_topic "fdc.dlq" 1 604800000                 # FDC Dead Letter Queue
create_topic "spc.dlq" 1 604800000                 # SPC Dead Letter Queue
create_topic "mes.dlq" 1 604800000                 # MES Dead Letter Queue
create_topic "fdc.schema-changes" 1 -1             # FDC 스키마 변경
create_topic "spc.schema-changes" 1 -1             # SPC 스키마 변경
create_topic "mes.schema-changes" 1 -1             # MES 스키마 변경

# ============================================================
# 토픽 목록 확인
# ============================================================
echo ""
echo "============================================================"
echo "생성된 토픽 목록:"
echo "============================================================"
kafka-topics --bootstrap-server $KAFKA_BOOTSTRAP --list | sort

echo ""
echo "토픽 생성 완료!"
