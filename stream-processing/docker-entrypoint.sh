#!/bin/bash
# ============================================================
# Flink Stream Processing - Docker Entrypoint
# ============================================================

set -e

# 기본값 설정
JOB_NAME="${JOB_NAME:-all}"
WAIT_FOR_KAFKA="${WAIT_FOR_KAFKA:-true}"
KAFKA_TIMEOUT="${KAFKA_TIMEOUT:-60}"

# 로깅 함수
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Kafka 대기
wait_for_kafka() {
    if [ "$WAIT_FOR_KAFKA" = "true" ]; then
        log "Waiting for Kafka to be ready..."

        start_time=$(date +%s)
        while true; do
            if python3 -c "
from kafka import KafkaConsumer
import sys
try:
    consumer = KafkaConsumer(bootstrap_servers='${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}')
    consumer.topics()
    consumer.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
                log "Kafka is ready!"
                break
            fi

            current_time=$(date +%s)
            elapsed=$((current_time - start_time))

            if [ $elapsed -ge $KAFKA_TIMEOUT ]; then
                log "ERROR: Timeout waiting for Kafka after ${KAFKA_TIMEOUT}s"
                exit 1
            fi

            log "Kafka not ready yet, waiting... (${elapsed}s/${KAFKA_TIMEOUT}s)"
            sleep 5
        done
    fi
}

# Redis 대기
wait_for_redis() {
    log "Waiting for Redis to be ready..."

    start_time=$(date +%s)
    while true; do
        if python3 -c "
import redis
import sys
import os
try:
    r = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'redis'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        password=os.environ.get('REDIS_PASSWORD', 'redis123'),
        socket_timeout=5
    )
    r.ping()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log "Redis is ready!"
            break
        fi

        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ $elapsed -ge 30 ]; then
            log "WARNING: Redis not available after 30s, continuing anyway..."
            break
        fi

        log "Redis not ready yet, waiting... (${elapsed}s)"
        sleep 3
    done
}

# 잡 실행
run_job() {
    local job_name=$1
    local job_file=$2

    log "Starting Flink job: $job_name"

    cd /app
    python3 -u src/jobs/${job_file}
}

# 모든 잡 실행 (백그라운드)
run_all_jobs() {
    log "Starting all Flink stream processing jobs..."

    # 잡들을 순차적으로 시작 (각 잡은 자체 프로세스)
    python3 -u src/jobs/fdc_enrichment.py &
    FDC_PID=$!
    log "FDC Enrichment job started (PID: $FDC_PID)"

    sleep 5

    python3 -u src/jobs/spc_control_chart.py &
    SPC_PID=$!
    log "SPC Control Chart job started (PID: $SPC_PID)"

    sleep 5

    python3 -u src/jobs/cep_anomaly_detection.py &
    CEP_PID=$!
    log "CEP Anomaly Detection job started (PID: $CEP_PID)"

    sleep 5

    python3 -u src/jobs/window_aggregation.py &
    AGG_PID=$!
    log "Window Aggregation job started (PID: $AGG_PID)"

    log "All jobs started. Waiting for any job to exit..."

    # 모든 백그라운드 프로세스 대기
    wait -n
    exit_code=$?

    log "A job exited with code: $exit_code"

    # 나머지 프로세스 종료
    kill $FDC_PID $SPC_PID $CEP_PID $AGG_PID 2>/dev/null || true

    exit $exit_code
}

# 메인 로직
main() {
    log "=========================================="
    log "Flink Stream Processing Container"
    log "=========================================="
    log "JOB_NAME: $JOB_NAME"
    log "KAFKA_BOOTSTRAP_SERVERS: ${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}"
    log "REDIS_HOST: ${REDIS_HOST:-redis}"
    log "=========================================="

    # 의존성 대기
    wait_for_kafka
    wait_for_redis

    # 잡 실행
    case "$JOB_NAME" in
        fdc_enrichment)
            run_job "FDC Enrichment" "fdc_enrichment.py"
            ;;
        spc_control_chart)
            run_job "SPC Control Chart" "spc_control_chart.py"
            ;;
        cep_anomaly_detection)
            run_job "CEP Anomaly Detection" "cep_anomaly_detection.py"
            ;;
        window_aggregation)
            run_job "Window Aggregation" "window_aggregation.py"
            ;;
        all)
            run_all_jobs
            ;;
        *)
            log "ERROR: Unknown job name: $JOB_NAME"
            log "Available jobs: fdc_enrichment, spc_control_chart, cep_anomaly_detection, window_aggregation, all"
            exit 1
            ;;
    esac
}

# 실행
main "$@"
