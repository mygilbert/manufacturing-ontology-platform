#!/bin/bash
# ============================================================
# Debezium CDC 커넥터 배포 스크립트
# ============================================================
# 사용법: ./scripts/deploy_connectors.sh [connector_name]
# 예시:
#   ./scripts/deploy_connectors.sh                  # 모든 커넥터 배포
#   ./scripts/deploy_connectors.sh fdc-oracle       # FDC Oracle 커넥터만 배포
# ============================================================

set -e

CONNECT_URL=${CONNECT_URL:-http://localhost:8083}
CONNECTORS_DIR=${CONNECTORS_DIR:-./ingestion/debezium-connectors}

echo "============================================================"
echo "Debezium CDC 커넥터 배포"
echo "Connect URL: $CONNECT_URL"
echo "Connectors Dir: $CONNECTORS_DIR"
echo "============================================================"

# Kafka Connect 상태 확인
check_connect() {
    echo "Checking Kafka Connect status..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$CONNECT_URL/" > /dev/null 2>&1; then
            echo "Kafka Connect is ready!"
            return 0
        fi
        echo "Waiting for Kafka Connect... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "ERROR: Kafka Connect is not available"
    exit 1
}

# 커넥터 배포 함수
deploy_connector() {
    local connector_file=$1
    local connector_name=$(basename "$connector_file" .json)

    echo ""
    echo "----------------------------------------"
    echo "Deploying connector: $connector_name"
    echo "----------------------------------------"

    # 환경 변수 치환
    local config=$(envsubst < "$connector_file")

    # 기존 커넥터 확인
    local exists=$(curl -s -o /dev/null -w "%{http_code}" "$CONNECT_URL/connectors/$connector_name")

    if [ "$exists" = "200" ]; then
        echo "Connector exists, updating..."
        # 커넥터 설정 업데이트
        local update_config=$(echo "$config" | jq '.config')
        curl -s -X PUT \
            -H "Content-Type: application/json" \
            -d "$update_config" \
            "$CONNECT_URL/connectors/$connector_name/config" | jq .
    else
        echo "Creating new connector..."
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$config" \
            "$CONNECT_URL/connectors" | jq .
    fi

    # 상태 확인
    sleep 2
    echo ""
    echo "Connector status:"
    curl -s "$CONNECT_URL/connectors/$connector_name/status" | jq .
}

# 커넥터 삭제 함수
delete_connector() {
    local connector_name=$1

    echo "Deleting connector: $connector_name"
    curl -s -X DELETE "$CONNECT_URL/connectors/$connector_name"
    echo "Deleted."
}

# 모든 커넥터 상태 확인
list_connectors() {
    echo ""
    echo "============================================================"
    echo "현재 배포된 커넥터 목록:"
    echo "============================================================"
    curl -s "$CONNECT_URL/connectors" | jq .

    echo ""
    echo "각 커넥터 상태:"
    for connector in $(curl -s "$CONNECT_URL/connectors" | jq -r '.[]'); do
        echo ""
        echo "--- $connector ---"
        curl -s "$CONNECT_URL/connectors/$connector/status" | jq '{name: .name, state: .connector.state, tasks: [.tasks[].state]}'
    done
}

# 메인 로직
main() {
    check_connect

    local target_connector=$1

    if [ "$target_connector" = "list" ]; then
        list_connectors
        exit 0
    fi

    if [ "$target_connector" = "delete" ]; then
        delete_connector "$2"
        exit 0
    fi

    if [ -n "$target_connector" ]; then
        # 특정 커넥터만 배포
        local connector_file="$CONNECTORS_DIR/${target_connector}-connector.json"
        if [ -f "$connector_file" ]; then
            deploy_connector "$connector_file"
        else
            echo "ERROR: Connector file not found: $connector_file"
            exit 1
        fi
    else
        # 모든 커넥터 배포
        for connector_file in "$CONNECTORS_DIR"/*-connector.json; do
            if [ -f "$connector_file" ]; then
                deploy_connector "$connector_file"
            fi
        done
    fi

    echo ""
    echo "============================================================"
    echo "배포 완료!"
    echo "============================================================"
    list_connectors
}

main "$@"
