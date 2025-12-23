# Manufacturing Ontology Platform

팔란티어 온톨로지 개념을 적용한 제조 데이터 실시간 분석 시스템

## 개요

FDC(Fault Detection & Classification), SPC(Statistical Process Control), MES(Manufacturing Execution System) 등 레거시 시스템의 데이터를 통합하여 **그래프 기반 온톨로지**로 모델링하고, 실시간 분석을 제공하는 플랫폼입니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Legacy Systems (FDC, SPC, MES)               │
└─────────────────────────────────────────────────────────────────┘
                              │ Debezium CDC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Apache Kafka                              │
└─────────────────────────────────────────────────────────────────┘
                              │ Apache Flink
                              ▼
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
     ▼                        ▼                        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PostgreSQL   │    │ TimescaleDB  │    │   Redis      │
│ + Apache AGE │    │ (시계열)      │    │  (캐시)      │
└──────────────┘    └──────────────┘    └──────────────┘
     │                        │                        │
     └────────────────────────┼────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI + GraphQL                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    React + D3.js Frontend                        │
└─────────────────────────────────────────────────────────────────┘
```

## 온톨로지 모델

### Object Types (정점)

| Object Type | 설명 |
|-------------|------|
| Equipment | 설비/장비 |
| Process | 공정 단계 |
| Lot | 생산 단위 |
| Wafer | 개별 웨이퍼 |
| Recipe | 공정 레시피 |
| Measurement | 측정값 |
| Alarm | 알람/이벤트 |

### Link Types (간선)

| Link Type | 관계 |
|-----------|------|
| PROCESSED_AT | Lot → Equipment |
| BELONGS_TO | Wafer → Lot |
| USES_RECIPE | Process → Recipe |
| GENERATES_ALARM | Equipment → Alarm |
| NEXT_STEP | Process → Process |
| MEASURED_BY | Wafer → Measurement |
| AFFECTS_LOT | Alarm → Lot |

## 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# 필요한 경우 .env 파일 수정
```

### 2. 서비스 시작

```bash
# 전체 서비스 시작
docker-compose up -d

# 또는 단계별로 시작
docker-compose up -d postgres timescaledb redis  # DB만
docker-compose up -d kafka zookeeper             # Kafka
docker-compose up -d api frontend                # 앱
```

### 3. 온톨로지 스키마 및 샘플 데이터 적용

```bash
# PostgreSQL 컨테이너에서 마이그레이션 실행
docker exec -it ontology-postgres psql -U ontology -d manufacturing \
  -f /docker-entrypoint-initdb.d/01-init.sql

# 샘플 데이터 적용
docker cp ontology/migrations/002_sample_data.sql ontology-postgres:/tmp/
docker exec -it ontology-postgres psql -U ontology -d manufacturing \
  -f /tmp/002_sample_data.sql
```

### 4. 검증

```bash
# Python 검증 스크립트 실행
pip install -r scripts/requirements.txt
python scripts/verify_ontology.py
```

## Cypher 쿼리 예시

```sql
-- AGE 로드
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- 설비별 처리 중인 Lot 조회
SELECT * FROM cypher('manufacturing', $$
  MATCH (l:Lot)-[r:PROCESSED_AT]->(e:Equipment)
  WHERE e.status = 'RUNNING'
  RETURN e.equipment_id, l.lot_id, r.recipe_id
$$) as (equipment agtype, lot agtype, recipe agtype);

-- 공정 경로 탐색
SELECT * FROM cypher('manufacturing', $$
  MATCH path = (p1:Process {process_id: 'PROC-LITHO-001'})-[:NEXT_STEP*]->(p2:Process)
  RETURN [n in nodes(path) | n.name] as route
$$) as (route agtype);

-- 알람 영향 분석
SELECT * FROM cypher('manufacturing', $$
  MATCH (e:Equipment)-[:GENERATES_ALARM]->(a:Alarm)-[:AFFECTS_LOT]->(l:Lot)
  WHERE a.severity = 'MAJOR'
  RETURN e.name, a.alarm_code, l.lot_id
$$) as (equipment agtype, alarm agtype, lot agtype);
```

## 디렉토리 구조

```
manufacturing-ontology-platform/
├── docker-compose.yml          # 서비스 오케스트레이션
├── .env.example                # 환경 변수 템플릿
│
├── ontology/                   # 온톨로지 정의
│   ├── schemas/
│   │   ├── objects/           # Object Type YAML
│   │   └── links/             # Link Type YAML
│   └── migrations/            # SQL 마이그레이션
│
├── infra/                      # 인프라 설정
│   ├── postgres/              # PostgreSQL + AGE
│   └── timescaledb/           # TimescaleDB
│
├── api/                        # FastAPI 서버
├── frontend/                   # React 프론트엔드
├── stream-processing/          # Flink 잡
├── analytics/                  # 분석 엔진
└── scripts/                    # 유틸리티 스크립트
```

## 기술 스택

- **그래프 DB**: PostgreSQL + Apache AGE
- **시계열 DB**: TimescaleDB
- **메시지 브로커**: Apache Kafka
- **CDC**: Debezium
- **스트림 처리**: Apache Flink
- **API**: FastAPI + GraphQL (Strawberry)
- **프론트엔드**: React + D3.js
- **캐시**: Redis

## 라이선스

MIT License
