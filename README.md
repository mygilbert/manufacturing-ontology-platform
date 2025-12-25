# Manufacturing Ontology Platform

팔란티어 온톨로지 개념을 적용한 제조 데이터 실시간 분석 시스템

## 개요

FDC(Fault Detection & Classification), SPC(Statistical Process Control), MES(Manufacturing Execution System) 등 레거시 시스템의 데이터를 통합하여 **그래프 기반 온톨로지**로 모델링하고, 실시간 분석을 제공하는 플랫폼입니다.

### 주요 기능

- **온톨로지 기반 데이터 모델링**: 설비, 공정, 품질 데이터를 그래프 구조로 연결
- **암묵적 관계 발견**: 상관분석, 인과성 분석으로 숨겨진 관계 자동 발견
- **실시간 이상 감지**: 앙상블 알고리즘 기반 이상 탐지 및 경보
- **도메인 지식 통합**: 엔지니어 경험을 구조화하여 AI Agent에 반영

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
│              + Relationship Discovery Engine                     │
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

### Link Types (간선) - 명시적 관계

| Link Type | 관계 |
|-----------|------|
| PROCESSED_AT | Lot → Equipment |
| BELONGS_TO | Wafer → Lot |
| USES_RECIPE | Process → Recipe |
| GENERATES_ALARM | Equipment → Alarm |
| NEXT_STEP | Process → Process |
| MEASURED_BY | Wafer → Measurement |
| AFFECTS_LOT | Alarm → Lot |

### Link Types (간선) - 암묵적 관계 (자동 발견)

| Link Type | 발견 방법 | 설명 |
|-----------|----------|------|
| CORRELATES_WITH | 상관분석 | 파라미터 간 통계적 상관관계 |
| INFLUENCES | 인과분석 | Granger Causality 기반 인과관계 |
| PRECEDES | 패턴마이닝 | 시간적 선후관계 |
| CO_OCCURS | 연관규칙 | 동시 발생 이벤트 패턴 |
| ROOT_CAUSE_OF | 인과분석 | 이상 발생의 근본 원인 |

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
│   └── src/
│       ├── routers/           # REST API 엔드포인트
│       ├── services/          # 비즈니스 로직
│       └── graphql/           # GraphQL 스키마
│
├── frontend/                   # React 프론트엔드
│
├── stream-processing/          # Flink 스트림 처리
│   └── src/jobs/
│       ├── fdc_enrichment.py  # FDC 데이터 보강
│       ├── spc_control_chart.py # SPC 관리도
│       ├── cep_anomaly_detection.py # 복합 이벤트 처리
│       └── window_aggregation.py # 윈도우 집계
│
├── ingestion/                  # 데이터 수집 (Debezium CDC)
│   └── transformers/          # 데이터 변환기
│
├── analytics/                  # 분석 엔진 (핵심 모듈)
│   ├── src/
│   │   ├── anomaly_detection/ # 이상 탐지 알고리즘
│   │   ├── spc/              # SPC 분석
│   │   ├── prediction/       # 예측 모델
│   │   └── relationship_discovery/  # 관계 발견 엔진 ★
│   ├── realtime_alert_system/ # 실시간 경보 시스템
│   ├── templates/             # Excel 템플릿
│   ├── sample_data/           # 테스트 데이터
│   └── results/               # 분석 결과
│
├── scripts/                    # 유틸리티 스크립트
└── docs/                       # 상세 문서
```

## 핵심 모듈 설명

### 1. 관계 발견 엔진 (Relationship Discovery)

데이터에서 숨겨진 관계를 자동으로 발견하는 핵심 모듈입니다.

**위치**: `analytics/src/relationship_discovery/`

| 파일 | 기능 |
|------|------|
| `correlation_analyzer.py` | Pearson/Spearman 상관분석, Cross-correlation |
| `causality_analyzer.py` | Granger Causality, Transfer Entropy |
| `pattern_detector.py` | Sequential Pattern Mining, Association Rules |
| `relationship_store.py` | 발견된 관계 → 온톨로지 저장 |
| `discovery_pipeline.py` | 통합 파이프라인, 리포트 생성 |
| `expert_knowledge_loader.py` | 도메인 지식 로더 |

**사용법**:
```python
from relationship_discovery import DiscoveryPipeline, DiscoveryConfig

config = DiscoveryConfig()
pipeline = DiscoveryPipeline(config)

# 관계 발견 실행
relationships = pipeline.discover_all(
    pv_data=df,           # 공정 변수 데이터
    event_data=events,    # 이벤트 데이터
    pv_columns=['temp', 'pressure', 'vibration']
)

# 리포트 생성
pipeline.export_results('report.html', format='html')
```

### 2. 실시간 경보 시스템 (Realtime Alert)

앙상블 이상 탐지 기반 실시간 경보 시스템입니다.

**위치**: `analytics/realtime_alert_system/`

**기능**:
- Z-Score, CUSUM, Isolation Forest, LOF 앙상블
- WebSocket 기반 실시간 대시보드
- 경보 이력 관리

**실행**:
```bash
cd analytics
python scripts/run_realtime_alert.py
# 브라우저: http://localhost:8000
```

### 3. 도메인 지식 템플릿

엔지니어의 경험 지식을 구조화하여 입력받는 Excel 템플릿입니다.

**위치**: `analytics/templates/`

| 템플릿 | 용도 |
|--------|------|
| `expert_knowledge_template.xlsx` | 일반 도메인 지식 (인과관계, 알람원인 등) |
| `equipment_control_relationship_template.xlsx` | 설비 제어 관계 (Setpoint-PV-품질) |

## 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# Python 가상환경
cd analytics
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 관계 발견 테스트 실행

```bash
# 샘플 데이터 생성
cd analytics
python scripts/generate_sample_data.py

# 관계 발견 테스트
python scripts/test_relationship_discovery.py

# 실제 데이터로 테스트
python scripts/test_real_data_discovery.py
```

### 3. 전체 서비스 시작 (Docker)

```bash
# 전체 서비스 시작
docker-compose up -d

# 단계별 시작
docker-compose up -d postgres timescaledb redis  # DB
docker-compose up -d kafka zookeeper             # Kafka
docker-compose up -d api frontend                # App
```

## 기술 스택

- **그래프 DB**: PostgreSQL + Apache AGE
- **시계열 DB**: TimescaleDB
- **메시지 브로커**: Apache Kafka
- **CDC**: Debezium
- **스트림 처리**: Apache Flink (PyFlink)
- **분석 엔진**: Python (NumPy, SciPy, scikit-learn)
- **API**: FastAPI + GraphQL (Strawberry)
- **프론트엔드**: React + D3.js
- **캐시**: Redis

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

-- 인과관계 경로 탐색 (발견된 관계 활용)
SELECT * FROM cypher('manufacturing', $$
  MATCH path = (p1:Parameter)-[:INFLUENCES*1..3]->(p2:Parameter)
  WHERE p1.name = 'PRESSURE' AND p2.name = 'ETCH_RATE'
  RETURN path, [r in relationships(path) | r.lag] as lags
$$) as (path agtype, lags agtype);

-- 알람 근본원인 분석
SELECT * FROM cypher('manufacturing', $$
  MATCH (p:Parameter)-[:ROOT_CAUSE_OF]->(a:Alarm)
  WHERE a.severity = 'CRITICAL'
  RETURN p.name as root_cause, a.alarm_code, a.description
  ORDER BY a.timestamp DESC
$$) as (root_cause agtype, alarm agtype, description agtype);
```

## 향후 계획

1. **Phase 1 (PoC)**: 샘플 데이터로 관계 발견 검증 ✅
2. **Phase 2**: DataWarehouse 연동, 실제 데이터 분석
3. **Phase 3**: AI Agent 통합, 자동 근본원인 분석
4. **Phase 4**: Production 배포, 실시간 모니터링

## 라이선스

MIT License
