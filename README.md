# Manufacturing Ontology Platform

팔란티어 온톨로지 개념을 적용한 제조 데이터 실시간 분석 시스템

## 개요

FDC(Fault Detection & Classification), SPC(Statistical Process Control), MES(Manufacturing Execution System) 등 레거시 시스템의 데이터를 통합하여 **그래프 기반 온톨로지**로 모델링하고, **AI Agent 기반 분석**을 제공하는 플랫폼입니다.

### 주요 기능

- **온톨로지 기반 데이터 모델링**: 설비, 공정, 품질 데이터를 그래프 구조로 연결
- **배터리 제조 계층 구조**: Roll → Cell → Module → Pack 추적성 지원
- **AI Agent 분석**: EXAONE 3.5 LLM 기반 자연어 질의응답
- **암묵적 관계 발견**: 상관분석, 인과성 분석으로 숨겨진 관계 자동 발견
- **실시간 이상 감지**: 앙상블 알고리즘 기반 이상 탐지 및 경보
- **도메인 지식 통합**: 배터리 제조 인과관계를 구조화하여 AI Agent에 반영

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Legacy Systems (FDC, MES, ERP)               │
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
│           + FDC Analysis Agent (EXAONE 3.5)                     │
│              + Relationship Discovery Engine                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              React + D3.js Frontend + Agent Chat                │
└─────────────────────────────────────────────────────────────────┘
```

## AI Agent (FDC Analysis)

EXAONE 3.5 (LG AI Research) 기반 배터리 제조 전문가 Agent입니다.

### 기능

| 기능 | 설명 |
|------|------|
| **자연어 질의** | "Cell 용량 불량 원인은?" 형태로 질문 |
| **도구 호출** | 온톨로지 검색, 시계열 분석, 알람 이력 조회 |
| **근본원인 분석** | 인과관계 기반 원인 추적 |
| **점검 순서 제안** | 도메인 지식 기반 점검 가이드 |

### 배터리 도메인 지식

```
## 공정별 인과관계

전극 공정 (Roll)
- COATING_THICKNESS → CELL_CAPACITY: 코팅 두께 변동 → 셀 용량 편차
- DRYING_TEMP → ELECTRODE_RESISTANCE: 건조 온도 이상 → 전극 저항 증가

화성/에이징 공정
- FORMATION_TEMP → SEI_QUALITY: 화성 온도 이상 → SEI 품질 저하
- FORMATION_CURRENT → CAPACITY_LOSS: 화성 전류 과다 → 용량 손실

모듈/팩 공정
- CELL_VOLTAGE_DEVIATION → MODULE_IMBALANCE: 셀 전압 편차 → 모듈 불균형
- COOLANT_FLOW → THERMAL_RUNAWAY_RISK: 냉각수 유량 부족 → 열폭주 위험
```

### Agent API

```bash
# 자연어 분석
POST /api/agent/analyze
{
  "query": "ETCH-001 온도 알람 발생. 원인은?"
}

# 알람 분석
POST /api/agent/analyze/alarm
{
  "equipment_id": "ETCH-001",
  "alarm_code": "ALM_HIGH_TEMP"
}

# 프롬프트 조회/수정
GET  /api/agent/prompt
PUT  /api/agent/prompt
```

## 준거 표준

주 도메인은 **배터리 셀 제조**다. 코어 스키마는 도메인 무관 표준을 따르고,
도메인 고유 개념은 프로파일로 분리한다.

| 표준 | 적용 대상 | 상태 |
|---|---|---|
| **ISO 22400-2** | 설비 상태 6종 ↔ OEE 시간 요소 매핑 | 반영 |
| **ISA-95 / IEC 62264** | 설비 계층, 작업 수행 이력 | 부분 |
| **W3C SSN/SOSA** | 관측 패턴 (Sensor / Observation / FeatureOfInterest) | 반영 |
| **AAS (Asset Administration Shell)** | 설비·자산 디지털 표현 | 부분 |
| **Catena-X SAMM** | 배터리 여권 Aspect Model | 계획 |
| **EU 2023/1542** | 디지털 배터리 여권 (2027.2.18~) | 계획 |
| SEMI E10 / E120 / E164 | 반도체 프로파일 전용 | 프로파일로 분리 |

> **이전 버전 주의**: 초기에는 SEMI 표준(반도체 전용)을 코어에 넣었으나,
> 주 도메인이 배터리이므로 코어를 ISO 22400 / ISA-95 / SOSA 기반으로 재정렬하고
> SEMI 참조는 반도체 프로파일로 옮겼다.

### 스키마 구조

```
ontology/schemas/
  core/              도메인 무관 (Equipment, EquipmentModule, Sensor, Measurement, Alarm ...)
  actions/           Action 계층 (상태 변경의 유일한 경로)
  profiles/
    battery/         Roll, WebSegment, ElectrodeLot, Cell, Module, Pack   <- 주 도메인
    semiconductor/   Lot, Wafer                                           <- 대조군
```

코어를 도메인 중립으로 유지하는 것은 테스트로 강제한다
(`tests/test_ontology_schemas.py`).

## 배터리 추적성 — 연속 공정과 이산 공정의 연결

배터리는 전극 공정이 **연속(roll-to-roll)**이라 반도체의 Lot/Wafer 모델이
그대로 맞지 않는다. 좌표계가 두 개이고 그 사이에 전환점이 있다.

```
[연속 좌표계]              전환점              [이산 좌표계]
(roll_id, position_m)  ──  ElectrodeLot  ──  cell_id → module_id → pack_id
```

`ElectrodeLot` 이 둘을 잇는 유일한 고리이며, 그 `position_start_m` /
`position_end_m` 이 스키마 전체에서 가장 중요한 필드다. 이 연결이 끊기면
근본원인 역추적, 영향 범위 순추적, 배터리 여권 대응이 모두 불가능해진다.

셀 하나에는 **양극 Lot 과 음극 Lot 이 각각** 들어가므로 계보가 **수렴**한다.
반도체의 단순 포함 관계와 다르다.

자세한 설계는 **[docs/10_배터리_추적성_설계.md](docs/10_배터리_추적성_설계.md)** 참조.

### 온톨로지 문서

| 문서 | 내용 |
|---|---|
| [01_프로젝트_개요](docs/01_프로젝트_개요.md) | 프로젝트 배경과 목표 |
| [02_관계발견_엔진](docs/02_관계발견_엔진.md) | 상관/인과/패턴 분석 |
| [03_도메인지식_템플릿](docs/03_도메인지식_템플릿.md) | 전문가 지식 수집 양식 |
| [04_개발진행_현황](docs/04_개발진행_현황.md) | 구현 현황 |
| **[05_역량질문](docs/05_역량질문.md)** | 온톨로지가 답해야 할 질문 목록 (요구사항이자 평가 기준) |
| **[06_표준_적용_가이드](docs/06_표준_적용_가이드.md)** | SEMI/ISA-95/SOSA 적용, 관계 메타데이터 규약 |
| **[07_개선_로드맵](docs/07_개선_로드맵.md)** | 실데이터 연동 전 과제, 배치/스트리밍 판단 기준 |
| **[08_액션_계층](docs/08_액션_계층.md)** | Kinetic Layer - 상태 변경의 유일한 경로, 권한/감사/시뮬레이션 |
| **[10_배터리_추적성_설계](docs/10_배터리_추적성_설계.md)** | 연속↔이산 계보, 전극 Lot 전환점, 추적 신뢰도, 여권 연계 |

## 계보 데모 — 설계가 실제로 작동하는지 확인

실데이터가 없어도 온톨로지 설계를 검증할 수 있다.
합성 데이터에 코팅 두께 이상을 심어두고, 계보를 타고 **스스로 찾아내는지**
확인한다. DB 구축 없이 Parquet 를 DuckDB 로 직접 질의한다.

```bash
pip install duckdb pyarrow pandas numpy
python scripts/generate_battery_genealogy.py    # landing/ 에 Parquet 생성
python scripts/demo_genealogy_queries.py        # 계보 질의 실행
```

생성되는 데이터는 실제 수집 형태와 동일하다 —
`landing/<table>/dt=YYYY-MM-DD/hour=HH/*.parquet` + `_SUCCESS` 마커.

| 질의 | 확인하는 것 |
|---|---|
| Q1 | 구간 조인 — 연속 좌표계와 이산 좌표계가 연결되는가 |
| Q2 | **역추적** — 불량 셀에서 원인 롤 구간을 찾아내는가 (근본원인) |
| Q3 | **순추적** — 이상 구간이 들어간 셀/모듈/팩 (격리·리콜 범위) |
| Q4 | 추적 신뢰도별 분리 — 추정 계보가 결론을 오염시키지 않는가 |
| Q5 | 위치 신뢰도 — 시간→위치 변환이 깨지는 구간 분리 |

실데이터가 오면 바꿀 것은 테이블/컬럼명뿐이고 질의 구조는 그대로다.

## Action 계층 (Kinetic Layer)

팔란티어 Foundry 온톨로지의 3계층 중 **운동 계층**에 해당한다.

> **상태 변경은 오직 Action을 통해서만 일어난다.**
> 서비스가 DB에 직접 쓰면 누가 무엇을 왜 바꿨는지 알 수 없고,
> 권한도 감사도 시뮬레이션도 불가능해진다.

```
GET  /api/actions                         액션 타입 목록
POST /api/actions/{type}/simulate         쓰기 없이 예상 결과만
POST /api/actions/{type}                  수행 (인증 필수, 전건 감사)
GET  /api/actions/audit/recent            감사 기록 (거부된 시도 포함)
```

| 액션 | 대상 | 사유 필수 |
|---|---|---|
| `VerifyRelationship` | 발견된 관계 검증/거부 | O |
| `RecordExpertRelationship` | 전문가 지식 관계 등록 | O |
| `AcknowledgeAlarm` | 알람 확인/에스컬레이션 | X |
| `UpdateEquipmentState` | SEMI E10 상태 전이 (전이 규칙 검증) | O |
| `HoldLot` | Lot 홀드/해제 (사람 확인 필요) | O |
| `RequestInspection` | 점검 지시 생성 (사람 확인 필요) | O |

모든 시도는 `action_audit` 테이블에 기록된다 — 누가, 언제, 무엇을, 왜,
무엇이 바뀌었나(before/after). 자세한 내용은
**[docs/08_액션_계층.md](docs/08_액션_계층.md)** 참조.

## 온톨로지 모델

### 배터리 제조 계층 구조

```
Roll (전극롤)
  │
  │ PRODUCES (1:N)
  ▼
Cell (셀)
  │
  │ ASSEMBLED_INTO (N:1)
  ▼
Module (모듈)
  │
  │ ASSEMBLED_INTO (N:1)
  ▼
Pack (팩)
```

### Object Types (정점)

| Object Type | 설명 | 주요 속성 |
|-------------|------|----------|
| **Roll** | 전극 롤 | roll_type, coating_thickness, porosity |
| **Cell** | 배터리 셀 | capacity_ah, voltage_v, resistance, grade |
| **Module** | 배터리 모듈 | cell_count, series/parallel, BMS 정보 |
| **Pack** | 배터리 팩 | energy_kwh, EOL 테스트, 출하 정보 |
| Equipment | 설비/장비 | type, status, location |
| Process | 공정 단계 | step_id, recipe |
| Alarm | 알람/이벤트 | severity, code, timestamp |

### Link Types (간선)

| Link Type | 관계 | 설명 |
|-----------|------|------|
| **PRODUCES** | Roll → Cell | 롤에서 셀 생산 (1:N) |
| **ASSEMBLED_INTO** | Cell → Module → Pack | 조립 관계 (N:1) |
| PROCESSED_AT | Lot → Equipment | 처리 설비 |
| CORRELATES_WITH | Parameter ↔ Parameter | 상관관계 (자동 발견) |
| INFLUENCES | Parameter → Parameter | 인과관계 (자동 발견) |

## 디렉토리 구조

```
manufacturing-ontology-platform/
├── docker-compose.yml          # 서비스 오케스트레이션
├── .env.example                # 환경 변수 템플릿
│
├── ontology/                   # 온톨로지 정의
│   ├── schemas/
│   │   ├── objects/           # Object Type YAML
│   │   │   ├── roll.yaml      # 전극 롤 ★
│   │   │   ├── cell.yaml      # 배터리 셀 ★
│   │   │   ├── module.yaml    # 배터리 모듈 ★
│   │   │   ├── pack.yaml      # 배터리 팩 ★
│   │   │   └── equipment.yaml
│   │   └── links/             # Link Type YAML
│   │       ├── produces_cell.yaml      # Roll→Cell ★
│   │       ├── assembled_into_module.yaml  # Cell→Module ★
│   │       └── assembled_into_pack.yaml    # Module→Pack ★
│   └── migrations/            # SQL 마이그레이션
│
├── api/                        # FastAPI 서버
│   └── src/
│       ├── routers/
│       │   ├── agent.py       # AI Agent API ★
│       │   ├── ontology.py
│       │   └── analytics.py
│       ├── services/
│       └── graphql/
│
├── analytics/                  # 분석 엔진
│   └── src/
│       ├── agent/             # FDC Analysis Agent ★
│       │   ├── fdc_agent.py   # Agent 코어
│       │   ├── ollama_client.py # Ollama LLM 클라이언트
│       │   └── tools.py       # 분석 도구
│       ├── anomaly_detection/
│       ├── relationship_discovery/
│       └── spc/
│
├── frontend/                   # React 프론트엔드
│   └── src/
│       ├── components/
│       │   ├── AgentChat/     # Agent 채팅 UI ★
│       │   ├── OntologyGraph/
│       │   └── Dashboard/
│       └── pages/
│           └── AgentPage.tsx  # Agent 페이지 ★
│
└── docs/                       # 상세 문서
```

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

### 2. Ollama + EXAONE 설치

```bash
# Ollama 설치 (https://ollama.ai)
# EXAONE 3.5 모델 다운로드
ollama pull exaone3.5:7.8b
```

### 3. API 서버 실행

```bash
cd api/src
# 리포지토리 루트를 PYTHONPATH에 포함해야 common/ 공용 모듈이 로드된다
PYTHONPATH=.:../..:../../analytics/src python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 3-1. Docker Compose 기동 (프로파일)

서비스 정의는 전부 유지하되, 필요한 것만 기동한다.

```bash
# 기본: PostgreSQL+AGE / TimescaleDB / Redis / API / Frontend / Analytics
docker compose up

# + Kafka, Flink 스트리밍 경로
docker compose --profile streaming up

# + Debezium CDC (Kafka Connect)
docker compose --profile cdc --profile streaming up
```

배치(Parquet 시간 단위 수집) 구조에서는 기본 프로파일만으로 동작한다.
스트리밍 전환 판단 기준은 `docs/07_개선_로드맵.md` 참조.

### 3-2. 인증

Action 엔드포인트는 인증이 필수다 (JWT Bearer).
로컬 개발 중에는 아래로 우회할 수 있으나 **운영에서는 반드시 꺼야 한다.**

```bash
export AUTH_DEV_MODE=true   # 토큰 없이 고정 주체(dev.user)로 동작
```

### 3-3. 테스트

```bash
pip install -r requirements-dev.txt
pytest            # 리포지토리 루트에서 실행
```

### 4. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
# 브라우저: http://localhost:3000/agent
```

### 5. Agent 테스트

```bash
# curl로 테스트
curl -X POST http://localhost:8001/api/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Cell 용량 불량이 발생했습니다. Roll 공정부터 점검 순서를 알려주세요."}'
```

## 접속 URL

| 서비스 | URL | 설명 |
|--------|-----|------|
| Frontend | http://localhost:3000 | React 대시보드 |
| **AI Agent** | http://localhost:3000/agent | Agent 채팅 UI |
| API Docs | http://localhost:8001/docs | Swagger UI |
| Ontology Graph | http://localhost:3000/ontology | 그래프 시각화 |

## 기술 스택

| 분류 | 기술 |
|------|------|
| **AI/LLM** | Ollama + EXAONE 3.5:7.8b (LG AI Research) |
| **그래프 DB** | PostgreSQL + Apache AGE |
| **시계열 DB** | TimescaleDB |
| **메시지 브로커** | Apache Kafka |
| **스트림 처리** | Apache Flink (PyFlink) |
| **API** | FastAPI + GraphQL (Strawberry) |
| **프론트엔드** | React + TypeScript + D3.js |
| **분석** | Python (NumPy, SciPy, scikit-learn) |

## 향후 계획

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 샘플 데이터로 관계 발견 검증 | ✅ 완료 |
| Phase 2 | 실시간 경보 시스템 구축 | ✅ 완료 |
| Phase 3 | AI Agent 통합, 배터리 도메인 지식 | ✅ 완료 |
| Phase 4 | 실제 DB 연동, RAG 지식 시스템 | 🔜 예정 |
| Phase 5 | Production 배포, 성능 최적화 | 🔜 예정 |

## 라이선스

MIT License
