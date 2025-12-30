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
PYTHONPATH=.:../../analytics/src python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
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
