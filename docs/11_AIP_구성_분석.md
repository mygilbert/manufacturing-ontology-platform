# Palantir AIP 구성 분석과 우리 프로젝트에의 적용

> 조사 시점 2026-09. 공개 문서와 그 미러를 근거로 정리했다.
> 제품명이 이동 중이다 (AIP Agent Studio → AIP Chatbot Studio).
> 세부는 현재 버전과 다를 수 있으므로 결론보다 **설계 원칙**에 무게를 둔다.

## 왜 이 문서가 있는가

이 리포지토리는 "팔란티어 온톨로지 개념 적용"에서 출발했다.
그렇다면 그 원본이 실제로 무엇으로 구성돼 있는지 확인해 두는 편이
우리가 무엇을 따라 하고 무엇을 버릴지 판단하는 데 도움이 된다.

결론부터: **AIP를 복제하는 것은 목표가 아니다.** 12계층 중 상당수는
"수천 고객이 각자 다른 도메인·규제 환경에서 쓴다"는 조건에서 나온 복잡도다.
고객이 하나인 우리에게는 필요 없다. 가져올 것은 **검증된 설계 원칙**이다.

---

## 1. 위치 관계

AIP는 Foundry 위에 얹힌 앱이 아니라 같은 층위의 구성원이다.

```
Foundry (데이터 운영)  +  AIP (생성형 AI)  +  Apollo (배포)
                  하나의 공유 서비스 메시
```

Foundry 내부는 두 층으로 나뉜다.

```
Object Layer (Ontology)   <- AIP 가 붙는 곳
Data Layer                <- 파이프라인, 데이터셋
```

**LLM 이 아니라 온톨로지가 중심에 있다.** 이것이 다른 AI 플랫폼과
구조적으로 갈리는 지점이다.

---

## 2. AIP 12계층

Palantir 자신이 AIP 를 12개 범주로 정리한다.

| # | 범주 | 내용 |
|---|---|---|
| 1 | 보안 LLM 접근 | Palantir 관리 인프라 경유. 제3자 보존·재학습 차단 |
| 2 | 종단 관측성 | 데이터 흐름, 사용자 행위, 에이전트 동작, **토큰 소비**, 실행 체인 |
| 3 | 컨텍스트 엔지니어링 | 배치·스트리밍·실시간 복제 전반의 맥락 통합 |
| 4 | **온톨로지 시스템** | 데이터·로직·액션·보안을 하나로 표현 |
| 5 | 벡터·컴퓨트·툴 서비스 | 임베딩, 확장 컴퓨트, "툴 팩토리" |
| 6 | 보안·거버넌스 | 역할 기반 + 마킹 기반 + **목적 기반**, 전면 감사 |
| 7 | 에이전트 수명주기 | no-/low-/pro-code 빌드 + 오케스트레이션 + 평가 |
| 8 | 운영 자동화 | 스케줄·이벤트·API 구동 |
| 9 | 개발 환경 | IDE, SDK |
| 10 | 사람+AI 애플리케이션 | 분석·실시간·거버넌스 워크플로 |
| 11 | 패키징·릴리스·배포 | DevOps 툴체인 |
| 12 | 엔터프라이즈 자동화 | **사람과 동일한 거버넌스 아래 동작하는** 에이전트 |

---

## 3. 빌더가 실제로 만지는 네 가지

| 도구 | 성격 | 산출물 |
|---|---|---|
| **AIP Logic** | 노코드 LLM 함수 개발 | Function |
| **AIP Chatbot Studio** (구 Agent Studio) | 대화형 에이전트 | Function 으로 published |
| **AIP Evals** | 평가 스위트 | 합격/불합격 + 분산 |
| 플랫폼 내장 | Pipeline Builder LLM 노드, Functions, Transforms, Code Workspaces | — |

### 3.1 AIP Logic — 블록 체인

Logic 함수는 입력(온톨로지 객체 또는 문자열)을 받아
**값을 반환하거나 온톨로지를 편집**한다.

구성 단위는 **블록**이다. 블록은 온톨로지 읽기/쓰기, 계산, 집계,
다른 함수 호출, LLM 상호작용을 수행하며,
**한 블록의 출력이 다음 블록의 입력**이 되어 체인을 이룬다.

실행 후 **디버거**가 LLM 의 사고 흐름(chain-of-thought)을 펼쳐 보인다 —
생성된 프롬프트, 호출한 툴, 각 단계.

### 3.2 AIP Chatbot — 7개 구성요소

| 요소 | 내용 |
|---|---|
| System Prompt | 지시 + **툴 설명 + 변수 설명**이 합성되어 만들어짐 |
| Application State | 프롬프트 내 변수 (구 parameters). LLM 동작 제어 |
| Retrieval Context | **메시지마다 결정론적으로 실행**되어 주입되는 정보 |
| Tools | LLM 이 쓸 수 있는 외부 기능 |
| Context Window | 위 전부 + 대화 이력이 들어가는 총량 |
| Sessions | 대화 인스턴스. 창을 넘기면 오류 → 새 세션 |
| Published Versions | **Function 으로 발행** → 플랫폼 어디서나 실행 가능 |

마지막 항목이 구조적으로 중요하다. 에이전트가 Function 이 되면
다른 Logic 에서 호출할 수도, Evals 에 걸 수도, 자동화에 넣을 수도 있다.
**에이전트가 1급 호출 단위가 된다.**

### 3.3 Retrieval Context 3종

매 사용자 메시지마다 **결정론적으로** 실행된다.
LLM 이 부를지 말지 고르는 것이 아니라 무조건 들어간다.

| 종류 | 동작 |
|---|---|
| Ontology context | 고정 객체 집합, 또는 질의에 대한 시맨틱 검색으로 관련 객체 선별 |
| Document context | 전문 모드 / 관련 청크 모드(시맨틱 검색) |
| Function-backed | 사용자가 직접 검색 로직 작성 |

### 3.4 Tool 4종

| 툴 | 하는 일 |
|---|---|
| Object query | 필터 · 집계 · 검사 + **링크 순회** |
| **Action** | **온톨로지 편집 실행.** 자동 실행 / 사용자 확인 후 실행 설정 가능 |
| Function | Foundry 함수 호출 (published Logic 함수 포함) |
| Update application variable | 앱 상태 갱신 |

Native tool calling 모드에서 이 4종이 지원된다.

---

## 4. 보안 모델 — 가장 배울 점

문서의 한 문장이 설계 전체를 요약한다.

> **LLM 은 툴에 직접 접근하지 못한다. LLM 은 툴 사용을 *요청*할 수 있을 뿐이고,
> 그 툴 호출은 *호출한 사용자의 권한으로* 실행된다.**

즉 **LLM 은 권한 주체가 아니다.** 제안자일 뿐이고 집행자는 플랫폼이다.
권한 상승 경로가 원천 차단된다.

그 위에 통제가 겹쳐진다.

- 역할 기반 + **마킹 기반**(데이터 등급) + **목적 기반(PBAC)** — 사전 정의된 용도로만 사용
- **지리 제한** — 모델 요청·응답이 규제 관할 밖으로 나가지 않음
- **제3자 보증** — 프롬프트/완성 미보존, 재학습 미사용. 기술적·계약적 보장.
  제공자 인력도 접근 불가. 완료 후 즉시 폐기

---

## 5. 모델 계층 — k-LLM

특정 모델에 종속되지 않는 추상화.

- 제공자: OpenAI, Anthropic, Google, Meta, xAI
- 호스팅 경로: Azure OpenAI, AWS Bedrock, GCP Vertex, Palantir 자체 호스팅(Llama 등)
- **Bring Your Own Model** 지원
- **AIP Model Catalog** 로 모델 관리
- LLM-provider compatible APIs — 기존 SDK 호환 계층

Evals 에서 **모델 간 성능 비교**가 가능한 이유가 이 추상화다.

---

## 6. AIP Evals — 비결정적 출력을 결정론적으로 검증

**평가 스위트 = 테스트 케이스 + 평가 함수.**

- 테스트 케이스: 수동 정의 / **오브젝트셋으로 대량 생성** / 혼합
- 평가 대상: 정확도뿐 아니라 **시간(비용)과 대화형 성능**까지
- **LLM-as-a-Judge** 지원
- 합격 기준 설정 시 케이스별 Passed/Failed 자동 판정, 전체 합격률 표시
- **모델 간 비교**, **실행 간 분산** 측정

그리고 설계적으로 가장 인상적인 지점:

> 온톨로지 편집(객체 생성·수정·삭제)을 포함하는 함수는,
> 각 테스트 케이스가 **온톨로지 시뮬레이션 안에서 실행된다.**

평가하느라 실제 상태를 바꾸지 않는다.
**우리 Action 계층의 `dry_run` 과 정확히 같은 개념이다** (docs/08 참조).
같은 문제에서 같은 답이 나온 것이다 — 상태를 바꾸는 연산을 통제하려면
"실행하지 않고 결과만 보는" 모드가 반드시 필요하다.

---

## 7. 우리 프로젝트와의 대조

| AIP | 우리 | 위치 |
|---|---|---|
| 온톨로지 (객체/링크/액션) | ✅ | `ontology/schemas/` |
| **Action = 상태 변경의 유일한 경로** | ✅ | `common/actions/` |
| **온톨로지 시뮬레이션** | ✅ `dry_run` | `common/actions/executor.py` |
| 호출자 권한으로 툴 실행 | ✅ `Principal` + 역할 검증 | `api/src/auth.py` |
| 감사 (누가·언제·무엇을·왜) | ✅ (아직 InMemory) | `action_audit` |
| 에이전트 대행 기록 | ✅ `is_agent` / `on_behalf_of` | — |
| 사람 확인 후 실행 | ✅ `requiresHumanConfirmation` | 액션 스키마 |
| Object query tool | △ 구조만, 하드코딩 샘플 | `analytics/src/agent/tools.py` |
| **Retrieval Context (결정론 주입)** | ❌ | — |
| **Evals (평가 스위트)** | ❌ | — |
| **관측성 / 토큰 회계** | ❌ | — |
| 에이전트를 Function 으로 발행 | ❌ | — |
| 마킹 · 목적 기반 접근통제 | ❌ (역할 기반만) | — |
| k-LLM 추상화 | ❌ (Ollama/EXAONE 고정) | — |

이미 맞춰진 것이 적지 않다. 특히 `dry_run` 은 우연이 아니다.

---

## 8. 무엇이 병목인가 — 데이터가 아닌 것들

"실데이터만 확보되면 AIP 비슷하게 만들 수 있는가"에 대한 답.

| 구분 | 항목 | 판단 |
|---|---|---|
| **진짜 데이터가 병목** | 계보 실체화, Object query 실동작, Retrieval Context 의 내용, 근본원인·영향범위 질의 | 구조 완성. 꽂으면 됨 |
| **데이터와 무관 · 완료** | Action, dry_run, 감사 | ✅ |
| **데이터와 무관 · 미구현** | **Evals**, **관측성**, **Retrieval Context 구조** | **지금 가능** |
| **사람이 병목** | 전문가 지식(계층2), 역량질문 확정, 검증 루프 | 인터뷰 필요 |
| **조직 · 법무가 병목** | 마킹/목적 기반 통제, 지리 제한, 제3자 무보존 계약 | 코드로 못 메움 |

마지막 줄이 중요하다. AIP 보안 계층의 상당 부분은 코드가 아니라
**사내 IAM 체계, 데이터 등급 분류, 법무 계약**이다.
사내 배포라면 기존 사내 통제를 쓰면 되고, 새로 만들 이유가 없다.

---

## 9. "AIP-like" 의 최소 정의

12계층을 전부 흉내 낼 필요는 없다. 실질을 만드는 것은 여섯 개다.

| # | 요소 | 상태 | 막고 있는 것 |
|---|---|---|---|
| 1 | 온톨로지 (객체 · 링크 · 계보) | ✅ | — |
| 2 | Action (단일 경로 + dry_run + 감사) | ✅ | — |
| 3 | Retrieval Context (결정론 주입) | ❌ | **없음. 지금 가능** |
| 4 | Tools (object query / action / function) | △ | **데이터** |
| 5 | Evals (역량질문 기반) | ❌ | **없음. 지금 가능** |
| 6 | 관측성 (체인 · 토큰 · 비용) | ❌ | **없음. 지금 가능** |

**여섯 개 중 데이터를 실제로 기다려야 하는 것은 4번 하나다.**

3 · 5 · 6 은 데이터 없이 만들 수 있고, **먼저 만들어두면 데이터가 왔을 때
바로 측정이 된다.** 현재 순서가 거꾸로 되어 있다.

---

## 10. 할 수 있는 것과 없는 것

| | |
|---|---|
| **가능** | 단일 도메인(배터리 전극) · 단일 조직용 **온톨로지 운영 시스템** |
| **불가능** | 다도메인 · 다조직 · 다테넌트 **플랫폼** |

AIP 복잡도의 상당 부분은 고객이 여럿이라는 조건에서 온다.
k-LLM 추상화, 모델 카탈로그, 지리 제한, 마킹 체계 — 전부 그 때문이다.
고객이 하나면 그 복잡도가 필요 없다.

목표를 "AIP 클론"으로 잡으면 실패하고,
**"AIP 가 검증한 설계 원칙을 우리 도메인에 적용"** 으로 잡으면 성공한다.
그 원칙 중 넷은 이미 적용했다 — 온톨로지 중심, Action 단일 경로,
시뮬레이션, 호출자 권한 실행.

---

## 11. 적용 계획

```
[데이터 없이 — 지금]
  1. Evals 프레임         역량질문 40개(docs/05) -> 평가 스위트
                         "지금 몇 개에 답하는가"가 숫자로 나온다
  2. Retrieval Context    결정론 주입 / 선택 조회 분리
  3. 관측성               체인 · 토큰 · 비용 기록 (Action 감사 확장)

[데이터 도착 후]
  4. Tools 실동작         Agent 도구를 실 DB 에 연결
  5. 전체 회귀            Evals 로 측정 -> 개선 루프
```

1번을 먼저 하는 이유: 지금 만들어두면
**"데이터 투입 전 0/40 → 투입 후 n/40"** 이 측정된다.
나중에 만들면 before 가 없어 비교가 불가능하다.

### 우선 적용할 세 가지 원칙

**① Retrieval Context 를 결정론적으로 분리**
현재 우리 Agent 는 LLM 이 툴을 부를지 말지 고른다.
AIP 는 **반드시 들어가야 하는 맥락**(설비 마스터, 검증된 관계, 해당 Lot 의 계보)을
매 메시지에 무조건 주입하고, 선택적 조회만 툴로 둔다.
이 분리가 답변 품질과 토큰 효율을 동시에 잡는다.

**② 역량질문을 평가 스위트로**
재료가 이미 있다 — 역량질문 40개(docs/05), `dry_run` 시뮬레이션,
합성 계보 데이터(`scripts/generate_battery_genealogy.py`).
온톨로지 편집을 포함하는 케이스는 `dry_run` 으로 실행해
실제 상태를 바꾸지 않고 평가한다. AIP Evals 와 같은 방식이다.

**③ 에이전트를 Action 처럼 호출 가능한 단위로**
Chatbot 을 Function 으로 발행하는 구조. 우리로 치면
Agent 분석 결과를 `RequestInspection` 액션의 입력으로 바로 넘기는 경로가 된다.

---

## 12. 가져오지 않을 것

명시해 둔다. 흉내 내면 비용만 든다.

| 항목 | 이유 |
|---|---|
| k-LLM 다중 제공자 추상화 | 고객이 하나. 모델 하나로 시작해 필요할 때 바꾼다 |
| 모델 카탈로그 | 위와 같음 |
| 지리 제한 | 사내망 배포면 해당 없음 |
| 마킹 · 목적 기반 접근통제 | 사내 기존 체계를 쓴다. 새로 만들면 이중 관리 |
| Apollo 급 배포 자동화 | 규모가 다르다 |
| 다테넌트 격리 | 테넌트가 하나 |

---

## 참고

- [AIP architecture overview](https://www.palantir.com/docs/foundry/architecture-center/aip-architecture)
- [AIP Overview](https://www.palantir.com/docs/foundry/aip/overview) ·
  [AIP, Foundry, and Apollo](https://www.palantir.com/docs/foundry/architecture-center/platforms)
- AIP Logic — [Overview](https://www.palantir.com/docs/foundry/logic/overview) ·
  [Core concepts](https://www.palantir.com/docs/foundry/logic/core-concepts) ·
  [Blocks](https://www.palantir.com/docs/foundry/logic/blocks)
- AIP Chatbot Studio — [Core concepts](https://www.palantir.com/docs/foundry/chatbot-studio/core-concepts) ·
  [Tools](https://www.palantir.com/docs/foundry/chatbot-studio/tools) ·
  [Retrieval context](https://www.palantir.com/docs/foundry/agent-studio/retrieval-context)
- AIP Evals — [Create an evaluation suite](https://www.palantir.com/docs/foundry/aip-evals/create-suite) ·
  [Evaluate Ontology edits](https://www.palantir.com/docs/foundry/aip-evals/ontology-edits) ·
  [Analyze run results](https://www.palantir.com/docs/foundry/aip-evals/analyze-run-results)
- [AIP security and privacy](https://www.palantir.com/docs/foundry/aip/aip-security) ·
  [AI ethics and governance](https://www.palantir.com/docs/foundry/aip/ethics-governance)
- [Supported LLMs](https://www.palantir.com/docs/foundry/aip/supported-llms) ·
  [Bring your own model](https://www.palantir.com/docs/foundry/aip/bring-your-own-model) ·
  [AIP Model Catalog](https://www.palantir.com/docs/foundry/model-catalog/overview)
- [Platform overview — AIP capabilities](https://www.palantir.com/docs/foundry/platform-overview/aip-capabilities)
- 문서 미러: [JeremyMeissner/palantir-docs](https://github.com/JeremyMeissner/palantir-docs)

### 관련 문서

- [06_표준_적용_가이드](06_표준_적용_가이드.md) — Bosch / Atlas Copco 온톨로지 구축 사례
- [08_액션_계층](08_액션_계층.md) — Kinetic Layer 구현. AIP Action 에 대응
- [05_역량질문](05_역량질문.md) — Evals 테스트 케이스의 원천
- [07_개선_로드맵](07_개선_로드맵.md) — 전체 과제 우선순위
