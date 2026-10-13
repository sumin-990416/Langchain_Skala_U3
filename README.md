# 배터리 분석 AI 스타트업 투자평가 Agentic RAG

> **SKALA 4기 · 울산 3반 · 4조**
> 팀원: 김대훈 · 강도희 · 이연주 · 정수민 · 손연우

LangGraph 기반 다중 에이전트(Multi-Agent)와 Agentic RAG를 활용하여, 배터리 분석 AI 스타트업의 비정형 문서를 수집·검증하고 **5가지 서로 다른 투자 관점**으로 병렬 평가한 뒤 최종 투자 검토 보고서를 자동 생성하는 파이프라인입니다.

---

## 목차

1. [도메인과 문제 정의](#1-도메인과-문제-정의)
2. [시스템 설계 원칙](#2-시스템-설계-원칙)
3. [Evidence RAG 전략](#3-evidence-rag-전략)
4. [공통 평가 기준과 점수 산식](#4-공통-평가-기준과-점수-산식)
5. [5개 Profile Agent](#5-5개-profile-agent)
6. [Graph 전체 흐름](#6-graph-전체-흐름)
7. [State Schema](#7-state-schema)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [핵심 기술 스택 및 코드 해설](#9-핵심-기술-스택-및-코드-해설)
10. [설치 및 실행 방법](#10-설치-및-실행-방법)
11. [실행 예시](#11-실행-예시)
12. [출력 보고서 예시](#12-출력-보고서-예시)
13. [구현·검증 계획과 담당](#13-구현검증-계획과-담당)

---

## 1. 도메인과 문제 정의

| 항목 | 내용 |
|---|---|
| **대분류** | Energy |
| **세부 분야** | Battery Intelligence · BMS · ESS 분석 AI |
| **분석 대상** | ACCURE Battery Intelligence, volytica diagnostics, Electra Vehicles |

**문제 인식**: 투자자는 배터리 AI 스타트업의 핵심 기술(SOC·SOH·RUL·열화 예측 등), 실제 성능, 고객 실적을 평가해야 하나, 기업마다 공개 지표와 시험 조건이 다르고 자체 주장과 독립 검증이 혼재되어 **직접 비교가 매우 어렵습니다.**

**해결책**: 단순 검색이나 LLM 요약이 아닌, 사실(Evidence) 수집과 투자 성향(Investment Profile) 판단을 **완벽히 분리**하여 다수결 합의 및 이견 분석을 수행하는 의사결정 시스템을 구축합니다.

---

## 2. 시스템 설계 원칙

시스템은 **공통 근거를 만드는 Evidence RAG Subgraph**와, 동일한 근거를 바탕으로 각기 다른 가중치와 투자 철학으로 해석하는 **5개의 Profile Agent**로 구성됩니다.

| 단계 | 동작 | 산출물 |
|---|---|---|
| 1. 고정 입력 | ACCURE·volytica·Electra와 선별 문서 로드 | `current_company` · `source metadata` |
| 2. Evidence RAG | 29개 항목 질문화 → 검색 → 근거 검증 → 재검색 | `validated_evidence` · `unknown_items` |
| 3. 공통 평가 | 동일 평가표로 영역점수와 근거 충족률 계산 | `domain_scores` · `evidence_coverage` |
| 4. Profile Agent | 5개 성향이 서로 다른 가중치로 병렬 판단 | 5개 `AgentResult` |
| 5. 종합 | 평균·편차·합의·소수의견·추가실사 결합 | `FinalInvestmentIndex` |
| 6. 기업 반복 | 3개 기업 결과 저장 후 다음 기업 처리 | `company_results` |
| 7. 보고서 | 기업 간 비교와 실제 사용 출처 정리 | 5쪽 이내 마크다운 보고서 |

**에이전트 설계 원칙**
- 각 Agent는 자유 문장이 아니라 **공통 Pydantic 스키마**로 결과를 반환한다.
- 모든 핵심 판단은 `evidence_id`와 원문 근거 문장에 연결한다.
- 자료가 없으면 낮은 점수를 **추정하지 않고** `unknown`으로 기록한다.
- 기업 자체 주장과 독립 자료를 구분하고, 시험 조건이 다른 성능 수치를 직접 비교하지 않는다.
- 투자금액·매출·고객 수·성능 수치를 **LLM이 추정해 만들지 못하게** 한다.

---

## 3. Evidence RAG 전략

### 3-1. 문서 설계

| 문서 유형 | 예상 분량 | 활용 목적 |
|---|---|---|
| 기업 공식 기술자료·백서 | 40쪽 | 제품·기술 원리·입출력·성능 주장 |
| 고객 사례·보도자료 | 30쪽 | 도입 실적·고객 효과·계약 신호 |
| 논문·특허·외부 검증 | 50쪽 | 기술 타당성·IP·한계 검증 |
| 시장·안전·규제 자료 | 50쪽 | 시장성·표준·법적 책임 |
| 투자·기업 현황 | 20쪽 | 투자단계·팀·재무 신호 |
| **합계** | **최대 200쪽** | 3개 기업 공통 코퍼스 |

### 3-2. Embedding 모델 선정

#### 최종 선정: `OpenAI text-embedding-3-small`

실제 구현에서는 아래의 이유로 **OpenAI `text-embedding-3-small`** 을 채택했습니다.

| 비교 항목 | OpenAI text-embedding-3-small | BAAI/bge-m3 (설계 단계 후보) |
|---|---|---|
| 다국어 | 한국어·영어·독일어 등 100+ 언어 | 한국어·영어·독일어 혼합 검색 |
| 차원 수 | 1,536 (축소 가능) | 1,024 |
| 입력 길이 | **최대 8,191 tokens** | 최대 8,192 tokens |
| 호스팅 | API 호출 (로컬 GPU 불필요) | 로컬 실행 (568M, GPU 권장) |
| 속도 | API 기반으로 **즉시 사용 가능** | 로컬 로딩·추론 시간 필요 |
| 비용 | $0.02 / 1M tokens (매우 저렴) | 무료 (GPU 인프라 비용 별도) |

**선정 이유**
1. **인프라 제약 해소**: 팀 환경에서 GPU 서버 없이도 바로 사용 가능
2. **빠른 프로토타이핑**: API 한 줄로 임베딩 생성 — 설치·모델 로딩 없이 즉시 실험 가능
3. **다국어 성능**: 한국어 질문 → 영어 기업문서 교차 검색에서 충분한 품질 확인
4. **비용 효율**: 200쪽(약 10만 토큰) 코퍼스 기준 임베딩 비용 약 $0.002 — 사실상 무료

#### 설계 단계 후보: `BAAI/bge-m3`

설계 산출물(PDF)에서는 Dense·Sparse·Multi-vector를 동시에 지원하는 `bge-m3`를 1차 후보로 선정했습니다. 향후 하이브리드 검색(키워드 + 의미 검색) 확장이 필요할 경우, 아래 평가 기준으로 모델을 전환할 수 있습니다.

| 평가축 | 지표 | 선정 규칙 |
|---|---|---|
| 정답 검색 | Hit Rate@5 | 우선순위 1 |
| 정답 순위 | MRR@5 | 우선순위 2 |
| 교차언어 | 한국어 질문 → 영문 근거 Hit@5 | 기업·영역별 편차 확인 |
| 운영성 | 평균 검색시간·최대 메모리 | 정확도 차이 3%p 이내면 경량 모델 |
| 안정성 | 동일 입력 반복 순위 편차 | 결과 재현성 확인 |

### 3-3. 청킹·검색 전략
- 제목·소제목·표 제목을 chunk 본문에 포함하고 페이지 번호를 유지
- 표는 행과 열 의미가 사라지지 않도록 헤더를 각 chunk에 반복
- 초기 **Top-K는 5**로 설정하고, 동일 문서 chunk 독점을 막기 위해 문서 다양성 적용
- 회사명·제품명·SOC·SOH·RUL·UL 9540A 등 정확 키워드는 metadata filter와 hybrid 검색 후보로 사용

### 3-4. 근거 등급 체계 (Evidence Grade)

| 등급 | 자료 유형 | 사용 원칙 |
|---|---|---|
| **A** | 규제기관·법원·특허청·공시·인증기관·동료평가 논문·독립 고객 원문 | 사실 확인의 주근거 |
| **B** | 회사 공식 제품문서·투자사 발표·실명 고객 사례 | 구체 조건 확인 후 보조근거 |
| **C** | 회사 홈페이지·보도자료·인터뷰·조건이 부족한 자체 주장 | 주장 존재 확인, 단독 고득점 제한 |
| **U** | 출처 불명·검색 스니펫·확인 불가·해결되지 않은 상충 | 점수근거 제외·unknown |

### 3-5. 재검색(Query Rewrite) 조건
- 검색 결과가 평가 질문과 직접 관련되지 않는다
- 회사 주장만 있고 독립적인 확인 자료가 없다
- 성능 수치에 표본·화학계·시험환경·오차 정의가 없다
- 고객 도입과 단순 파트너십을 구분할 수 없다
- 출처 간 날짜·금액·성능·회사 상태가 상충한다

---

## 4. 공통 평가 기준과 점수 산식

### 4-1. 5대 평가 영역

| 영역 | 주요 평가 질문 | 공통 산출 |
|---|---|---|
| 기술 경쟁력 | AI 모델·입력데이터·SOC/SOH/RUL·성능조건·실시간성·BMS/ESS·특허·현장확장 | 0~100 기술점수 |
| 시장성 | 고객문제·시장규모·경쟁구도·지역/세그먼트 확장·지불의사 | 0~100 시장점수 |
| 재무·고객 실적 | 투자단계·매출·유료고객·파일럿·계약·반복매출·수익모델 | 0~100 재무점수 |
| 안정성 | 안전·규제·제조물책임·데이터 권리·재무연속성·주장 상충 | 0~100 안정성점수 |
| 팀·사업 실행력 | 창업진 경험·기술/영업 균형·제품출시·파트너십·설치/운영 확장 | 0~100 실행점수 |

### 4-2. 공통 점수 척도 (1~5)

| 점수 | 판단 기준 |
|---|---|
| **5** | 독립적이고 구체적인 근거로 우수성이 확인됨 |
| **4** | 구체적 근거가 있으나 일부 제한 또는 미확인 사항이 존재함 |
| **3** | 기본 역량은 확인되지만 차별성이나 성과 근거가 제한적임 |
| **2** | 회사 주장은 있으나 외부 검증 또는 상용 실적이 부족함 |
| **1** | 핵심 역량의 근거가 거의 없거나 중대한 약점이 확인됨 |
| **미확인** | 필요한 공개자료가 없음. **0점으로 환산하지 않고** 근거 충족률에만 반영 |

### 4-3. 점수 계산 산식

```
세부항목 점수       = 1~5점 또는 미확인
DomainScore(d)     = 확인된 항목의 가중평균 × 20
ProfileScore(p)    = Σ[ DomainScore(d) × ProfileWeight(p,d) ]
FinalInvestmentIndex = mean(5개 ProfileScore)
ViewpointDispersion  = stdev(5개 ProfileScore)
EvidenceCoverage     = 확인된 세부항목 수 / 전체 세부항목 수 × 100
```

### 4-4. 최종 판정 기준

| 조건 | 기본 판정 | 출력 원칙 |
|---|---|---|
| 최종지표 ≥ 75점 · 근거충족률 ≥ 80% | **투자검토 추천** | 핵심 근거와 미확인 항목 병기 |
| 최종지표 60~74점 또는 근거충족률 60~79% | **조건부 검토** | 충족해야 할 실사조건 제시 |
| 최종지표 < 60점 | **보류** | 낮은 점수의 근거와 개선조건 제시 |
| 근거충족률 < 60% | **판단 유보** | 점수보다 데이터룸 요청을 우선 |

---

## 5. 5개 Profile Agent

다섯 Agent는 같은 **공통 영역점수**를 입력받고 투자 성향에 따라 **가중치만 달리**합니다.

| 담당 | Agent | 핵심 책임 | 기술 | 시장 | 재무 | 안정성 | 팀·사업 |
|---|---|---|---|---|---|---|---|
| 손연우 | **기술중심형** | 모델·데이터·BMS/ESS 연동·특허·현장확장성 | **35%** | 20% | 10% | 15% | 20% |
| 강도희 | **안정형** | 투자단계·재무연속성·안전·규제·데이터 권리 | 15% | 15% | **30%** | **30%** | 10% |
| 김대훈 | **성장형** | AI 확장성·시장성장·고객확대·제품 확장 가능성 | 20% | **35%** | 15% | 10% | 20% |
| 이연주 | **균형형** | 기술·시장·재무·안정·사업을 동일 비중으로 해석하고 편향 점검 | 20% | 20% | 20% | 20% | 20% |
| 정수민 | **사업성중심형** | 수익모델·고객·계약·팀·실행력·통계 타당성 | 15% | 25% | 20% | 10% | **30%** |

**공통 AgentResult 스키마**

| 필드 | 설명 |
|---|---|
| `profile_name` | 기술·안정·성장·균형·사업 중 하나 |
| `weighted_score` | 해당 Profile 가중치를 적용한 0~100점 |
| `domain_contributions` | 영역별 점수·가중치·기여점수 |
| `supporting_evidence_ids` | 찬성 근거 ID |
| `contrary_evidence_ids` | 반대·상충 근거 ID |
| `unknown_items` | 공개자료로 확인하지 못한 항목 |
| `recommendation` | 추천·조건부 검토·보류 |
| `due_diligence_questions` | 투자 전 추가 확인 질문 |

---

## 6. Graph 전체 흐름

```
                              ┌─────────────────────────────────┐
                              │          [START]                │
                              └────────────┬────────────────────┘
                                           ▼
                              ┌─────────────────────────────────┐
                              │  고정 기업 3곳 + 문서 코퍼스 로드  │
                              └────────────┬────────────────────┘
                                           ▼
                     ┌─────── │  현재 기업 선택 (1/3)             │ ◄──────────┐
                     │        └────────────┬────────────────────┘            │
                     │                     ▼                                 │
                     │        ┌─────────────────────────────────┐            │
                     │        │  29개 항목 평가 질문 생성         │            │
                     │        └────────────┬────────────────────┘            │
                     │                     ▼                                 │
                     │   ┌──► ┌─────────────────────────────────┐            │
                     │   │    │  Evidence RAG 검색 (Top-K=5)     │            │
                     │   │    └────────────┬────────────────────┘            │
                     │   │                 ▼                                 │
                     │   │    ┌─────────────────────────────────┐            │
                     │   │    │  관련성 · 출처등급 · 상충 검증    │            │
                     │   │    └──┬──────────┬───────────────┬───┘            │
                     │   │       │          │               │                │
                     │   │   충분 ▼     부족(< 2회)     2회 도달             │
                     │   │       │          │               │                │
                     │   │       │    ┌─────┴─────┐   ┌─────┴─────┐          │
                     │   │       │    │  Query     │   │  미확인   │          │
                     │   │       │    │  Rewrite   │   │  항목기록 │          │
                     │   │       │    └─────┬─────┘   └─────┬─────┘          │
                     │   │       │          │               │                │
                     │   └───────┘──────────┘               │                │
                     │                     ┌────────────────┘                │
                     │                     ▼                                 │
                     │        ┌─────────────────────────────────┐            │
                     │        │  공통 5개 영역 점수 계산          │            │
                     │        │  + 근거 충족률(Coverage) 산출     │            │
                     │        └────────────┬────────────────────┘            │
                     │                     ▼                                 │
                     │        ┌─────────────────────────────────────────┐    │
                     │        │  5개 Profile Agent 병렬 실행 (Send API) │    │
                     │        │                                         │    │
                     │        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
                     │        │  │기술형│ │안정형│ │성장형│ │균형형│ │사업형│
                     │        │  │손연우│ │강도희│ │김대훈│ │이연주│ │정수민│
                     │        │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
                     │        └─────┼────────┼────────┼────────┼────────┼────┘
                     │              └────────┴────────┼────────┴────────┘
                     │                                ▼
                     │        ┌─────────────────────────────────┐
                     │        │  Agent 결과 종합 · 최종 투자지표  │
                     │        └────────────┬────────────────────┘
                     │                     ▼
                     │        ┌─────────────────────────────────┐
                     │        │  기업 결과 저장                  │
                     │        └────────────┬────────────────────┘
                     │                     ▼
                     │              ┌──────┴──────┐
                     │              │ 남은 기업?   │
                     │              └──┬───────┬──┘
                     │            예   │       │  아니오
                     └─────────────────┘       ▼
                              ┌─────────────────────────────────┐
                              │  3개 기업 비교                   │
                              └────────────┬────────────────────┘
                                           ▼
                     ┌──────► ┌─────────────────────────────────┐
                     │        │  투자보고서 생성 (5쪽 이내)       │
                     │        └────────────┬────────────────────┘
                     │                     ▼
                     │        ┌─────────────────────────────────┐
                     │  오류  │  인용 · 수치 · REFERENCE 검증    │
                     └────────┤                                 │
                              └────────────┬────────────────────┘
                                      통과 │
                                           ▼
                              ┌─────────────────────────────────┐
                              │          [END]                  │
                              └─────────────────────────────────┘
```

**흐름을 한 줄로 요약하면:**

> 기업 로드 → 질문 생성 → RAG 검색 → 공통 점수 → State에 구조화 계획 저장 → 계획된 Profile Worker 병렬 실행 → 기업별 종합 → 보고서 생성 → 품질 평가 → 필요하면 후속 검색·재평가(최대 2회) → 최종 보고서

| 구조 | 설계 내용 | 종료·통제 |
|---|---|---|
| Workflow | 기업별 질문·검색·점수 → 구조화된 계획 → 동적 worker → 종합 | 모든 기업 처리 후 보고서 단계 |
| Profile Fan-out | planner가 State에 기록한 task를 `Send`로 실행 | task별 상태·결과를 reducer가 수집 |
| Company Loop | `company_result` 저장 후 `current_index` 증가 | 설정된 기업 처리 뒤 비교 |
| Quality Loop | Judge가 근거성·중립성·편향 통제를 확인하고 후속 조사 | 최대 2회, 미달·Judge 오류는 사람 검토로 표시 |
| Step Bound | planner·worker·평가 단계의 실행 횟수를 State에서 누적 | `max_steps` 초과 시 추가 조사 중지 |

---

## 7. State Schema

```python
class InvestmentAgentState(TypedDict):
    selected_companies: List[Company]          # 사전 선정 완료된 3개 기업
    current_index: int                         # 현재 기업 위치
    current_company: Optional[Company]         # 현재 분석 기업
    evaluation_questions: List[Question]       # 29개 항목 기반 평가 질문
    retrieved_evidence: List[Evidence]         # Top-K 검색 결과
    validated_evidence: List[Evidence]         # 근거검증 통과 결과
    evidence_coverage: float                   # 질문·항목 근거 충족률
    retry_count: Dict[str, int]               # 질문별 재검색 횟수
    unknown_items: List[str]                   # 재검색 후 미확인 항목
    domain_scores: Dict[Domain, float]        # 공통 5개 영역 점수
    profile_results: Annotated[List, reset_or_add] # 기업 내 병렬 Profile 결과
    company_result: Optional[CompanyResult]    # 현재 기업 종합결과
    company_results: Annotated[List, merge_company_results] # 기업 결과 누적·교체
    final_comparison: Optional[ComparisonResult] # 기업 간 비교·최종지표
    report_draft: str                          # 5쪽 보고서 작성용 내용
    report_errors: List[str]                   # 인용·수치·형식 검증 오류
    trace_id: str                              # 실행 추적 식별자
    structured_tasks: List[Dict[str, Any]]     # planner가 만든 worker 작업
    worker_status: Annotated[Dict, merge]      # task별 성공·실패
    step_count: Annotated[int, operator.add]   # 실행 상한 제어
    max_steps: int                             # 전체 실행 상한
    evaluation_round: int                      # 품질 평가 라운드
    max_evaluation_rounds: int                 # 품질 평가 상한
    report_quality: Dict[str, Any]             # Judge verdict와 재작업 제안
    evidence_by_company: Annotated[Dict, merge_company_evidence] # 제한된 근거 발췌
    followup_results: Annotated[List, merge_followup_results] # 재평가 결과
```

> **State 설계 원칙**
> - 노드는 State 전체가 아니라 자신이 담당한 key만 부분 업데이트한다.
> - `profile_results`, `company_results`, `worker_status`, `evidence_by_company`, `followup_results`는 **Reducer**를 사용해 병렬·반복 결과를 병합한다.
> - 기존 RAG 코드의 `retry_count`·Query Rewrite 경로는 남아 있지만, 현재 `route_evidence_check`는 임시 전체 통과 분기라 해당 재검색 상한이 실제로 작동하지 않는다. 이 한계는 Agent 품질 평가의 후속 조사 루프와 별개다.

| 설계 항목 | 적용 방식 | 이유·한계 |
|---|---|---|
| 제어와 페이로드 | `trace_id`, 상태·라운드·상한은 제어 메타; task·평가 결과·보고서는 작업 데이터 | 라우팅에 필요한 값과 분석 결과를 구분 |
| 관측성 | 로거와 LangSmith metadata에 trace ID·분기 결정을 기록 | 결정 사유를 State에 중복 적재하지 않음 |
| State 크기 | 재조사 입력은 회사별 최대 160개 근거 발췌, 각 900자; 전체 보고서는 최종 산출물 | 재작업에 필요한 원문 일부만 유지 |
| 상관 관계 | UUID `trace_id`를 LangGraph `thread_id`와 LangSmith metadata에 공통 사용 | UI trace와 앱 로그를 연결 |
| 복구 | 단일 Worker 실패는 오류 결과로 남기고 다른 Worker를 계속 실행; 품질 미달은 제한된 재조사 | 자동 복구 범위는 task 단위이며, 프로세스 재시작 후 이어하기 위한 영구 checkpointer는 구성하지 않음 |
| 동시 쓰기 | profile·company·evidence·follow-up 결과별 reducer 사용 | fan-out 결과의 누락·덮어쓰기 방지 |
| 종료 보장 | `max_steps`, `max_evaluation_rounds`, retrieval retry 상한 | 반복 재조사와 검색 루프를 제한 |

---

## 8. 프로젝트 구조

```
capstone-v1/
├── main.py                    # 파이프라인 진입점 (환경점검 → GUI 파일선택 → 실행 → 보고서 자동 열기)
├── src/
│   ├── schemas.py             # Pydantic 모델 (Company, Evidence, ProfileResult, State 등)
│   ├── graph.py               # LangGraph 그래프 정의 (노드·엣지·조건부 분기·Send)
│   ├── nodes/
│   │   └── core.py            # 핵심 노드 함수 (load_inputs ~ generate_report)
│   ├── agents/
│   │   └── profile.py         # 5개 Profile Agent 노드 (LLM + Structured Output)
│   └── rag/
│       ├── ingest.py          # PDF 파싱 → 청킹 → FAISS 인덱스 구축
│       └── vector_store.py    # FAISS 벡터 검색 (retrieve_top_k)
├── configs/
│   └── profiles.yaml          # 5개 에이전트 가중치·성향 설정
├── data/
│   ├── raw/                   # 분석할 기업 문서 (PDF, TXT, DOCX)
│   └── processed/             # FAISS 인덱스 저장소
├── output/                    # 자동 생성된 투자 보고서
├── tests/                     # 단위 테스트
└── pyproject.toml             # 의존성 관리 (uv)
```

---

## 9. 핵심 기술 스택 및 코드 해설

### 9-1. 사용 기술 스택

| 기술 | 역할 | 버전 |
|---|---|---|
| **LangChain** | LLM 체이닝, 프롬프트 템플릿, Structured Output | `>=1.4` |
| **LangGraph** | 상태 기반 Multi-Agent 그래프 오케스트레이션 | `>=1.2` |
| **OpenAI GPT-4o-mini** | 투자 판단 LLM (Structured Output 지원) | — |
| **FAISS** | 벡터 유사도 검색 (로컬 인덱스) | `faiss-cpu` |
| **HuggingFace Sentence Transformers** | 로컬 문서 임베딩 (`intfloat/multilingual-e5-base`) | — |
| **Pydantic v2** | State Schema / Agent 출력 스키마 강제 | `>=2.13` |
| **Rich** | 터미널 대시보드 UI (스피너, 패널, 프로그레스바) | `>=14.3` |
| **PyPDF** | PDF 문서 파싱 | `>=6.19` |

### 9-2. LangChain — Prompt ➜ LLM ➜ Structured Output 체이닝

Profile Agent 내부에서는 LangChain의 **LCEL(LangChain Expression Language)** 파이프 연산자(`|`)를 사용해 프롬프트 → LLM → 구조화 출력을 하나의 체인으로 연결합니다.

```python
# src/agents/profile.py 에서 발췌

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. LLM + Structured Output 바인딩
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(LLMProfileOutput)  # Pydantic 스키마 강제

# 2. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다. ..."),
    ("user", "[기업명]: {company_name}\n[도메인별 점수]\n{domain_scores_text}\n...")
])

# 3. LCEL 파이프 체이닝: Prompt → LLM (자동으로 Pydantic 객체 반환)
chain = prompt | structured_llm

# 4. 체인 실행 — LLM이 자유 텍스트가 아닌 Pydantic 객체를 반환
result: LLMProfileOutput = chain.invoke({
    "profile_name": "기술중심형",
    "company_name": "ACCURE",
    "domain_scores_text": "- technology: 90.0\n- market: 80.0\n...",
    "evidence_text": "- [TECH-01] ACCURE는 BMS 연동 기반의..."
})
# result.recommendation  →  "추천"
# result.due_diligence_questions  →  ["독립 성능 검증 보고서 확보 여부", ...]
```

> **핵심 포인트**: `with_structured_output()`을 사용하면 LLM이 자유 텍스트 대신 **Pydantic 스키마에 맞는 JSON 객체**를 반환합니다. 덕분에 5개 에이전트의 출력 형식이 100% 동일하게 유지되어, 후속 종합(Synthesis) 단계에서 안정적으로 점수를 비교할 수 있습니다.

### 9-3. LangGraph — StateGraph 기반 Multi-Agent 오케스트레이션

LangGraph는 LangChain 위에 구축된 **상태 머신(State Machine) 프레임워크**입니다. 노드(함수)와 엣지(전이 규칙)를 정의하면 복잡한 분기·반복·병렬 실행을 선언적으로 관리할 수 있습니다.

```python
# src/graph.py 에서 발췌

from langgraph.graph import StateGraph, END
from langgraph.constants import Send

workflow = StateGraph(InvestmentAgentState)   # TypedDict 기반 상태 정의

# ── 노드 등록 (각 노드는 State → dict 를 반환하는 순수 함수) ──
workflow.add_node("load_inputs", load_inputs)
workflow.add_node("build_questions", build_questions)
workflow.add_node("retrieve_evidence", retrieve_evidence)
workflow.add_node("score_common", score_common)
workflow.add_node("profile_agent", profile_agent_node)
workflow.add_node("synthesize_company", synthesize_company)
# ... (총 14개 노드)

# ── 직선 엣지: A 끝나면 B로 ──
workflow.add_edge("load_inputs", "select_company")
workflow.add_edge("select_company", "build_questions")
workflow.add_edge("build_questions", "retrieve_evidence")
```

#### 조건부 분기 (Conditional Edge)

근거가 충분한지에 따라 **다음 노드를 동적으로 선택**합니다.

```python
# 라우터 함수: State를 받아 다음 노드 이름(문자열)을 반환
def route_evidence_check(state) -> Literal["score_common", "rewrite_query", "record_unknown"]:
    if is_sufficient:
        return "score_common"        # 충분 → 점수 계산으로
    elif total_retry < 2:
        return "rewrite_query"       # 부족 → 질의 수정 후 재검색
    else:
        return "record_unknown"      # 2회 초과 → 미확인 기록

# 그래프에 조건부 엣지로 등록
workflow.add_conditional_edges(
    "validate_evidence",             # 출발 노드
    route_evidence_check,            # 라우터 함수
    {
        "score_common": "score_common",
        "rewrite_query": "rewrite_query",
        "record_unknown": "record_unknown"
    }
)

# 재검색 루프: rewrite → retrieve → validate → (다시 판단)
workflow.add_edge("rewrite_query", "retrieve_evidence")
```

#### 병렬 실행 — `Send` API (Fan-out)

Planner가 만든 구조화 task를 **동시에** 실행하기 위해 LangGraph의 `Send` API를 사용합니다. 프로필 수는 코드에 고정하지 않고, 설정에서 불러와 State에 저장한 task 개수로 정합니다.

```python
from langgraph.constants import Send

def route_profile_workers(state):
    """Planner가 State에 저장한 작업만 worker에 전달"""
    return [
        Send("profile_worker", {
            "task": task,
            "domain_scores": state["domain_scores"],              # 공통 점수 (동일)
            "validated_evidence": state["validated_evidence"],     # 검증된 근거 (동일)
            "current_company": state["current_company"]           # 현재 기업 (동일)
        })
        for task in state["structured_tasks"]  # → 계획된 수만큼 Send 객체 반환
    ]

# score_common 뒤 planner가 계획을 State에 저장하고, 그 다음 worker fan-out 실행
workflow.add_edge("score_common", "plan_profile_tasks")
workflow.add_conditional_edges("plan_profile_tasks", route_profile_workers, ["profile_worker"])
```

> **핵심 포인트**: `Send`는 동일한 노드(`profile_agent`)를 **서로 다른 입력**으로 여러 번 동시에 띄우는 LangGraph의 병렬 실행 메커니즘입니다. 5개의 결과는 State의 `profile_results` 필드에 `Annotated[List, operator.add]` Reducer를 통해 자동으로 누적됩니다.

#### 기업 반복 루프 (Company Loop)

```python
def route_next_company(state) -> Literal["select_company", "compare_companies"]:
    """current_index가 기업 수 미만이면 다음 기업으로, 아니면 비교 단계로"""
    if state["current_index"] < len(state["selected_companies"]):
        return "select_company"       # 다음 기업으로 되돌아감
    else:
        return "compare_companies"    # 모든 기업 완료 → 비교

workflow.add_conditional_edges("save_company_result", route_next_company, {...})
```

#### 최종 컴파일 및 실행

```python
graph = workflow.compile()   # 그래프를 실행 가능한 Runnable 객체로 변환

# 스트리밍 실행: 각 노드가 완료될 때마다 이벤트 발생
for event in graph.stream(initial_state):
    for node_name, node_output in event.items():
        print(f"노드 완료: {node_name}")
```

### 9-4. 기술 적용 요약도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py (진입점)                             │
│  Rich Console UI  ·  환경점검  ·  Tkinter GUI  ·  보고서 자동 열기  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  graph.stream(initial_state)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                             │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │load_input│───▶│build_    │───▶│retrieve_ │───▶│validate_ │      │
│  │          │    │questions │    │evidence  │    │evidence  │      │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│                                       ▲               │            │
│                                       │         ┌─────┴─────┐      │
│                                  rewrite_query  │score_common│      │
│                                                 └─────┬─────┘      │
│                                                       │            │
│                               ┌───────────────────────┼──────┐     │
│                               │   Send API (Fan-out)  │      │     │
│                               │  ┌────┐┌────┐┌────┐┌────┐┌────┐   │
│                               │  │기술││안정││성장││균형││사업│   │
│                               │  └─┬──┘└─┬──┘└─┬──┘└─┬──┘└─┬──┘   │
│                               └────┼─────┼─────┼─────┼─────┼──┘   │
│                                    └─────┴──┬──┴─────┘     │      │
│     LangChain LCEL:                         ▼              │      │
│     prompt | structured_llm     ┌──────────────────┐       │      │
│     (각 Agent 내부에서 실행)     │synthesize_company│       │      │
│                                 └────────┬─────────┘       │      │
│                                          ▼                 │      │
│                                 ┌──────────────────┐       │      │
│                                 │ generate_report  │       │      │
│                                 └────────┬─────────┘       │      │
│                                          ▼                 │      │
│                                        [END]               │      │
└─────────────────────────────────────────────────────────────┘      │
                                                                     │
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Layer                                   │
│  ingest.py (PyPDF → Chunk → OpenAI Embedding → FAISS 저장)         │
│  vector_store.py (FAISS 로드 → Top-K 유사도 검색)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 설치 및 실행 방법

### Step 1. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 API 키를 입력합니다.
```env
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=SKALA
```

### Step 2. 시스템 라이브러리 설치 (PDF 자동 생성용)
고품질 PDF 리포트를 자동 생성하기 위해 운영체제별로 아래 필수 라이브러리를 설치해주세요.
- **Mac (Homebrew)**:
  ```bash
  brew install pango libffi cairo
  ```
- **Windows / Linux (Ubuntu)**:
  WeasyPrint 공식 문서를 참고하여 Pango 및 Cairo 런타임을 설치해주세요. (Ubuntu: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0`)

### Step 3. 패키지 설치 및 실행
파이썬 패키지를 설치하고 파이프라인을 실행합니다.
```bash
uv run python main.py
```

실행하면 아래 4단계가 **자동으로** 순서대로 진행됩니다.

1. **환경 점검** — OpenAI API 연결과 LangSmith 설정을 자동 확인합니다. 실패 시 원인을 알려주고 안전하게 종료합니다.
2. **파일 선택** — macOS 기본 파일 선택 창(GUI)이 팝업됩니다. 분석할 기업의 PDF/TXT 파일을 마우스로 선택하면 `data/raw/`로 자동 복사됩니다. (기존 파일이 있으면 그대로 사용할지 물어봅니다.)
3. **Multi-Agent 파이프라인 구동** — 문서 임베딩 → 5대 영역 평가 → 5개 에이전트 병렬 분석 → 기업 비교 → 보고서 생성이 자동으로 진행됩니다.
4. **보고서 자동 열기** — 완료 즉시 생성된 마크다운 보고서 파일이 화면에 열립니다.

---

## 11. 실행 예시

`uv run python main.py`를 실행하면 터미널에 다음과 같은 대시보드가 출력됩니다.

```text
╭────────────────────────────────────────────────────╮
│ 배터리 스타트업 투자평가 Multi-Agent 파이프라인 🚀 │
╰────────────────────────────────────────────────────╯

🔍 1단계: API 및 환경 변수 점검
  ✔ OpenAI API 연결 성공
  ✔ LangSmith 트레이싱 활성화 (프로젝트: SKALA)
=> 환경 점검 완료!

📂 2단계: 분석 대상 문서 로드
기존 data/raw 폴더에 파일이 있습니다. 새로 파일을 선택하시겠습니까? [y/n]: n
✔ 기존 폴더의 파일을 그대로 사용합니다.

🧠 3단계: 문서 스캔 및 지식 베이스 구축
📄 읽은 파일: ACCURE_RAG_source_pack.pdf (기업명: ACCURE) - 6 페이지/섹션
✂️ 청킹 완료: 총 14 개의 청크 생성 (크기: 500, 오버랩: 50)
🧠 로컬 HuggingFace 임베딩 및 FAISS DB 생성 중...
✅ DB 저장 완료!
✔ 문서 임베딩 완료

🤖 4단계: Multi-Agent 투자 심사 진행
 ➔ 📂 분석 타겟: ACCURE

🎯 타겟 기업 설정됨: ACCURE
 ➔ ✨ Sub-Agent 작업 완료: build_questions
 ➔ ✨ Sub-Agent 작업 완료: retrieve_evidence
 ➔ ✨ Sub-Agent 작업 완료: score_common
 ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료
 ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료
 ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료
 ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료
 ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료
 ➔ ⚖️ 최종 투자 판정: 투자검토 추천 (84.1점)
 ➔ 📊 기업 비교 및 최종 테이블 작성 완료
 ➔ 📑 투자 심사 마크다운 보고서 생성 완료
⠇ Multi-Agent 파이프라인 실행 완료!

==================================================
╭──────────────────────────────────────────────────────────────────────────────╮
│ ✅ 파이프라인 실행 완료                                                      │
│ 최종 투자 심사 보고서가 final_multi_agent_report.md에 저장되었습니다.        │
╰──────────────────────────────────────────────────────────────────────────────╯

📄 생성된 투자 보고서를 자동으로 엽니다...
```

---

## 12. 출력 보고서 예시

파이프라인이 완료되면 `output/final_multi_agent_report.md`에 다음과 같은 보고서가 자동 생성됩니다.

```markdown
# 투자 심사 요약 보고서 (Multi-Agent RAG)

## 1. 종합 비교표
| 기업 | 기술 | 시장 | 재무 | 안정성 | 팀·사업 | 근거충족률 | 최종 점수 | 판정 |
|---|---|---|---|---|---|---|---|---|
| ACCURE | 90.0 | 80.0 | 70.0 | 84.0 | 96.0 | 85.0% | 84.1 | 투자검토 추천 |

## 2. 기업별 상세
### ACCURE (투자검토 추천)
- **평균 점수**: 84.1점 (표준편차: 1.6)
- **근거 충족률**: 85.0%

#### Profile Agent 결과
- **기술중심형 (손연우)**: 86.3점 — 배터리 관리 솔루션에서 뛰어난 기술력을 보유하며 
  BMS/ESS 연동의 기술적 타당성이 매우 높음. (강력 추천)
- **사업성중심형 (정수민)**: 84.7점 — 팀의 경험과 배경이 강력하여 뛰어난 사업 실행력이 
  기대됨. (추천)
- **성장형 (김대훈)**: 84.1점 — 배터리 분석 시장에서의 성장 가능성이 높음. (추천)
- **균형형 (이연주)**: 84.0점 — 기술력과 시장 입지는 양호하나 재무적 측면 보완 필요. 
  (조건부 추천)
- **안정형 (강도희)**: 81.3점 — 기술은 훌륭하나 재무적 안정성과 고객 실적의 독립적 
  검증이 부족함. (추가 실사 요망)
```

> **Insight**: 기술중심형 에이전트는 `86.3점`의 고득점을 부여한 반면, 재무와 데이터 출처를 엄격하게 따지는 안정형 에이전트는 `81.3점`이라는 가장 낮은 점수를 부여했습니다. 이처럼 의도된 **"관점의 충돌"**을 통해 투자 리스크를 다각도에서 검증하는 것이 본 시스템의 핵심입니다.

---

## 13. 구현·검증 계획과 담당

| 단계 | 구현 내용 | 완료 조건 |
|---|---|---|
| 1. 데이터 | 63개 출처에서 200쪽 이하 문서 선별·파싱·metadata 부착 | `source_id`·`page`·`source_type` 추적 가능 |
| 2. Retrieval | bge-m3/e5 인덱스와 45문항 평가 | Hit@5·MRR@5·지연시간 비교 |
| 3. Evidence RAG | retrieve·validate·rewrite·unknown 노드 | 재검색 2회 종료·인용 연결 |
| 4. Scoring | 공통점수·근거충족률·가중치 계산 | 단위테스트 계산오차 0 |
| 5. Agents | 5개 ProfileConfig와 병렬 fan-out | 스키마 준수·5개 결과 reducer |
| 6. Synthesis | 평균·표준편차·합의·이견·추가실사 | 3개 기업 비교 가능 |
| 7. Report | 5쪽 보고서 구조화·인용 검증 | REFERENCE 역참조·형식 오류 0 |

**구성원 담당**

| 구성원 | Agent 담당 | 구현·검증 산출물 |
|---|---|---|
| 손연우 | 기술중심형 | 기술·연동 평가항목, BMS/ESS 구조, 기술 Agent 테스트 |
| 강도희 | 안정형 | 투자단계·법률·안전·데이터 권리 근거, 안정 Agent 테스트 |
| 김대훈 | 성장형 | AI 모델·데이터·성능·확장성, 성장 Agent 테스트 |
| 이연주 | 균형형 | 열화·화학계·실운용 타당성, 공통 기준·편차 검증 |
| 정수민 | 사업성중심형 | Scorecard·시장·고객·통계·45문항 평가셋 |

---

**설계 범위와 한계**
- 본 시스템은 공개문서 기반 투자검토 보조도구이며 실제 투자·법률·안전 실사를 대체하지 않는다.
- 비공개 매출·고객계약·모델데이터는 unknown으로 남기고 데이터룸 요청으로 연결한다.
- Embedding 선정과 Agent 임계값은 설계 가설이며 실제 평가 결과로 보정한다.

---

## 14. AI Agent 실습: Orchestrator-Workers

이 브랜치는 기존 RAG 분석 흐름과 보고서 구성을 유지하며 Orchestrator-Workers 패턴을 적용합니다. PDF는 기존 보고서에 근거 발췌와 품질 평가를 더해 7쪽이며, 안내된 최대 10쪽 이내입니다.

### 작업 흐름

1. Planner가 설정된 Profile을 회사별 구조화 작업으로 State에 기록합니다.
2. `Send`가 저장된 작업 수에 맞춰 worker를 동적으로 실행합니다. 각 worker는 기존 근거 평가 모듈을 사용합니다.
3. worker 결과는 reducer로 누적되고, 기존 synthesizer가 관점별 점수·근거 충족률·판정 유보를 종합합니다.
4. 보고서 생성 뒤 규칙 검사와 Judge를 결합해 Groundedness, Neutrality, Bias control을 평가합니다.
5. 미달이면 Judge 제안 작업을 검증해 실행하고, 제안이 없거나 유효하지 않으면 실패 항목에 맞춘 대체 질문을 계획합니다. 로컬 RAG 검색과 해당 Profile 재평가 후 보고서를 갱신합니다. 최대 2회 평가하며, 즉 후속 조사 루프는 최대 1회입니다.

### 실습 가이드 정합성

| 가이드 요구 | 적용 위치 | 선택·상태 |
|---|---|---|
| 구조화 계획을 먼저 State에 저장 | `plan_profile_tasks` → `structured_tasks` | 프로필 설정마다 회사·관점·focus를 task로 생성 |
| 계획 이후 동적 fan-out | `route_profile_workers` | task 목록을 `Send`로 보내므로 worker 수는 코드에 고정하지 않음 |
| 병렬 결과를 Synthesizer에서 통합 | `profile_worker` → `synthesize_company` | 기존 프로필 판단을 누적한 뒤 기업 단위로 집계 |
| Worker 실패 대안 | `profile_agent_node`, `followup_worker` | 오류 결과를 기록하고 다른 Worker를 계속 진행 |
| State 필드 충돌·무한 증가 제어 | `src/schemas.py` reducer | 결과별 병합 규칙과 기업당 근거 수·길이 상한 적용 |
| 보고서 품질 평가와 미달 시 Loop | `evaluate_report_quality` → `plan_followup_tasks` | 결정 규칙 + LLM Judge의 hybrid 평가, 최대 2회 평가 |
| trace와 관측성 | `main.py`, `src/agents/orchestrator.py` | `trace_id`를 `thread_id`·LangSmith metadata·분기 로그에 전달 |
| 복구 경계 | worker 상태·오류 결과·평가 라운드 | task 실패는 계속 처리. 프로세스가 종료된 뒤 재개하는 영구 checkpointer는 아직 없음 |

### State와 복구

- **Payload:** 계획된 worker task, 기업별 최대 160개 발췌 근거, 평가 결과, 보고서.
- **Control:** `trace_id`, worker별 상태, `step_count`, 평가 라운드 및 상한.
- 기업별 근거는 출처와 평가 항목 기준으로 중복 제거하고 각 발췌를 900자로 제한합니다. 원문 전체를 반복 저장하지 않습니다.
- worker 하나가 실패하면 오류 상태와 해당 Profile의 판단 유보를 기록하고 나머지 결과를 계속 처리합니다.
- Judge 오류는 통과로 간주하지 않습니다. 대체 질문으로 후속 조사하고 한 번 재평가한 뒤, 여전히 미달이면 PDF에 사람 검토 필요로 표시합니다.

### 실행 산출물

- Markdown: `output/final_multi_agent_report.md`
- PDF: `output/final_report.pdf` (기존 보고서 구성에 품질 평가 페이지를 추가, 최대 10쪽)
- LangSmith metadata: `trace_id`, `pattern=orchestrator-workers`
- LangSmith trace screenshots: 동적 worker 분기를 포함한 실제 실행 후 `tracing-1.png`, `tracing-2.png` 순서로 저장해야 합니다.

### Contributors

- **강도희** — 구조화된 작업 계획, 동적 worker 분배, State reducer·복구 메타데이터, 보고서 품질 평가와 제한된 후속 조사 흐름.
