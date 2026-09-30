from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from ...schemas import Domain, FinalDecision, ProfileResult, ResultCategory
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# 성장형 Agent의 독자 평가항목입니다. 이 가중치의 합은 1.0이며,
# 산출된 custom_score 전체가 최종 투자점수의 30%를 차지합니다.
GROWTH_PARAMETER_WEIGHTS: Dict[str, float] = {
    "market_growth": 0.25,
    "customer_conversion": 0.25,
    "revenue_growth": 0.20,
    "scalability": 0.15,
    "data_network_effect": 0.10,
    "execution_capability": 0.05,
}

GROWTH_PARAMETER_LABELS: Dict[str, str] = {
    "market_growth": "시장 성장률·시장 확대 가능성",
    "customer_conversion": "파일럿의 유료 고객 전환 가능성",
    "revenue_growth": "매출·ARR 성장성",
    "scalability": "제품·지역·고객군 확장성",
    "data_network_effect": "데이터 축적에 따른 네트워크 효과",
    "execution_capability": "성장 전략 실행역량",
}

GROWTH_EVIDENCE_THRESHOLD = 0.80
GROWTH_CORE_PARAMETERS = {"customer_conversion", "revenue_growth"}

GrowthEvidenceStatus = Literal[
    "SUPPORTED",
    "PARTIAL",
    "UNKNOWN",
    "NOT_APPLICABLE",
    "CONTRADICTED",
]

def _clamp_score(value: float) -> float:
    """점수를 0~100 범위로 제한합니다."""
    return round(max(0.0, min(100.0, value)), 2)


class GrowthParameterAssessment(BaseModel):
    """성장형 파라미터 하나의 근거 상태와 평가 결과입니다."""

    status: GrowthEvidenceStatus
    score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="SUPPORTED·PARTIAL·CONTRADICTED인 경우에만 부여하는 점수",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="해당 파라미터 판정에 사용한 Source ID",
    )
    reason: str = Field(description="근거 상태와 점수 판정 이유")


class GrowthParameterAssessments(BaseModel):
    """공통 도메인 점수와 독립적으로 산출하는 성장형 전용 평가입니다."""

    market_growth: GrowthParameterAssessment
    customer_conversion: GrowthParameterAssessment
    revenue_growth: GrowthParameterAssessment
    scalability: GrowthParameterAssessment
    data_network_effect: GrowthParameterAssessment
    execution_capability: GrowthParameterAssessment


class GrowthCustomEvaluation(BaseModel):
    parameter_assessments: GrowthParameterAssessments
    rationale: str = Field(description="제공된 근거만 사용한 성장형 세부점수 산정 이유")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="성장형 점수 산정에 사용한 Source ID"
    )
    unknown_items: List[str] = Field(
        default_factory=list, description="성장형 평가에 필요하지만 확인되지 않은 항목"
    )


class GrowthEvidenceInsufficientError(ValueError):
    """성장형 점수를 산출할 만큼 근거가 충분하지 않을 때 발생합니다."""


def calculate_growth_evidence_coverage(
    assessments: Dict[str, GrowthParameterAssessment],
) -> Tuple[float, List[str], List[str]]:
    """N/A를 제외한 가중치 기준 근거 충족률과 부족·제외 항목을 계산합니다."""
    coverage_factor = {
        "SUPPORTED": 1.0,
        "CONTRADICTED": 1.0,
        "PARTIAL": 0.5,
        "UNKNOWN": 0.0,
    }
    applicable_weight = 0.0
    covered_weight = 0.0
    insufficient_parameters: List[str] = []
    not_applicable_parameters: List[str] = []

    for key, weight in GROWTH_PARAMETER_WEIGHTS.items():
        assessment = assessments[key]
        if assessment.status == "NOT_APPLICABLE":
            not_applicable_parameters.append(key)
            continue

        applicable_weight += weight
        covered_weight += weight * coverage_factor[assessment.status]

        if assessment.status in {"UNKNOWN", "PARTIAL"}:
            insufficient_parameters.append(key)

    if applicable_weight == 0:
        return 0.0, insufficient_parameters, not_applicable_parameters

    return (
        round(covered_weight / applicable_weight, 4),
        insufficient_parameters,
        not_applicable_parameters,
    )


def calculate_growth_custom_score(
    assessments: Dict[str, GrowthParameterAssessment],
) -> float:
    """점수 산출 조건을 통과한 확인 항목만 재가중해 성장형 점수를 계산합니다."""
    score_total = 0.0
    scored_weight = 0.0

    for key, weight in GROWTH_PARAMETER_WEIGHTS.items():
        assessment = assessments[key]
        if assessment.status in {"NOT_APPLICABLE", "UNKNOWN"}:
            continue
        if assessment.score is None:
            raise ValueError(f"점수가 필요한 성장형 파라미터입니다: {key}")

        score_total += assessment.score * weight
        scored_weight += weight

    if scored_weight == 0:
        raise GrowthEvidenceInsufficientError("채점 가능한 성장형 파라미터가 없습니다.")

    return _clamp_score(score_total / scored_weight)

from ._evidence import check_profile_evidence, attach_evidence_check, profile_decision


# rubric.yaml의 항목 ID 기준. 담당자가 평가 범위에 맞게 조정할 수 있습니다.
# 핵심 확인 항목: 시장 성장성, 고객 도입·계약
EVIDENCE_POLICY = {
    "min_coverage": 80.0,
    "critical_items": ("mkt_01", "fin_02"),
}

def run_growth_agent(state: Any) -> ProfileResult:
    """김대훈 팀원이 담당하는 '성장형' 에이전트 커스텀 로직"""
    
    # ==========================================
    # [1] 공통 파라미터 및 점수 (70% 반영) - 수정 지양
    # ==========================================
    profile_id = "growth_focused"
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]

    # 근거 부족이면 공통/커스텀 점수를 계산하지 않고 정상적으로 판단 유보를 반환합니다.
    evidence_check = check_profile_evidence(state, config, **EVIDENCE_POLICY)
    if not evidence_check.sufficient:
        return evidence_check.hold_result(profile_name)
    
    # 공통 점수 계산 (도메인 점수 * 가중치)
    common_score = calculate_profile_score(state["domain_scores"], weights)
    
    # ==========================================
    # [2] 팀원 커스텀 평가 로직 (30% 반영) - 자유롭게 수정!
    # ==========================================
    confirmed_source_ids = set(evidence_check.source_ids)
    validated_evidence = [
        evidence
        for evidence in state.get("validated_evidence", [])
        if evidence.source_id in confirmed_source_ids
    ]

    evidence_text = ""
    for ev in validated_evidence:
        evidence_text += (
            f"- [ID: {ev.item_id}, Source: {ev.source_id}] "
            f"(등급: {ev.evidence_grade}, 기업 직접성: {ev.is_direct}): "
            f"{ev.content}\n"
        )
    if not evidence_text:
        evidence_text = "확인된 근거가 없습니다."

    current_company = state.get("current_company")
    company_name = current_company.name if current_company else "Unknown Startup"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if validated_evidence:
        growth_scoring_llm = llm.with_structured_output(GrowthCustomEvaluation)
        growth_scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 성장주 투자전략을 담당하는 심사역입니다.

아래 제공되는 검증 근거만 사용하여 성장형 전용 6개 지표를 각각 0~100점으로 평가하세요.
공통 도메인 점수나 다른 Agent 점수는 제공되지 않으며, 추정해서도 안 됩니다.

성장형 전용 평가항목과 세부평가 내 가중치:
- 시장 성장률·시장 확대 가능성: 25%
- 파일럿의 유료·반복 고객 전환: 25%
- 매출·ARR 성장성과 반복매출: 20%
- 제품·지역·고객군 확장성: 15%
- 데이터 축적에 따른 네트워크 효과: 10%
- 성장전략 실행역량: 5%

채점 원칙:
1. 회사의 주장보다 A/B급 근거와 기업 직접성이 높은 자료를 우선합니다.
2. 수치·계약·고객명이 확인된 실적을 단순 시장 전망보다 높게 평가합니다.
3. 근거가 충분하면 SUPPORTED, 일부만 확인되면 PARTIAL, 없으면 UNKNOWN으로 분류합니다.
4. NOT_APPLICABLE은 평가 전에 제품·사업 범위상 적용 대상이 아님이 근거로 확인된 경우에만 사용합니다.
5. UNKNOWN과 NOT_APPLICABLE에는 점수를 부여하지 않습니다.
6. 부정 또는 상충 근거가 충분하면 CONTRADICTED로 분류하고 낮은 점수를 부여합니다.
7. evidence_ids와 supporting_evidence_ids에는 입력에 존재하는 Source ID만 사용합니다.
8. 파일럿의 유료 고객 전환과 매출·ARR 성장성은 점수 산출에 필요한 핵심 지표입니다."""),
            ("user", """
[기업명]
{company_name}

[성장형 평가에 사용할 검증 근거]
{evidence_text}
""")
        ])
        growth_evaluation: GrowthCustomEvaluation = (
            growth_scoring_prompt | growth_scoring_llm
        ).invoke({
            "company_name": company_name,
            "evidence_text": evidence_text,
        })
    else:
        growth_evaluation = GrowthCustomEvaluation(
            parameter_assessments=GrowthParameterAssessments(
                market_growth=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
                customer_conversion=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
                revenue_growth=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
                scalability=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
                data_network_effect=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
                execution_capability=GrowthParameterAssessment(
                    status="UNKNOWN", reason="검증된 근거 없음"
                ),
            ),
            rationale="검증된 성장 근거가 없어 모든 성장형 세부지표를 미확인 처리했습니다.",
            supporting_evidence_ids=[],
            unknown_items=list(GROWTH_PARAMETER_LABELS.values()),
        )

    growth_parameter_assessments = {
        key: value
        for key, value in growth_evaluation.parameter_assessments
    }
    (
        growth_evidence_coverage,
        insufficient_parameters,
        not_applicable_parameters,
    ) = calculate_growth_evidence_coverage(growth_parameter_assessments)

    missing_core_parameters = [
        key
        for key in GROWTH_CORE_PARAMETERS
        if growth_parameter_assessments[key].status not in {
            "SUPPORTED",
            "CONTRADICTED",
        }
    ]

    if (
        growth_evidence_coverage < GROWTH_EVIDENCE_THRESHOLD
        or missing_core_parameters
    ):
        insufficient_labels = [
            GROWTH_PARAMETER_LABELS[key]
            for key in insufficient_parameters
        ]
        core_labels = [
            GROWTH_PARAMETER_LABELS[key]
            for key in missing_core_parameters
        ]
        coverage_percent = round(growth_evidence_coverage * 100, 2)
        unknown_items = list(dict.fromkeys(
            growth_evaluation.unknown_items
            + [f"성장형 근거 부족: {label}" for label in insufficient_labels]
            + [f"성장형 핵심 지표 미확인: {label}" for label in core_labels]
        ))
        return ProfileResult(
            profile_name=profile_name,
            weighted_score=None,
            decision=FinalDecision.HOLD_RAG,
            evidence_coverage=coverage_percent,
            result_category=(
                ResultCategory.PARTIAL_EVIDENCE
                if coverage_percent > 0
                else ResultCategory.INSUFFICIENT_EVIDENCE
            ),
            supporting_evidence_ids=list(dict.fromkeys(
                growth_evaluation.supporting_evidence_ids
            )),
            unknown_items=unknown_items,
            recommendation=(
                "점수 없음 / 근거 부족 판단 유보 | "
                f"성장형 세부 근거충족률 {coverage_percent:.1f}% "
                f"(기준 {GROWTH_EVIDENCE_THRESHOLD * 100:.0f}%). "
                f"미확인 핵심 지표: {', '.join(core_labels) or '없음'}"
            ),
            due_diligence_questions=list(dict.fromkeys(
                [f"{label}을 확인할 수 있는 계약·매출·고객 자료를 제출할 것" for label in (
                    core_labels or insufficient_labels
                )]
            )),
        )

    custom_score = calculate_growth_custom_score(growth_parameter_assessments)
    
    # 최종 점수 산출 (공통 70% + 커스텀 30%)
    final_score = _clamp_score((common_score * 0.7) + (custom_score * 0.3))
    
    # ==========================================
    # [3] 팀원 커스텀 LLM 프롬프트 - 자유롭게 수정!
    # ==========================================
    structured_llm = llm.with_structured_output(LLMProfileOutput)
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.
        
당신의 투자 성향 및 최우선 판단 기준:
{focus}

제공된 '도메인별 평가 점수', '성장형 세부지표'와 '확인된 근거'를 바탕으로 성장 잠재력을 평가하세요.

평가 원칙:
1. 시장규모 전망보다 실제 고객 전환, 반복매출, 확장 증거를 우선합니다.
2. 근거가 없는 낙관적 추론은 unknown_items와 실사 질문으로 분리합니다.
3. supporting_evidence_ids와 contrary_evidence_ids에는 제공된 Source ID만 사용합니다.
4. 추천 의견은 계산된 최종점수와 모순되지 않아야 합니다.
5. 성장성이 높더라도 고객 집중, 현금소진, 규제·안전 위험을 반대 근거로 명시합니다."""),
        ("user", """
[기업명]: {company_name}
[당신의 최종 산출 점수 (공통 70% + 커스텀 30%)]: {final_score} / 100
[공통 평가 점수]: {common_score} / 100
[성장형 세부 평가 점수]: {custom_score} / 100
[성장형 근거 충족률]: {growth_evidence_coverage}%

[성장형 세부 파라미터]
{growth_parameter_assessments_text}

[평가 비대상 파라미터]
{not_applicable_parameters_text}

[성장형 독립평가 근거 요약]
{growth_rationale}

[도메인별 공통 점수]
{domain_scores_text}

[확인된 주요 근거]
{evidence_text}
""")
    ])
    
    chain = prompt | structured_llm
    
    domain_scores_text = "\n".join(
        f"- {key.value if isinstance(key, Domain) else key}: {value}"
        for key, value in state["domain_scores"].items()
    )
    growth_parameter_assessments_text = "\n".join(
        f"- {GROWTH_PARAMETER_LABELS[key]}: "
        f"상태={assessment.status}, "
        f"점수={assessment.score if assessment.score is not None else '미산출'}, "
        f"가중치={GROWTH_PARAMETER_WEIGHTS[key] * 100:.0f}%, "
        f"근거={assessment.evidence_ids or '없음'}"
        for key, assessment in growth_parameter_assessments.items()
    )
    not_applicable_parameters_text = ", ".join(
        GROWTH_PARAMETER_LABELS[key]
        for key in not_applicable_parameters
    ) or "없음"
    result: LLMProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "final_score": final_score,
        "common_score": common_score,
        "custom_score": custom_score,
        "growth_evidence_coverage": round(growth_evidence_coverage * 100, 1),
        "growth_parameter_assessments_text": growth_parameter_assessments_text,
        "not_applicable_parameters_text": not_applicable_parameters_text,
        "growth_rationale": growth_evaluation.rationale,
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text
    })
    
    supporting_evidence_ids = list(dict.fromkeys(
        growth_evaluation.supporting_evidence_ids
        + result.supporting_evidence_ids
    ))
    unknown_items = list(dict.fromkeys(
        growth_evaluation.unknown_items
        + result.unknown_items
    ))

    return attach_evidence_check(ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    ), evidence_check)
