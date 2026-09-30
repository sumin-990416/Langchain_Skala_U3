from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ...schemas import Domain, ProfileResult
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

def _clamp_score(value: float) -> float:
    """점수를 0~100 범위로 제한합니다."""
    return round(max(0.0, min(100.0, value)), 2)


class GrowthParameterScores(BaseModel):
    """공통 도메인 점수와 독립적으로 산출하는 성장형 전용 점수입니다."""

    market_growth: float = Field(ge=0, le=100, description="시장 성장률과 시장 확대 가능성")
    customer_conversion: float = Field(
        ge=0, le=100, description="파일럿에서 유료·반복 고객으로 전환된 증거"
    )
    revenue_growth: float = Field(ge=0, le=100, description="매출·ARR 성장성과 반복매출 증거")
    scalability: float = Field(ge=0, le=100, description="제품·지역·고객군 확장 가능성")
    data_network_effect: float = Field(
        ge=0, le=100, description="데이터 축적이 성능과 진입장벽으로 연결되는 정도"
    )
    execution_capability: float = Field(ge=0, le=100, description="성장전략을 실행한 실적")


class GrowthCustomEvaluation(BaseModel):
    parameter_scores: GrowthParameterScores
    rationale: str = Field(description="제공된 근거만 사용한 성장형 세부점수 산정 이유")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="성장형 점수 산정에 사용한 Source ID"
    )
    unknown_items: List[str] = Field(
        default_factory=list, description="성장형 평가에 필요하지만 확인되지 않은 항목"
    )


def calculate_growth_custom_score(parameter_scores: Dict[str, float]) -> float:
    """독립적으로 평가된 성장형 파라미터를 30% 반영용 점수로 합산합니다."""
    custom_score = sum(
        parameter_scores[key] * weight
        for key, weight in GROWTH_PARAMETER_WEIGHTS.items()
    )
    return _clamp_score(custom_score)


def run_growth_agent(state: Any) -> ProfileResult:
    """김대훈 팀원이 담당하는 '성장형' 에이전트 커스텀 로직"""
    
    # ==========================================
    # [1] 공통 파라미터 및 점수 (70% 반영) - 수정 지양
    # ==========================================
    profile_id = "성장형"
    try:
        config = load_profile_config(profile_id)
    except KeyError:
        # 현재 profiles.yaml의 식별자가 growth_focused인 경우에도
        # growth.py 단독으로 정상 동작하도록 호환합니다.
        config = load_profile_config("growth_focused")
    profile_name = config["name"]
    weights = config["weights"]
    
    # 공통 점수 계산 (도메인 점수 * 가중치)
    common_score = calculate_profile_score(state["domain_scores"], weights)
    
    # ==========================================
    # [2] 팀원 커스텀 평가 로직 (30% 반영) - 자유롭게 수정!
    # ==========================================
    validated_evidence = state.get("validated_evidence", [])

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
3. 해당 항목의 근거가 없으면 임의로 추정하지 말고 50점을 부여한 뒤 unknown_items에 기록합니다.
4. 부정 또는 상충 근거가 있으면 점수를 하향하고 rationale에 명시합니다.
5. supporting_evidence_ids에는 입력에 존재하는 Source ID만 사용합니다."""),
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
            parameter_scores=GrowthParameterScores(
                market_growth=50.0,
                customer_conversion=50.0,
                revenue_growth=50.0,
                scalability=50.0,
                data_network_effect=50.0,
                execution_capability=50.0,
            ),
            rationale="검증된 성장 근거가 없어 모든 성장형 세부지표를 중립값으로 처리했습니다.",
            supporting_evidence_ids=[],
            unknown_items=list(GROWTH_PARAMETER_LABELS.values()),
        )

    growth_parameter_scores = growth_evaluation.parameter_scores.model_dump()
    custom_score = calculate_growth_custom_score(growth_parameter_scores)
    
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

[성장형 세부 파라미터]
{growth_parameter_scores_text}

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
    growth_parameter_scores_text = "\n".join(
        f"- {GROWTH_PARAMETER_LABELS[key]}: {score:.2f}점 "
        f"(세부평가 내 비중 {GROWTH_PARAMETER_WEIGHTS[key] * 100:.0f}%)"
        for key, score in growth_parameter_scores.items()
    )
    result: LLMProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "final_score": final_score,
        "common_score": common_score,
        "custom_score": custom_score,
        "growth_parameter_scores_text": growth_parameter_scores_text,
        "growth_rationale": growth_evaluation.rationale,
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text
    })

    supporting_evidence_ids = list(dict.fromkeys(
        growth_evaluation.supporting_evidence_ids + result.supporting_evidence_ids
    ))
    unknown_items = list(dict.fromkeys(
        growth_evaluation.unknown_items + result.unknown_items
    ))
    
    return ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    )
