from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ...schemas import ProfileResult, Domain
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ._evidence import check_profile_evidence, attach_evidence_check, profile_decision


BUSINESS_CRITERIA = {
    "revenue_model": {"weight": 0.30, "items": ("fin_01",)},
    "customer_conversion": {"weight": 0.40, "items": ("fin_02",)},
    "team_execution": {"weight": 0.30, "items": ("team_01",)},
}


class BusinessCriterionAssessment(BaseModel):
    score: Optional[Literal[0, 25, 50, 75, 100]] = Field(
        description="자료로 판단할 수 없으면 null, 아니면 근거에 맞는 0/25/50/75/100"
    )
    evidence_ids: List[str] = Field(default_factory=list)
    rationale: str = Field(description="인용한 사실과 점수 기준의 연결")
    missing_information: str = Field(default="")


class BusinessCustomAssessment(BaseModel):
    revenue_model: BusinessCriterionAssessment
    customer_conversion: BusinessCriterionAssessment
    team_execution: BusinessCriterionAssessment


class BusinessProfileOutput(LLMProfileOutput):
    custom_assessment: BusinessCustomAssessment


def calculate_business_custom_score(assessment: BusinessCustomAssessment, source_ids_by_item: Dict[str, List[str]]):
    """해당 평가 항목에서 인용 검증된 ID가 있는 기준만으로 0~100점을 계산합니다."""
    scores, evidence_ids, missing, covered_weight = {}, {}, [], 0.0
    for key, criterion in BUSINESS_CRITERIA.items():
        item = getattr(assessment, key)
        allowed = {source_id for item_id in criterion["items"]
                   for source_id in source_ids_by_item.get(item_id, [])}
        cited = list(dict.fromkeys(item.evidence_ids))
        valid = bool(cited) and set(cited).issubset(allowed) and item.score is not None
        if valid:
            scores[key] = float(item.score)
            evidence_ids[key] = cited
            covered_weight += criterion["weight"]
        else:
            missing.append(key)
            evidence_ids[key] = []
    coverage = round(covered_weight / sum(c["weight"] for c in BUSINESS_CRITERIA.values()) * 100, 2)
    if coverage < 80.0:
        return None, coverage, scores, evidence_ids, missing
    score = sum(scores[k] * BUSINESS_CRITERIA[k]["weight"] for k in scores) / covered_weight
    return round(score, 2), coverage, scores, evidence_ids, missing


# rubric.yaml의 항목 ID 기준. 담당자가 평가 범위에 맞게 조정할 수 있습니다.
# 핵심 확인 항목: 수익모델, 고객 계약, 팀 실행력
EVIDENCE_POLICY = {
    "min_coverage": 80.0,
    "critical_items": ("fin_01", "fin_02", "team_01"),
}

def run_business_agent(state: Any) -> ProfileResult:
    """정수민 팀원이 담당하는 '사업성중심형' 에이전트 커스텀 로직"""
    
    # ==========================================
    # [1] 공통 파라미터 및 점수 (70% 반영) - 수정 지양
    # ==========================================
    profile_id = "business_focused"
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]

    # 근거 부족이면 공통/커스텀 점수를 계산하지 않고 정상적으로 판단 유보를 반환합니다.
    evidence_check = check_profile_evidence(state, config, **EVIDENCE_POLICY)
    if not evidence_check.sufficient:
        return evidence_check.hold_result(profile_name)
    
    # 공통 점수 계산 (도메인 점수 * 가중치)
    common_score = calculate_profile_score(state["domain_scores"], weights)
    
    # 커스텀 점수도 루브릭 항목별 근거를 인용하는 구조화 평가로 산출합니다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(BusinessProfileOutput)
    
    evidence_text = evidence_check.evidence_text

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.
        
당신의 투자 성향 및 최우선 판단 기준:
{focus}

사업성 세부 기준을 평가하고, 각 기준에 대해 제공된 Source ID만 인용하세요.
인용 ID는 해당 항목의 검증된 ID와 일치해야 합니다. 알 수 없으면 score=null, evidence_ids=[]로 둡니다.
0/25/50/75/100은 각각 확인된 실패, 계획·주장 수준, 구체적 초기 검증, 유료 계약·매출, 반복·다중 고객으로 입증된 수준입니다.
팀 실행력은 팀의 실제 수행 이력과 납품/운영 성과가 구체적으로 확인되는 정도로 평가합니다.
각 기준의 rationale과 missing_information을 작성하고, 추천·점수·판정은 임의로 계산하지 마세요.
확인 근거만 사용하고 미확인 수치를 추정하지 마세요."""),
        ("user", """
[기업명]: {company_name}
[프로필 근거 충족률]: {coverage_text}%

[도메인별 공통 점수]
{domain_scores_text}

[확인된 주요 근거]
{evidence_text}

[커스텀 기준 및 허용 근거 ID]
{criteria_text}
""")
    ])
    
    chain = prompt | structured_llm
    
    domain_scores_text = "\n".join([f"- {k.value}: {v}" for k, v in state["domain_scores"].items()])
    current_company = state.get("current_company")
    company_name = current_company.name if current_company else "Unknown Startup"
    
    criteria_text = "\n".join(
        f"- {key} (허용 항목: {', '.join(value['items'])}; "
        f"검증된 ID: {', '.join(dict.fromkeys(i for item_id in value['items'] for i in evidence_check.source_ids_by_item.get(item_id, []))) or '없음'})"
        for key, value in BUSINESS_CRITERIA.items()
    )
    result: BusinessProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "coverage_text": f"{evidence_check.coverage:.1f}",
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text,
        "criteria_text": criteria_text,
    })

    custom_score, custom_coverage, _, custom_ids, missing = calculate_business_custom_score(
        result.custom_assessment, evidence_check.source_ids_by_item
    )
    if custom_score is None:
        held = evidence_check.hold_result(profile_name)
        held.unknown_items.extend(f"사업성 세부 기준 미확인: {key}" for key in missing)
        held.due_diligence_questions.extend(
            getattr(result.custom_assessment, key).missing_information or f"{key} 관련 추가 자료 확인"
            for key in missing
        )
        return held
    final_score = round((common_score * 0.7) + (custom_score * 0.3), 2)
    
    return attach_evidence_check(ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=(
            f"사업성 커스텀 점수 {custom_score:.1f}/100 (세부 기준 근거 충족 {custom_coverage:.0f}%). "
            + result.recommendation
        ),
        due_diligence_questions=result.due_diligence_questions,
        supporting_evidence_ids=list(dict.fromkeys(
            result.supporting_evidence_ids + [i for ids in custom_ids.values() for i in ids]
        )),
    ), evidence_check)
