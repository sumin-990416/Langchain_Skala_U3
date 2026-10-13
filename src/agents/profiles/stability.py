from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ...schemas import ProfileResult
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from ._evidence import check_profile_evidence, attach_evidence_check


# 안정형 투자자가 중점적으로 확인할 실사 영역과 사전 공개된 상대 가중치입니다.
STABILITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "safety_regulation": {
        "label": "안전·인증·규제 준수",
        "weight": 0.25,
        "items": ("rsk_01",),
    },
    "financial_continuity": {
        "label": "투자단계·재무 지속성",
        "weight": 0.20,
        "items": ("fin_01", "fin_02"),
    },
    "data_rights": {
        "label": "데이터 이용권·개인정보",
        "weight": 0.15,
        "items": ("rsk_02",),
    },
    "product_liability": {
        "label": "제조물책임·보증·보험",
        "weight": 0.15,
        "items": ("rsk_01",),
    },
    "intellectual_property": {
        "label": "특허·지식재산 방어력",
        "weight": 0.10,
        "items": ("tech_03",),
    },
}


class StabilityCriterionAssessment(BaseModel):
    score: Optional[Literal[0, 25, 50, 75, 100]] = Field(
        description="근거로 판단할 수 없으면 null, 아니면 기준에 맞는 0/25/50/75/100"
    )
    evidence_ids: List[str] = Field(default_factory=list)
    rationale: str
    missing_information: str = ""


class StabilityCustomAssessment(BaseModel):
    safety_regulation: StabilityCriterionAssessment
    financial_continuity: StabilityCriterionAssessment
    data_rights: StabilityCriterionAssessment
    product_liability: StabilityCriterionAssessment
    intellectual_property: StabilityCriterionAssessment


class StabilityProfileOutput(LLMProfileOutput):
    custom_assessment: StabilityCustomAssessment


def calculate_stability_custom_score(assessment: StabilityCustomAssessment, source_ids_by_item: Dict[str, List[str]]):
    """검증된 루브릭 인용이 있는 안정형 기준만 가중 정규화합니다."""
    scores, evidence_ids, missing, covered_weight = {}, {}, [], 0.0
    for key, criterion in STABILITY_CRITERIA.items():
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
    total_weight = sum(c["weight"] for c in STABILITY_CRITERIA.values())
    coverage = round(covered_weight / total_weight * 100, 2)
    if coverage < 80.0:
        return None, coverage, scores, evidence_ids, missing
    score = sum(scores[k] * STABILITY_CRITERIA[k]["weight"] for k in scores) / covered_weight
    return round(score, 2), coverage, scores, evidence_ids, missing


def _format_stability_assessment(source_ids_by_item: Dict[str, List[str]]) -> str:
    """기준별로 인용 가능한 검증 근거 ID를 LLM에 전달합니다."""
    lines: List[str] = []
    for criterion_id, criterion in STABILITY_CRITERIA.items():
        ids = list(dict.fromkeys(
            source_id for item_id in criterion["items"]
            for source_id in source_ids_by_item.get(item_id, [])
        ))
        source_text = ", ".join(ids) if ids else "미확인"
        lines.append(f"- {criterion_id} / {criterion['label']} (허용 항목: {', '.join(criterion['items'])}; 근거 ID: {source_text})")
    return "\n".join(lines)

# rubric.yaml의 항목 ID 기준. 담당자가 평가 범위에 맞게 조정할 수 있습니다.
# 핵심 확인 항목: 재무 지속성, 안전·규제, 데이터 이용 권한
EVIDENCE_POLICY = {
    "min_coverage": 80.0,
    "critical_items": ("fin_01", "rsk_01", "rsk_02"),
}

def run_stability_agent(state: Any) -> ProfileResult:
    """강도희 팀원이 담당하는 '안정형' 에이전트 커스텀 로직."""

    # [1] 공통 평가 점수 70%
    # YAML은 영문 key(stability_focused)를 쓰고, fallback 설정은 한글 key를
    # 쓰므로 두 경로를 모두 지원합니다.
    try:
        config = load_profile_config("stability_focused")
    except KeyError:
        config = load_profile_config("안정형")
    profile_name = config["name"]
    weights = config["weights"]

    # 근거 부족이면 공통/커스텀 점수를 계산하지 않고 정상적으로 판단 유보를 반환합니다.
    evidence_check = check_profile_evidence(state, config, **EVIDENCE_POLICY)
    if not evidence_check.sufficient:
        return evidence_check.hold_result(profile_name)

    common_score = calculate_profile_score(state["domain_scores"], weights)

    # 세부 평가와 서술을 한 번의 구조화 호출로 생성합니다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(StabilityProfileOutput)

    evidence_text = evidence_check.evidence_text

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.

당신의 투자 성향 및 최우선 판단 기준:
{focus}

제공된 검증 근거로 안정형 커스텀 기준 점수를 평가하고 투자의견을 작성하세요.

판단 원칙:
1. 안전·규제, 재무 지속성, 데이터 권리, 제조물책임, 특허 방어력을 우선합니다.
2. 제공된 근거 ID만 인용하고 없는 ID나 수치를 만들지 마세요.
3. 근거가 없는 항목은 score=null, evidence_ids=[]로 두고 부정적 사실로 단정하지 마세요.
4. 각 기준은 0/25/50/75/100의 단계로 평가합니다: 확인된 실패, 미흡·계획 수준,
   일부 구체적 확인, 대부분 확인, 독립적·반복적으로 확인된 수준입니다.
5. 해당 기준의 허용 항목에 속한 ID만 인용하고, 인용 ID와 rationale의 사실을 일치시키세요.
6. 확인된 인용이 없으면 점수를 만들지 않습니다. 출처 등급을 점수로 바꾸지 마세요.
7. 추천/최종 점수는 계산하지 말고 미확인 수치를 추정하지 마세요."""),
        ("user", """
[기업명]: {company_name}
[프로필 근거 충족률]: {coverage_text}%

[안정형 세부 평가]
{stability_assessment}

[도메인별 공통 점수]
{domain_scores_text}

[확인된 주요 근거]
{evidence_text}
""")
    ])

    chain = prompt | structured_llm
    domain_scores_text = "\n".join(
        [f"- {key.value}: {value}" for key, value in state["domain_scores"].items()]
    )
    current_company = state.get("current_company")
    company_name = current_company.name if current_company else "Unknown Startup"

    result: StabilityProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "stability_assessment": _format_stability_assessment(
            evidence_check.source_ids_by_item
        ),
        "coverage_text": f"{evidence_check.coverage:.1f}",
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text,
    })

    custom_score, custom_coverage, _, custom_ids, missing = calculate_stability_custom_score(
        result.custom_assessment, evidence_check.source_ids_by_item
    )
    if custom_score is None:
        held = evidence_check.hold_result(profile_name)
        held.unknown_items.extend(f"안정형 세부 기준 미확인: {key}" for key in missing)
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
            f"안정형 커스텀 점수 {custom_score:.1f}/100 (세부 기준 근거 충족 {custom_coverage:.0f}%). "
            + result.recommendation
        ),
        due_diligence_questions=result.due_diligence_questions,
        supporting_evidence_ids=list(dict.fromkeys(
            result.supporting_evidence_ids + [i for ids in custom_ids.values() for i in ids]
        )),
    ), evidence_check)
