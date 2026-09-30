from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ...schemas import ProfileResult
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config


# 안정형 투자자가 중점적으로 확인할 실사 영역입니다.
# 키워드는 검색된 근거를 영역별로 분류하는 용도로만 사용합니다.
STABILITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "safety_regulation": {
        "label": "안전·인증·규제 준수",
        "weight": 0.25,
        "keywords": (
            "안전", "규제", "인증", "표준", "열폭주", "화재", "safety",
            "regulation", "regulatory", "certification", "thermal runaway",
            "ul 9540", "ul9540", "iec 62619", "iso 26262", "un 38.3",
            "battery regulation",
        ),
    },
    "financial_continuity": {
        "label": "투자단계·재무 지속성",
        "weight": 0.20,
        "keywords": (
            "비상장", "투자", "매출", "현금", "계약", "인수", "합병", "상장",
            "seed", "series", "funding", "funded", "runway", "revenue",
            "cash flow", "contract", "private company", "exit", "acquisition", "ipo",
        ),
    },
    "data_rights": {
        "label": "데이터 이용권·개인정보",
        "weight": 0.15,
        "keywords": (
            "데이터 권리", "데이터 소유", "이용권", "동의", "라이선스",
            "개인정보", "data right", "data ownership", "consent", "license",
            "privacy", "gdpr", "data processing",
        ),
    },
    "product_liability": {
        "label": "제조물책임·보증·보험",
        "weight": 0.15,
        "keywords": (
            "제조물책임", "책임보험", "배상", "보증", "면책", "리콜",
            "product liability", "liability insurance", "indemn", "warranty", "recall",
        ),
    },
    "intellectual_property": {
        "label": "특허·지식재산 방어력",
        "weight": 0.10,
        "keywords": (
            "특허", "지식재산", "영업비밀", "patent", "intellectual property",
            "trade secret", "freedom to operate", "fto",
        ),
    },
}

GRADE_SCORE = {"A": 95.0, "B": 80.0, "C": 60.0, "U": 30.0}
UNKNOWN_CRITERION_SCORE = 40.0


def _get_field(item: Any, field_name: str, default: Any = None) -> Any:
    """Pydantic 모델과 dict 형태의 근거를 모두 지원합니다."""
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def _match_evidence(evidence: Iterable[Any], keywords: Sequence[str]) -> List[Any]:
    """근거 본문을 안정형 세부 실사 영역에 분류합니다."""
    normalized_keywords = tuple(keyword.casefold() for keyword in keywords)
    matched: List[Any] = []
    for item in evidence:
        content = str(_get_field(item, "content", "")).casefold()
        if any(keyword in content for keyword in normalized_keywords):
            matched.append(item)
    return matched


def _criterion_score(matched_evidence: Sequence[Any]) -> float:
    """
    근거 등급과 독립적인 source_id 수로 세부 실사 신뢰도를 산출합니다.

    근거가 없는 항목은 0점으로 단정하지 않고 40점(미확인)을 적용하여,
    추가 실사가 필요하다는 의미로 해석합니다.
    """
    if not matched_evidence:
        return UNKNOWN_CRITERION_SCORE

    grade_scores = [
        GRADE_SCORE.get(str(_get_field(item, "evidence_grade", "U")).upper(), 30.0)
        for item in matched_evidence
    ]
    unique_sources = {
        str(_get_field(item, "source_id", "")).strip()
        for item in matched_evidence
        if str(_get_field(item, "source_id", "")).strip()
    }
    diversity_bonus = min(6.0, max(0, len(unique_sources) - 1) * 2.0)
    return round(min(100.0, (sum(grade_scores) / len(grade_scores)) + diversity_bonus), 2)


def _evidence_reliability(evidence: Sequence[Any]) -> float:
    """전체 근거의 출처 등급과 출처 다양성을 평가합니다."""
    if not evidence:
        return 30.0

    grade_average = sum(
        GRADE_SCORE.get(str(_get_field(item, "evidence_grade", "U")).upper(), 30.0)
        for item in evidence
    ) / len(evidence)
    unique_sources = {
        str(_get_field(item, "source_id", "")).strip()
        for item in evidence
        if str(_get_field(item, "source_id", "")).strip()
    }
    source_diversity_score = min(100.0, 40.0 + len(unique_sources) * 15.0)
    return round((grade_average * 0.75) + (source_diversity_score * 0.25), 2)


def calculate_stability_custom_score(
    evidence: Sequence[Any],
) -> Tuple[float, Dict[str, float], Dict[str, List[str]]]:
    """안정형 Agent의 커스텀 30% 점수와 항목별 근거 ID를 산출합니다."""
    evidence_list = list(evidence)
    breakdown: Dict[str, float] = {}
    evidence_ids: Dict[str, List[str]] = {}
    weighted_total = 0.0

    for criterion_id, criterion in STABILITY_CRITERIA.items():
        matched = _match_evidence(evidence_list, criterion["keywords"])
        score = _criterion_score(matched)
        breakdown[criterion_id] = score
        evidence_ids[criterion_id] = sorted({
            str(_get_field(item, "source_id", "")).strip()
            for item in matched
            if str(_get_field(item, "source_id", "")).strip()
        })
        weighted_total += score * float(criterion["weight"])

    reliability = _evidence_reliability(evidence_list)
    breakdown["evidence_reliability"] = reliability
    evidence_ids["evidence_reliability"] = sorted({
        str(_get_field(item, "source_id", "")).strip()
        for item in evidence_list
        if str(_get_field(item, "source_id", "")).strip()
    })
    weighted_total += reliability * 0.15

    return round(weighted_total, 2), breakdown, evidence_ids


def _format_stability_assessment(
    breakdown: Dict[str, float], evidence_ids: Dict[str, List[str]]
) -> str:
    """계산된 실사 점수와 근거 ID를 LLM 프롬프트용 문자열로 만듭니다."""
    lines: List[str] = []
    for criterion_id, criterion in STABILITY_CRITERIA.items():
        ids = evidence_ids.get(criterion_id, [])
        source_text = ", ".join(ids) if ids else "미확인"
        lines.append(
            f"- {criterion['label']}: {breakdown[criterion_id]:.1f}점 "
            f"(근거: {source_text})"
        )

    reliability_ids = evidence_ids.get("evidence_reliability", [])
    reliability_sources = ", ".join(reliability_ids) if reliability_ids else "미확인"
    lines.append(
        f"- 근거 신뢰도: {breakdown['evidence_reliability']:.1f}점 "
        f"(근거: {reliability_sources})"
    )
    return "\n".join(lines)


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
    common_score = calculate_profile_score(state["domain_scores"], weights)

    # [2] 안정형 커스텀 실사 점수 30%
    validated_evidence = list(state.get("validated_evidence", []))
    custom_score, stability_breakdown, stability_evidence_ids = (
        calculate_stability_custom_score(validated_evidence)
    )
    final_score = round((common_score * 0.7) + (custom_score * 0.3), 2)

    # [3] 근거 기반 안정형 투자의견 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(LLMProfileOutput)

    evidence_text = ""
    for ev in validated_evidence:
        evidence_text += (
            f"- [ID: {ev.item_id}, Source: {ev.source_id}] "
            f"(등급: {ev.evidence_grade}): {ev.content}\n"
        )
    if not evidence_text:
        evidence_text = "확인된 근거가 없습니다."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.

당신의 투자 성향 및 최우선 판단 기준:
{focus}

제공된 도메인 점수와 확인된 근거만으로 안정형 투자의견을 도출하세요.

판단 원칙:
1. 안전·규제, 재무 지속성, 데이터 권리, 제조물책임, 특허 방어력을 우선합니다.
2. 제공된 근거 ID만 인용하고 없는 ID나 수치를 만들지 마세요.
3. 근거가 없는 항목은 부정적 사실로 단정하지 말고 '미확인'으로 기록하세요.
4. 회사 자체 주장과 독립 검증을 구분하고 A·B등급 근거를 우선하세요.
5. 최종 점수가 75점 이상이면 '추천', 60~74.99점이면 '조건부 검토',
   60점 미만이면 '보류'를 기본으로 하되 핵심 항목이 미확인이면 추가 실사 조건을 붙이세요."""),
        ("user", """
[기업명]: {company_name}
[최종 산출 점수(공통 70% + 커스텀 30%)]: {final_score} / 100
[안정형 커스텀 실사 점수]: {custom_score} / 100

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

    result: LLMProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "final_score": final_score,
        "custom_score": custom_score,
        "stability_assessment": _format_stability_assessment(
            stability_breakdown, stability_evidence_ids
        ),
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text,
    })

    return ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=result.supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions,
    )
