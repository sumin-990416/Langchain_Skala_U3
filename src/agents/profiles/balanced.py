"""균형형: 공통 점수와 별도로 네 가지 연결성을 근거 기반으로 평가한다."""

import re
from typing import Any, Dict, List, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ...schemas import Domain, FinalDecision, ProfileResult, ResultCategory
from ..profile import calculate_profile_score, load_profile_config


CRITERION_WEIGHTS: Dict[str, float] = {
    "기술-시장 적합성": 0.30,
    "시장-수익 전환성": 0.25,
    "기술-리스크 양립성": 0.25,
    "팀-사업 실행 연결성": 0.20,
}
GRADE_LABELS = {0: "매우 낮음", 25: "낮음", 50: "보통", 75: "좋음", 100: "매우 좋음"}
SHORT_LABELS = {
    "기술-시장 적합성": "기술-시장",
    "시장-수익 전환성": "시장-수익",
    "기술-리스크 양립성": "기술-리스크",
    "팀-사업 실행 연결성": "팀-사업",
}
ASSESSMENT_FIELDS = {
    "기술-시장 적합성": "technology_market_fit",
    "시장-수익 전환성": "market_revenue_conversion",
    "기술-리스크 양립성": "technology_risk_compatibility",
    "팀-사업 실행 연결성": "team_business_execution",
}


class CriterionAssessment(BaseModel):
    score: Optional[Literal[0, 25, 50, 75, 100]] = Field(
        description="근거에 맞는 점수. 해당 기업의 직접 근거가 없으면 null"
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="실제로 판단에 사용한 EV-번호 또는 원문 근거 ID(A-01, V-01, E-01 등) 목록",
    )
    rationale: str = Field(description="근거와 한계를 구분한 짧은 평가 이유")
    missing_information: str = Field(
        default="", description="추가로 확인해야 하는 자료 또는 수치"
    )


class BalancedAssessment(BaseModel):
    technology_market_fit: CriterionAssessment
    market_revenue_conversion: CriterionAssessment
    technology_risk_compatibility: CriterionAssessment
    team_business_execution: CriterionAssessment


class InsufficientBalancedEvidenceError(ValueError):
    """공통 숫자형 스키마에서 미산출을 0점으로 위장하지 않기 위한 명시적 중단."""


def _is_company_evidence(content: str, company_name: str) -> bool:
    """공유 검색 결과에 다른 기업 청크가 섞여도 해당 기업 근거만 사용한다."""
    lower = content.casefold()
    company = company_name.casefold()
    if "accure" in company:
        return "accure" in lower or bool(re.search(r"\bA-?\d{2}\b", content))
    if "volytica" in company:
        return "volytica" in lower or bool(re.search(r"\bV-?\d{2}\b", content))
    if "electra" in company:
        return "electra" in lower or bool(re.search(r"\bE-?\d{2}\b", content))
    return company in lower


def _prepare_evidence(state: Any, company_name: str) -> Dict[str, Dict[str, Any]]:
    """중복을 제거하고, LLM 인용용 ID와 실제 PDF 근거 ID를 연결한다."""
    selected: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for ev in state.get("validated_evidence", []):
        content = ev.content.strip()
        if not content or content in seen or not _is_company_evidence(content, company_name):
            continue
        seen.add(content)
        key = f"EV-{len(selected) + 1:03d}"
        pdf_ids = list(dict.fromkeys(re.findall(r"\b[AVE]-?\d{2}\b", content)))
        selected[key] = {
            "content": content,
            "source": ev.source_id or "출처 미상",
            "public_ids": pdf_ids or [ev.source_id or key],
        }
        if len(selected) >= 25:
            break
    return selected


def calculate_balanced_custom_score(scores: Dict[str, Optional[int]]) -> Optional[float]:
    """네 항목 중 하나라도 근거 부족이면 전용 점수 전체를 미산출한다."""
    if set(scores) != set(CRITERION_WEIGHTS):
        return None
    if any(score is None for score in scores.values()):
        return None
    return round(sum(scores[name] * weight for name, weight in CRITERION_WEIGHTS.items()), 2)


def run_balanced_agent(state: Any) -> ProfileResult:
    """공통 70% + 독립 근거로 채점한 균형형 전용 네 항목 30%."""
    config = load_profile_config("balanced")
    profile_name = config["name"]
    weights = config["weights"]
    common_score = calculate_profile_score(state["domain_scores"], weights)

    current_company = state.get("current_company")
    company_name = current_company.name if current_company else "Unknown Startup"
    evidence = _prepare_evidence(state, company_name)
    assessments: Optional[BalancedAssessment] = None

    if evidence:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 배터리 AI 스타트업의 균형형 투자 심사역입니다.
공통 영역 점수를 재가중하거나 그대로 복사하지 말고, 제공된 해당 기업 원문 근거만으로 네 연결성을 각각 판정하세요.
외부 링크를 방문한 것으로 가정하지 마세요. 근거에 적힌 사실·출처·한계만 사용하세요.

점수는 0, 25, 50, 75, 100 중 하나입니다. 자료가 없어 평가할 수 없으면 반드시 null입니다.
0점은 근거가 '없다'는 뜻이 아니라, 직접적인 실패·불일치 근거가 확인된 경우에만 사용합니다.
기업의 목표/홍보 주장과 실제 현장 적용, 계약 체결과 매출 수취, 이상 탐지와 모델 오탐·미탐 검증을 구분하세요.
항목마다 최소 하나의 관련 원문 근거 ID(A-01, V-01, E-01 등) 또는 EV-번호를 정확히 인용해야 점수를 줄 수 있습니다.
가능하면 항목에 직접 대응하는 원문 근거 ID를 인용하세요. 인용할 근거가 없으면 null로 두세요.
근거에 없는 유료 매출, 계약 금액, 재계약, 독립 검증, 위험 감소 실적을 추정하지 마세요.

기술-시장 적합성: 25=문제·해결책 주장, 50=구체적 제품 기능과 고객 문제 연결,
75=현장 적용과 문제 해결 사례, 100=독립 확인된 복수 고객 성과.
시장-수익 전환성: 25=선정·도입 계획, 50=실제 고객 운영 적용,
75=체결된 상용 계약 또는 유료 거래의 근거, 100=금액·반복매출·갱신까지 확인.
기술-리스크 양립성: 25=위험 대응 주장, 50=구체적 감시·시험·대응 방식,
75=현장 위험 발견과 사람의 대응이 확인된 사례, 100=독립 검증된 실패율·안전성과 반복 운영.
팀-사업 실행 연결성: 25=관련 인력·파트너만 제시, 50=역할 분담과 통합 시험·계획,
75=역할이 연결된 완료된 현장 운영, 100=다수 프로젝트에서 반복 가능한 실행을 외부 확인.
75점 이상의 기준을 못 채운다는 이유만으로 null로 두지 말고 확인된 낮은 등급을 선택하세요."""),
            ("user", "기업: {company_name}\n\n검색된 기업별 근거:\n{evidence_text}"),
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | llm.with_structured_output(BalancedAssessment)
        assessments = chain.invoke({
            "company_name": company_name,
            "evidence_text": "\n\n".join(
                f"[{key}] [원자료 {item['source']}] {item['content']}"
                for key, item in evidence.items()
            ),
        })

    criterion_results: Dict[str, Dict[str, Any]] = {}
    for label, field_name in ASSESSMENT_FIELDS.items():
        assessment = getattr(assessments, field_name) if assessments else None
        cited_public_ids = []
        for ref in assessment.evidence_ids if assessment else []:
            if ref in evidence:
                cited_public_ids.extend(evidence[ref]["public_ids"])
            elif any(ref in item["public_ids"] for item in evidence.values()):
                cited_public_ids.append(ref)
        cited_public_ids = list(dict.fromkeys(cited_public_ids))
        score = assessment.score if assessment and cited_public_ids else None
        criterion_results[label] = {
            "score": score,
            "label": GRADE_LABELS[score] if score is not None else "미산출",
            "rationale": (
                assessment.rationale if score is not None
                else "해당 기업의 직접 인용 가능한 근거가 부족함"
            ),
            "evidence_ids": cited_public_ids if score is not None else [],
            "missing_information": (
                assessment.missing_information if assessment
                else "해당 항목의 기업별 원문 근거"
            ),
        }

    custom_score = calculate_balanced_custom_score({
        name: item["score"] for name, item in criterion_results.items()
    })
    if custom_score is None:
        missing = [name for name, item in criterion_results.items() if item["score"] is None]
        covered_weight = sum(
            CRITERION_WEIGHTS[name]
            for name, item in criterion_results.items()
            if item["score"] is not None
        )
        coverage = round(covered_weight * 100, 2)
        return ProfileResult(
            profile_name=profile_name,
            weighted_score=None,
            decision=FinalDecision.HOLD_RAG,
            evidence_coverage=coverage,
            result_category=(
                ResultCategory.PARTIAL_EVIDENCE
                if coverage > 0
                else ResultCategory.INSUFFICIENT_EVIDENCE
            ),
            supporting_evidence_ids=list(dict.fromkeys(
                ref for item in criterion_results.values()
                for ref in item["evidence_ids"]
            )),
            unknown_items=[
                f"{name}: {criterion_results[name]['missing_information'] or '직접 근거 미확인'}"
                for name in missing
            ],
            recommendation=(
                "점수 없음 / 근거 부족 판단 유보 | "
                f"균형형 세부 근거충족률 {coverage:.1f}%. "
                f"근거 부족 항목: {', '.join(missing)}"
            ),
            due_diligence_questions=[
                f"{name}: {criterion_results[name]['missing_information'] or '기업별 직접 근거를 제출할 것'}"
                for name in missing
            ],
        )

    final_score = round(common_score * 0.7 + custom_score * 0.3, 2)
    if final_score >= 75:
        decision = FinalDecision.RECOMMEND
    elif final_score >= 60:
        decision = FinalDecision.CONDITIONAL
    else:
        decision = FinalDecision.HOLD_SCORE
    grade_summary = "·".join(
        f"{SHORT_LABELS[name]} {item['label']}"
        for name, item in criterion_results.items()
    )
    recommendation = f"{decision.value} | {grade_summary}"
    unknown_items = [
        f"{name}: {item['missing_information']}"
        for name, item in criterion_results.items() if item["missing_information"]
    ]

    domain_contributions = {
        domain.value: round(
            state["domain_scores"].get(domain, 0) * weights.get(domain.value, 0) * 0.7, 2
        )
        for domain in Domain
    }
    supporting_ids = list(dict.fromkeys(
        ref for item in criterion_results.values()
        for ref in item["evidence_ids"]
    ))
    return ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        decision=decision,
        evidence_coverage=100.0,
        result_category=(
            ResultCategory.PARTIAL_EVIDENCE
            if unknown_items
            else ResultCategory.COMPLETE
        ),
        domain_contributions=domain_contributions,
        supporting_evidence_ids=supporting_ids,
        contrary_evidence_ids=[],
        unknown_items=unknown_items,
        recommendation=recommendation,
        due_diligence_questions=[
            f"{name}: {item['missing_information']}"
            for name, item in criterion_results.items() if item["missing_information"]
        ],
    )
