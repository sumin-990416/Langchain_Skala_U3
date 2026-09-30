from typing import Dict, Any, List
from ...schemas import ProfileResult, Domain
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ._evidence import check_profile_evidence, attach_evidence_check, profile_decision


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
    
    # ==========================================
    # [2] 팀원 커스텀 평가 로직 (30% 반영) - 자유롭게 수정!
    # 1. 재무 및 시장 도메인 가중 평가 (최대 30점)
    market_score = state.get("domain_scores", {}).get(Domain.MARKET, 0)
    finance_score = state.get("domain_scores", {}).get(Domain.FINANCE, 0)
    avg_biz_score = (market_score + finance_score) / 2
    
    # 평균 60점 이상부터 점수를 제대로 부여 (기본적인 사업성은 갖춰야 함)
    if avg_biz_score < 60.0:
        biz_base_points = (avg_biz_score / 100.0) * 15.0 # 낮은 점수는 절반만 인정
    else:
        biz_base_points = min(30.0, (avg_biz_score / 100.0) * 30.0)
    
    # 2. 확실한 '돈 버는' 비즈니스 근거 유무 (최대 40점)
    strong_biz_keywords = ["매출", "계약", "수익", "유료", "고객", "b2b", "상용화"]
    risk_keywords = ["적자", "지연", "불확실", "개선 필요"]
    
    strong_evidence_count = 0
    risk_evidence_count = 0
    
    for ev in state.get("validated_evidence", []):
        content_lower = ev.content.lower()
        
        if any(kw in content_lower for kw in strong_biz_keywords):
            strong_evidence_count += 1
            if getattr(ev, "evidence_grade", "C") in ["A", "B"]:
                strong_evidence_count += 1 # A/B급이면 가중치 부여 (총 2번 카운트)
                
        # 리스크(부정적 팩트) 발견 시 감점 카운트
        if any(kw in content_lower for kw in risk_keywords):
            risk_evidence_count += 1
                
    # 긍정 근거 1포인트당 10점 (최대 40점)
    evidence_points = min(40.0, strong_evidence_count * 10.0)
    
    # 3. 리스크 페널티 감점 (건당 -5점)
    penalty_points = risk_evidence_count * 5.0
    
    # 4. 사업 실행력(TEAM) 평가 반영 (최대 30점)
    team_score = state.get("domain_scores", {}).get(Domain.TEAM, 0)
    team_points = min(30.0, (team_score / 100.0) * 30.0)
    
    # 총 커스텀 점수 합산
    custom_score = biz_base_points + evidence_points + team_points - penalty_points
    
    # [소프트 과락 시스템]
    # 매출/계약/수익 등 사업적 키워드가 하나도 발견되지 않았다면 총점에서 20점 감점 (0점 처리는 아님)
    if strong_evidence_count == 0:
        custom_score -= 20.0
        
    custom_score = round(max(0.0, min(100.0, custom_score)), 2)
    
    # 최종 점수 산출 (공통 70% + 커스텀 30%)
    final_score = round((common_score * 0.7) + (custom_score * 0.3), 2)
    
    # ==========================================
    # [3] 팀원 커스텀 LLM 프롬프트 - 자유롭게 수정!
    # ==========================================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(LLMProfileOutput)
    
    evidence_text = evidence_check.evidence_text

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.
        
당신의 투자 성향 및 최우선 판단 기준:
{focus}

제공된 '도메인별 평가 점수'와 '확인된 근거'를 바탕으로, 당신의 성향에 입각해 근거에 기반한 투자의견을 도출하세요.
코드가 결정한 판정을 변경하지 말고 근거와 한계를 설명하세요. 미확인 수치를 추정하지 마세요.
비즈니스 모델, 수익성, 그리고 팀의 실행력에 중점을 두어 평가를 진행해야 합니다."""),
        ("user", """
[기업명]: {company_name}
[당신의 최종 산출 점수 (공통 70% + 커스텀 30%)]: {final_score} / 100
[코드가 결정한 판정]: {decision_text}
[프로필 근거 충족률]: {coverage_text}%

[도메인별 공통 점수]
{domain_scores_text}

[확인된 주요 근거]
{evidence_text}
""")
    ])
    
    chain = prompt | structured_llm
    
    domain_scores_text = "\n".join([f"- {k.value}: {v}" for k, v in state["domain_scores"].items()])
    current_company = state.get("current_company")
    company_name = current_company.name if current_company else "Unknown Startup"
    
    result: LLMProfileOutput = chain.invoke({
        "profile_name": profile_name,
        "focus": config["focus"],
        "company_name": company_name,
        "final_score": final_score,
        "decision_text": profile_decision(final_score, evidence_check).value,
        "coverage_text": f"{evidence_check.coverage:.1f}",
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text
    })
    
    return attach_evidence_check(ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=result.supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    ), evidence_check)
