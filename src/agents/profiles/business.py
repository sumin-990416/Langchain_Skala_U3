from typing import Dict, Any, List
from ...schemas import ProfileResult, Domain
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_business_agent(state: Any) -> ProfileResult:
    """정수민 팀원이 담당하는 '사업성중심형' 에이전트 커스텀 로직"""
    
    # ==========================================
    # [1] 공통 파라미터 및 점수 (70% 반영) - 수정 지양
    # ==========================================
    profile_id = "business_focused"
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]
    
    # 공통 점수 계산 (도메인 점수 * 가중치)
    common_score = calculate_profile_score(state["domain_scores"], weights)
    
    # ==========================================
    # [2] 팀원 커스텀 평가 로직 (30% 반영) - 자유롭게 수정!
    # ==========================================
    # TODO: [정수민]님, 근거(validated_evidence)나 도메인 점수를 활용해 
    # 사업성중심형에 맞는 독자적인 커스텀 점수(0~100)를 산출하세요.
    # 1. 재무 및 시장 도메인 가중 평가 (최대 30점)
    # 평균 70점 미만이면 아예 0점 처리 (매우 엄격)
    market_score = state.get("domain_scores", {}).get(Domain.MARKET, 0)
    finance_score = state.get("domain_scores", {}).get(Domain.FINANCE, 0)
    avg_biz_score = (market_score + finance_score) / 2
    
    if avg_biz_score < 70.0:
        biz_base_points = 0.0
    else:
        biz_base_points = min(30.0, ((avg_biz_score - 70) / 30.0) * 30.0)
    
    # 2. 확실한 '돈 버는' 비즈니스 근거 유무 (최대 40점)
    strong_biz_keywords = ["매출", "계약", "수익", "유료 고객", "arr", "mrr", "흑자", "상용화"]
    negative_biz_keywords = ["적자", "지연", "불확실", "개선 필요", "위험", "한계", "우려"]
    
    strong_evidence_count = 0
    negative_evidence_count = 0
    
    for ev in state.get("validated_evidence", []):
        content_lower = ev.content.lower()
        
        # 긍정 팩트는 출처 신뢰도가 높은(A, B급) 경우에만 인정!
        if getattr(ev, "evidence_grade", "C") in ["A", "B"]:
            if any(kw in content_lower for kw in strong_biz_keywords):
                strong_evidence_count += 1
                
        # 리스크(부정적 팩트) 발견 시 감점 카운트
        if any(kw in content_lower for kw in negative_biz_keywords):
            negative_evidence_count += 1
                
    # A/B급의 확실한 돈 버는 근거 1개당 20점 (최대 40점)
    evidence_points = min(40.0, strong_evidence_count * 20.0)
    
    # 3. 리스크 페널티 감점 (제한 없이 건당 -15점 깎음)
    penalty_points = negative_evidence_count * 15.0
    
    # 4. 사업 실행력(TEAM) 평가 반영 (최대 30점)
    # 실행력(팀) 점수가 70점 미만이면 가차없이 0점
    team_score = state.get("domain_scores", {}).get(Domain.TEAM, 0)
    team_points = min(30.0, max(0.0, (team_score - 70.0) / 30.0) * 30.0)
    
    # 총 커스텀 점수 합산
    custom_score = biz_base_points + evidence_points + team_points - penalty_points
    
    # [극단적 과락 시스템 💀]
    # 1) A/B급의 수익/계약 근거가 단 한 개도 없다면 무조건 0점 처리
    # 2) 재무/시장 평균 점수가 65점 미만이면 무조건 0점 처리
    if strong_evidence_count == 0 or avg_biz_score < 65:
        custom_score = 0.0
        
    custom_score = round(max(0.0, min(100.0, custom_score)), 2)
    
    # 최종 점수 산출 (공통 70% + 커스텀 30%)
    final_score = round((common_score * 0.7) + (custom_score * 0.3), 2)
    
    # ==========================================
    # [3] 팀원 커스텀 LLM 프롬프트 - 자유롭게 수정!
    # ==========================================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(LLMProfileOutput)
    
    evidence_text = ""
    for ev in state["validated_evidence"]:
        evidence_text += f"- [ID: {ev.item_id}, Source: {ev.source_id}] (등급: {ev.evidence_grade}): {ev.content}\n"
    if not evidence_text:
        evidence_text = "확인된 근거가 없습니다."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 배터리 AI 스타트업 전문 심사역 '{profile_name}'입니다.
        
당신의 투자 성향 및 최우선 판단 기준:
{focus}

제공된 '도메인별 평가 점수'와 '확인된 근거'를 바탕으로, 당신의 성향에 입각해 철저하게 편향된 투자의견을 도출하세요.
비즈니스 모델, 수익성, 그리고 팀의 실행력에 중점을 두어 평가를 진행해야 합니다."""),
        ("user", """
[기업명]: {company_name}
[당신의 최종 산출 점수 (공통 70% + 커스텀 30%)]: {final_score} / 100

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
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text
    })
    
    return ProfileResult(
        profile_name=profile_name,
        weighted_score=final_score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=result.supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    )
