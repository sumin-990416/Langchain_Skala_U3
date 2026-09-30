from typing import Dict, Any, List
from ...schemas import ProfileResult, Domain
from ..profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_technology_agent(state: Any) -> ProfileResult:
    """손연우 팀원이 담당하는 '기술중심형' 에이전트 커스텀 로직"""
    
    # ==========================================
    # [1] 공통 파라미터 및 점수 (70% 반영) - 수정 지양
    # ==========================================
    profile_id = "기술중심형"
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]
    
    # 공통 점수 계산 (도메인 점수 * 가중치)
    common_score = calculate_profile_score(state["domain_scores"], weights)
    
    # ==========================================
    # [2] 팀원 커스텀 평가 로직 (30% 반영) - 자유롭게 수정!
    # ==========================================
    # TODO: [손연우]님, 근거(validated_evidence)나 도메인 점수를 활용해 
    # 기술중심형에 맞는 독자적인 커스텀 점수(0~100)를 산출하세요.
    custom_score = 80.0  # 예시 기본값
    
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
기술적 우위와 차별성에 집중하여 평가를 진행해야 합니다."""),
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
