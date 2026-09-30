from typing import Dict, Any, List
from src.schemas import ProfileResult, Domain
from src.agents.profile import calculate_profile_score, LLMProfileOutput, load_profile_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_technology_agent(state: Any) -> ProfileResult:
    """손연우 팀원이 담당하는 '기술중심형' 에이전트 커스텀 로직"""
    profile_id = "기술중심형"
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]
    focus = config["focus"]
    
    score = calculate_profile_score(state["domain_scores"], weights)
    
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
[계산된 당신의 총점]: {score} / 100

[도메인별 점수]
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
        "focus": focus,
        "company_name": company_name,
        "score": score,
        "domain_scores_text": domain_scores_text,
        "evidence_text": evidence_text
    })
    
    return ProfileResult(
        profile_name=profile_name,
        weighted_score=score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=result.supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    )
