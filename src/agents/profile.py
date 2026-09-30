import yaml
from pathlib import Path
from typing import Dict, Any, List, TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.schemas import ProfileResult, Domain

class ProfileSendState(TypedDict):
    profile_name: str
    domain_scores: Dict[Domain, float]
    validated_evidence: List[Any]
    current_company: Any

def get_all_profiles() -> List[str]:
    config_path = Path(__file__).parent.parent.parent / "configs" / "profiles.yaml"
    if not config_path.exists():
        # 기본 프로필 제공
        return ["기술중심형", "안정형", "성장형", "균형형", "사업성중심형"]
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data["profiles"].keys())

class DomainContributions(BaseModel):
    technology: float = Field(description="기술 영역 기여 점수")
    market: float = Field(description="시장 영역 기여 점수")
    finance: float = Field(description="재무 영역 기여 점수")
    risk: float = Field(description="리스크 영역 기여 점수")
    team: float = Field(description="팀 영역 기여 점수")

class LLMProfileOutput(BaseModel):
    domain_contributions: DomainContributions = Field(description="각 영역별 점수가 가중치와 결합되어 총점에 기여한 점수 맵핑")
    supporting_evidence_ids: List[str] = Field(description="찬성 근거들의 source_id 목록")
    contrary_evidence_ids: List[str] = Field(description="반대 혹은 상충 근거들의 source_id 목록")
    unknown_items: List[str] = Field(description="문서에서 파악하지 못한 항목들")
    recommendation: str = Field(description="추천, 조건부 검토, 혹은 보류 판정 텍스트")
    due_diligence_questions: List[str] = Field(description="투자 전 추가 실사 질문")

def load_profile_config(profile_id: str) -> Dict[str, Any]:
    """YAML 설정 파일에서 특정 프로필 정보를 불러옵니다. 없을 경우 기본값 반환."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "profiles.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data["profiles"][profile_id]
        
    # 기본 하드코딩 설정 (설계안 기준)
    defaults = {
        "기술중심형": {"weights": {"technology": 0.35, "market": 0.20, "finance": 0.10, "risk": 0.15, "team": 0.20}, "focus": "기술 차별성 및 검증 수준 중심"},
        "안정형": {"weights": {"technology": 0.15, "market": 0.15, "finance": 0.30, "risk": 0.30, "team": 0.10}, "focus": "재무 지속성 및 안전성, 규제 위험 최소화"},
        "성장형": {"weights": {"technology": 0.20, "market": 0.35, "finance": 0.15, "risk": 0.10, "team": 0.20}, "focus": "시장 성장성, 고객 확대 및 확장 중심"},
        "균형형": {"weights": {"technology": 0.20, "market": 0.20, "finance": 0.20, "risk": 0.20, "team": 0.20}, "focus": "전 영역 균형 평가"},
        "사업성중심형": {"weights": {"technology": 0.15, "market": 0.25, "finance": 0.20, "risk": 0.10, "team": 0.30}, "focus": "유료 고객, 반복 매출 및 실행력 중심"}
    }
    return {"name": profile_id, "weights": defaults[profile_id]["weights"], "focus": defaults[profile_id]["focus"]}

def calculate_profile_score(domain_scores: Dict[Domain, float], weights: Dict[str, float]) -> float:
    """공통 점수와 가중치를 곱해 최종 점수를 계산"""
    total_score = 0.0
    for domain, score in domain_scores.items():
        weight = weights.get(domain.value, 0.2)
        total_score += score * weight
    return round(total_score, 2)

def profile_agent_node(state: ProfileSendState) -> Dict[str, Any]:
    """
    단일 Profile Agent를 실행하는 노드 함수
    """
    profile_id = state["profile_name"]
    config = load_profile_config(profile_id)
    profile_name = config["name"]
    weights = config["weights"]
    focus = config["focus"]
    
    # 1. 가중치 반영 점수 계산
    score = calculate_profile_score(state["domain_scores"], weights)
    
    # 2. LLM 평가 생성 (gpt-4o-mini 사용)
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
동일한 사실이라도 성향에 따라 강점이나 약점으로 다르게 해석되어야 합니다."""),
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
    
    profile_result = ProfileResult(
        profile_name=profile_name,
        weighted_score=score,
        domain_contributions=result.domain_contributions.model_dump(),
        supporting_evidence_ids=result.supporting_evidence_ids,
        contrary_evidence_ids=result.contrary_evidence_ids,
        unknown_items=result.unknown_items,
        recommendation=result.recommendation,
        due_diligence_questions=result.due_diligence_questions
    )
    
    # Reducer (`operator.add`)를 통해 graph state 배열에 자동 누적됨
    return {"profile_results": [profile_result]}
