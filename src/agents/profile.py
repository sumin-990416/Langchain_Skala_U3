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
    단일 Profile Agent를 실행하는 노드 함수.
    각 성향별로 분리된 파일의 함수를 호출합니다.
    """
    profile_id = state["profile_name"]
    
    # 성향별 커스텀 모듈 라우팅
    if profile_id == "technology_focused":
        from src.agents.profiles.technology import run_technology_agent
        result = run_technology_agent(state)
    elif profile_id == "stability_focused":
        from src.agents.profiles.stability import run_stability_agent
        result = run_stability_agent(state)
    elif profile_id == "growth_focused":
        from src.agents.profiles.growth import run_growth_agent
        result = run_growth_agent(state)
    elif profile_id == "balanced":
        from src.agents.profiles.balanced import run_balanced_agent
        result = run_balanced_agent(state)
    elif profile_id == "business_focused":
        from src.agents.profiles.business import run_business_agent
        result = run_business_agent(state)
    else:
        raise ValueError(f"알 수 없는 프로필입니다: {profile_id}")

    # Reducer (`operator.add`)를 통해 graph state 배열에 자동 누적됨
    return {"profile_results": [result]}
