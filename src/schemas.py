import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

# --- Enums ---
class Domain(str, Enum):
    TECHNOLOGY = "technology"
    MARKET = "market"
    FINANCE = "finance"
    RISK = "risk"
    TEAM = "team"

class FinalDecision(str, Enum):
    RECOMMEND = "투자검토 추천"
    CONDITIONAL = "조건부 검토"
    HOLD_RAG = "근거 부족 판단 유보"
    HOLD_GATE = "투자 보류 및 추가 실사"
    HOLD_SCORE = "보류 (기준 미달)"

# --- Data Models (Pydantic) ---

class Company(BaseModel):
    name: str = Field(description="기업명")
    description: str = Field(default="", description="기업 기본 설명")

class Question(BaseModel):
    domain: Domain
    item_id: str = Field(description="조사 항목 고유 ID")
    query: str = Field(description="평가를 위한 검색 질문")

class Evidence(BaseModel):
    item_id: str
    content: str = Field(description="검색된 문서 내용 (Chunk)")
    source_type: str = Field(description="문서 출처 유형 (예: 논문, 뉴스, 홈페이지)")
    source_id: str = Field(default="", description="근거 식별자")
    evidence_grade: str = Field(default="U", description="근거 등급 (A, B, C, U)")
    is_direct: bool = Field(default=False, description="기업과 직접 연관된 근거 여부")

class ProfileResult(BaseModel):
    profile_name: str = Field(description="기술·안정·성장·균형·사업 중 하나")
    weighted_score: Optional[float] = Field(
        default=None, ge=0, le=100, allow_inf_nan=False,
        description="해당 Profile 가중치를 적용한 0~100점. 근거 부족으로 미산출이면 None"
    )
    decision: Optional[FinalDecision] = Field(
        default=None, description="구조화된 프로필 판정. 종합 단계에서 판단 유보를 전달하는 데 사용"
    )
    evidence_coverage: Optional[float] = Field(
        default=None, ge=0, le=100, allow_inf_nan=False,
        description="해당 프로필의 근거 충족률(0~100%). 계산하지 않은 프로필은 None"
    )
    domain_contributions: Dict[str, float] = Field(default_factory=dict, description="영역별 점수·가중치·기여점수")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="찬성 근거 ID")
    contrary_evidence_ids: List[str] = Field(default_factory=list, description="반대·상충 근거 ID")
    unknown_items: List[str] = Field(default_factory=list, description="공개자료로 확인하지 못한 항목")
    recommendation: str = Field(description="추천·조건부 검토·보류")
    due_diligence_questions: List[str] = Field(default_factory=list, description="투자 전 추가 확인 질문")

class CompanyResult(BaseModel):
    company_name: str
    domain_scores: Dict[Domain, float]
    evidence_coverage: float
    profile_results: List[ProfileResult]
    average_score: Optional[float] = Field(
        default=None, ge=0, le=100, allow_inf_nan=False,
        description="산출된 프로필 점수의 평균. 모두 미산출이면 None"
    )
    score_std_dev: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False,
        description="산출된 프로필 점수의 모집단 표준편차. 모두 미산출이면 None"
    )
    final_decision: str
    summary_reason: str

class ComparisonResult(BaseModel):
    companies: List[CompanyResult] = Field(default_factory=list)
    comparison_table: str = ""

# --- Graph State ---
def reset_or_add(left: List[ProfileResult], right: List[ProfileResult]) -> List[ProfileResult]:
    if right == []:
        return []
    return left + right

class InvestmentAgentState(TypedDict):
    selected_companies: List[Company]
    current_index: int
    current_company: Optional[Company]
    
    evaluation_questions: List[Question]
    current_question: Optional[Question]
    retrieved_evidence: List[Evidence]
    validated_evidence: List[Evidence]
    evidence_coverage: float
    retry_count: Dict[str, int]
    unknown_items: List[str]
    
    domain_scores: Dict[Domain, float]
    profile_results: Annotated[List[ProfileResult], reset_or_add]
    company_result: Optional[CompanyResult]
    company_results: Annotated[List[CompanyResult], operator.add]
    
    final_comparison: Optional[ComparisonResult]
    used_sources: List[str]
    report_draft: str
    report_errors: List[str]
    errors: List[str]
