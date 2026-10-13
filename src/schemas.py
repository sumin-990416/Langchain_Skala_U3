import operator
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, TypedDict
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

class ResearchTask(BaseModel):
    """Orchestrator가 런타임에 생성하여 개별 Worker에 전달하는 조사 단위."""
    task_id: str = Field(description="조사 작업 고유 ID")
    domain: Domain = Field(description="조사 대상 평가 영역")
    domains: List[Domain] = Field(default_factory=list, description="이 작업이 다루는 모든 평가 영역")
    query: str = Field(description="다른 작업과 독립적으로 수행 가능한 검색 질문")
    objective: str = Field(default="", description="이 작업으로 확인할 결과")
    success_criteria: List[str] = Field(default_factory=list, description="검색 성공 판정 기준")

class Evidence(BaseModel):
    item_id: str
    content: str = Field(description="검색된 문서 내용 (Chunk)")
    source_type: str = Field(description="문서 출처 유형 (예: 논문, 뉴스, 홈페이지)")
    source_id: str = Field(default="", description="근거 식별자")
    source_document: str = Field(default="", description="독립 출처 판단용 원본 문서명")
    evidence_grade: str = Field(default="U", description="근거 등급 (A, B, C, U)")
    is_direct: bool = Field(default=False, description="기업과 직접 연관된 근거 여부")

class ResearchResult(BaseModel):
    """한 Research Worker가 반환하는 결과."""
    task: ResearchTask
    evidence: List[Evidence] = Field(default_factory=list)
    status: Literal["SUCCESS", "EMPTY", "FAILED"] = "SUCCESS"
    attempts: int = 1
    error: Optional[str] = None
    duration_ms: int = 0


class ResearchSynthesis(BaseModel):
    successful_task_ids: List[str] = Field(default_factory=list)
    failed_task_ids: List[str] = Field(default_factory=list)
    empty_task_ids: List[str] = Field(default_factory=list)
    missing_domains: List[Domain] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    is_sufficient: bool = False


class ReportEvaluation(BaseModel):
    groundedness: bool = False
    neutrality: bool = False
    bias_control: bool = False
    passed: bool = False
    feedback: List[str] = Field(default_factory=list)


class WorkflowError(BaseModel):
    node: str
    message: str
    task_id: Optional[str] = None
    attempts: int = 1

class ProfileResult(BaseModel):
    profile_name: str = Field(description="기술·안정·성장·균형·사업 중 하나")
    weighted_score: float = Field(description="해당 Profile 가중치를 적용한 0~100점")
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
    average_score: Optional[float]
    score_std_dev: Optional[float]
    scoring_eligible: bool = True
    final_decision: str
    summary_reason: str
    evidence_sources: List[str] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    research_synthesis: Optional[ResearchSynthesis] = None

class ComparisonResult(BaseModel):
    companies: List[CompanyResult] = Field(default_factory=list)
    comparison_table: str = ""

# --- Graph State ---
def reset_or_add(left: List[ProfileResult], right: List[ProfileResult]) -> List[ProfileResult]:
    if right == []:
        return []
    return left + right

def reset_or_add_research(
    left: List[ResearchResult], right: List[ResearchResult]
) -> List[ResearchResult]:
    """병렬 Worker 결과는 합치고, 다음 기업 시작 시 빈 리스트로 초기화한다."""
    if right == []:
        return []
    return left + right

class InvestmentAgentState(TypedDict):
    run_id: str
    status: str
    termination_reason: str
    plan_version: int
    plan_error: str
    evidence_retry_count: int
    max_evidence_retries: int
    report_retry_count: int
    max_report_retries: int
    selected_companies: List[Company]
    current_index: int
    current_company: Optional[Company]
    
    evaluation_questions: List[Question]
    research_tasks: List[ResearchTask]
    research_results: Annotated[List[ResearchResult], reset_or_add_research]
    research_synthesis: Optional[ResearchSynthesis]
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
    report_uri: str
    report_errors: List[str]
    report_evaluation: Optional[ReportEvaluation]
    errors: Annotated[List[WorkflowError], operator.add]
