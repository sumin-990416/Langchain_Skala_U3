"""State = 제어 흐름 + 재개에 필요한 작은 산출물.

원문 청크·계획 이력·결정 로그는 output/runs/<trace_id>/에 저장한다.
State에는 원문 경로와 짧은 인용, 최신 task_id별 결과만 유지한다.
병렬 노드는 worker_results만 쓰며 task_id reducer로 중복 실행을 병합한다.
계획 크기, 회사 수, 인용 길이, 재시도 횟수를 제한해 상태 증식을 막는다.
"""

from enum import Enum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from src.schemas import Company


class Perspective(str, Enum):
    TECHNOLOGY = "technology_maturity"
    MARKET = "market_viability"
    STAKEHOLDERS = "stakeholders"
    APPLICATION = "domain_application"


PERSPECTIVE_LABELS = {
    Perspective.TECHNOLOGY: "기술 성숙도",
    Perspective.MARKET: "시장성",
    Perspective.STAKEHOLDERS: "이해관계자",
    Perspective.APPLICATION: "도메인 적용",
}


class ResearchTask(BaseModel):
    task_id: str = ""
    company_name: str = Field(min_length=1, max_length=120)
    perspective: Perspective
    question: str = Field(min_length=1, max_length=500)
    queries: list[str] = Field(min_length=1, max_length=3)
    rationale: str = Field(min_length=1, max_length=500)


class ResearchPlan(BaseModel):
    tasks: list[ResearchTask] = Field(min_length=1, max_length=40)


class Citation(BaseModel):
    evidence_id: str
    quote: str = Field(default="", max_length=360)
    quote_id: str = Field(default="", max_length=32, description="입력 quotable_spans의 ID. 원문 인용은 코드에서 복원")


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    kind: Literal["support", "limitation", "context"]
    citations: list[Citation] = Field(default_factory=list, max_length=4)


class SourceReference(BaseModel):
    evidence_id: str
    source_id: str
    source_file: str
    page: int = Field(ge=1)
    url: str = ""
    source_group: str = Field(description="독립 출처를 알 수 없으면 동일 파일을 한 출처로 취급")
    source_type: str = "document"


class SourceRecord(SourceReference):
    content: str = Field(min_length=1, max_length=2000)


class WorkerAnalysis(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=6)
    unknowns: list[str] = Field(default_factory=list, max_length=8)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=8)
    source_limitations: list[str] = Field(default_factory=list, max_length=8)


class WorkerResult(WorkerAnalysis):
    task_id: str
    company_name: str
    perspective: Perspective
    status: Literal["completed", "insufficient_evidence", "failed"]
    attempts: int = Field(ge=0, le=3)
    error_type: str = ""
    evidence_path: str = ""
    sources: list[SourceReference] = Field(default_factory=list, max_length=15)


class PerspectiveSummary(BaseModel):
    company_name: str
    perspective: Perspective
    task_ids: list[str]
    status: Literal["completed", "partial", "insufficient_evidence", "failed"]


class RetryTarget(BaseModel):
    company_name: str
    perspective: Perspective
    reason: str


class QualityCheck(BaseModel):
    passed: bool
    reason: str


class JudgeVerdict(BaseModel):
    groundedness: QualityCheck
    neutrality: QualityCheck
    bias_control: QualityCheck
    coverage: QualityCheck
    retry_targets: list[RetryTarget] = Field(default_factory=list)


class QualityVerdict(JudgeVerdict):
    passed: bool
    evaluator: str
    judge_error: str = ""


def merge_by_task_id(left: dict, right: dict) -> dict:
    """병렬 결과를 task_id로 병합해 재실행 결과의 중복 누적을 막는다."""
    return {**left, **right}


class OrchestratorState(TypedDict, total=False):
    # 작업 페이로드: 원문은 외부 파일, State는 최신 계획·작은 결과·보고서만.
    selected_companies: list[Company]
    target_domain: str
    plan: list[ResearchTask]
    task_registry: Annotated[dict[str, ResearchTask], merge_by_task_id]
    worker_results: Annotated[dict[str, WorkerResult], merge_by_task_id]
    synthesis: list[PerspectiveSummary]
    report_draft: str
    eval_result: QualityVerdict
    report_errors: list[str]

    # 제어 메타: 동일 trace_id/thread_id로 checkpoint와 외부 로그를 연결.
    trace_id: str
    output_dir: str
    execution_mode: Literal["live", "demo"]
    generator_model: str
    judge_model: str
    quality_mode: Literal["hybrid", "deterministic"]
    plan_revision: int
    retrieval_mode: Literal["faiss", "keyword"]
    quality_retry_count: int
    max_quality_retries: int
    max_worker_attempts: int
    max_tasks_per_company: int
    step_count: int
    max_steps: int
    node_status: dict[str, str]
    last_error: str | None
    termination_reason: str
    generate_pdf: bool


class WorkerState(TypedDict):
    task: ResearchTask
    trace_id: str
    output_dir: str
    execution_mode: Literal["live", "demo"]
    generator_model: str
    max_worker_attempts: int
    plan_revision: int
    retrieval_mode: Literal["faiss", "keyword"]
