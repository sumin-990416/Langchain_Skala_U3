from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from src.schemas import InvestmentAgentState
from src.nodes.core import (
    load_inputs,
    select_company,
    build_questions,
    retrieve_evidence,
    validate_evidence,
    rewrite_query,
    record_unknown,
    score_common,
    synthesize_company,
    save_company_result,
    compare_companies,
    generate_report,
    validate_report
)
from src.agents.profile import (
    dispatch_profile_tasks,
    plan_profile_tasks,
    profile_agent_node,
)
from src.nodes.report_quality import (
    evaluate_report_quality,
    plan_report_review,
    report_reviewer_node,
    route_to_report_reviewers,
    synthesize_report_corrections,
)

PROFILE_WORKER_NODES = {
    "technology_focused": "worker_기술중심형",
    "stability_focused": "worker_안정형",
    "growth_focused": "worker_성장형",
    "balanced": "worker_균형형",
    "business_focused": "worker_사업성중심형",
    "기술중심형": "worker_기술중심형",
    "안정형": "worker_안정형",
    "성장형": "worker_성장형",
    "균형형": "worker_균형형",
    "사업성중심형": "worker_사업성중심형",
}


def build_graph_run_config(trace_id: str) -> dict:
    """Attach searchable LangSmith run metadata to each complete graph execution."""
    return {
        "run_name": "battery-startup-investment-report",
        "tags": ["rag", "investment-report", "langgraph"],
        "metadata": {"trace_id": trace_id},
    }


def route_evidence_check(state: InvestmentAgentState) -> Literal["score_common", "rewrite_query", "record_unknown"]:
    """
    근거 충분 여부를 판단하여 라우팅합니다.
    (현재는 임시로 모두 통과한다고 가정)
    """
    is_sufficient = True 
    if is_sufficient:
        return "score_common"
    else:
        # 질문별 재검색 횟수 확인 (여기선 단순화)
        total_retry = sum(state.get("retry_count", {}).values())
        if total_retry < 2:
            return "rewrite_query"
        else:
            return "record_unknown"

def route_to_profiles(state: InvestmentAgentState):
    """Route each planned profile task to its visibly named worker node."""
    sends = []
    for task in state.get("profile_task_plan", []):
        worker_node = PROFILE_WORKER_NODES.get(task["profile_name"])
        if worker_node is None:
            raise ValueError(
                f"프로필 Worker 노드가 등록되지 않았습니다: {task['profile_name']}"
            )
        attempt_count = next(
            status["attempts"]
            for status in state["profile_task_statuses"].values()
            if status["task_id"] == task["task_id"]
        )
        sends.append(Send(worker_node, {
            "task_id": task["task_id"],
            "attempts": attempt_count,
            "profile_name": task["profile_name"],
            "domain_scores": state.get("domain_scores", {}),
            "validated_evidence": state.get("validated_evidence", []),
            "current_company": state.get("current_company")
        }))
    return sends

def route_next_company(state: InvestmentAgentState) -> Literal["select_company", "compare_companies"]:
    """남은 기업이 있는지 확인합니다."""
    idx = state.get("current_index", 0)
    companies = state.get("selected_companies", [])
    if idx < len(companies):
        return "select_company"
    else:
        return "compare_companies"

def route_report_quality(
    state: InvestmentAgentState,
) -> Literal["plan_report_review", "record_quality_failure", "__end__"]:
    verdict = state.get("report_quality_verdict")
    if verdict is None or verdict.judge_error:
        return END
    passed = (
        verdict.groundedness_pass
        and verdict.neutrality_pass
        and verdict.source_bias_pass
    )
    if passed:
        return END
    if state.get("quality_retry_count", 0) >= state.get("quality_max_retries", 1):
        return "record_quality_failure"
    return "plan_report_review"


def record_quality_failure(state: InvestmentAgentState) -> dict:
    verdict = state.get("report_quality_verdict")
    if verdict is None:
        return {}
    failed = [
        label
        for label, passed in (
            ("Groundedness", verdict.groundedness_pass),
            ("중립성", verdict.neutrality_pass),
            ("편향 통제", verdict.source_bias_pass),
        )
        if not passed
    ]
    errors = list(state.get("report_errors", []))
    errors.append(
        f"보고서 품질 기준 미달 (재작업 {state.get('quality_retry_count', 0)}회): "
        + ", ".join(failed)
    )
    return {"report_errors": list(dict.fromkeys(errors))}


def route_report_review(state: InvestmentAgentState):
    sends = route_to_report_reviewers(state)
    return sends if sends else "record_quality_failure"

def build_graph() -> StateGraph:
    workflow = StateGraph(InvestmentAgentState)

    # 1. 노드 등록
    workflow.add_node("load_inputs", load_inputs)
    workflow.add_node("select_company", select_company)
    workflow.add_node("build_questions", build_questions)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("validate_evidence", validate_evidence)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("record_unknown", record_unknown)
    workflow.add_node("score_common", score_common)
    workflow.add_node("orchestrator_plan_workers", plan_profile_tasks)
    workflow.add_node("orchestrator_dispatch_workers", dispatch_profile_tasks)
    for worker_node in set(PROFILE_WORKER_NODES.values()):
        workflow.add_node(worker_node, profile_agent_node)
    workflow.add_node("synthesizer_aggregate_company", synthesize_company)
    workflow.add_node("save_company_result", save_company_result)
    workflow.add_node("compare_companies", compare_companies)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("validate_report", validate_report)
    workflow.add_node("evaluate_report_quality", evaluate_report_quality)
    workflow.add_node("record_quality_failure", record_quality_failure)
    workflow.add_node("plan_report_review", plan_report_review)
    workflow.add_node("report_reviewer", report_reviewer_node)
    workflow.add_node("synthesize_report_corrections", synthesize_report_corrections)

    # 2. 엣지 연결
    workflow.set_entry_point("load_inputs")
    workflow.add_edge("load_inputs", "select_company")
    workflow.add_edge("select_company", "build_questions")
    workflow.add_edge("build_questions", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "validate_evidence")

    # 조건부 엣지: 근거 충분?
    workflow.add_conditional_edges(
        "validate_evidence",
        route_evidence_check,
        {
            "score_common": "score_common",
            "rewrite_query": "rewrite_query",
            "record_unknown": "record_unknown"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve_evidence")
    workflow.add_edge("record_unknown", "score_common")

    # Plan first, then dynamically fan out only the tasks stored in State.
    worker_nodes = sorted(set(PROFILE_WORKER_NODES.values()))
    workflow.add_edge("score_common", "orchestrator_plan_workers")
    workflow.add_edge("orchestrator_plan_workers", "orchestrator_dispatch_workers")
    workflow.add_conditional_edges(
        "orchestrator_dispatch_workers",
        route_to_profiles,
        worker_nodes,
    )
    for worker_node in worker_nodes:
        workflow.add_edge(worker_node, "synthesizer_aggregate_company")
    
    workflow.add_edge("synthesizer_aggregate_company", "save_company_result")

    # 조건부 엣지: 남은 기업?
    workflow.add_conditional_edges(
        "save_company_result",
        route_next_company,
        {
            "select_company": "select_company",
            "compare_companies": "compare_companies"
        }
    )

    workflow.add_edge("compare_companies", "generate_report")
    workflow.add_edge("generate_report", "validate_report")
    workflow.add_edge("validate_report", "evaluate_report_quality")

    workflow.add_conditional_edges(
        "evaluate_report_quality",
        route_report_quality,
        {
            "plan_report_review": "plan_report_review",
            "record_quality_failure": "record_quality_failure",
            END: END,
        }
    )
    workflow.add_edge("record_quality_failure", END)
    workflow.add_conditional_edges(
        "plan_report_review",
        route_report_review,
        {
            "report_reviewer": "report_reviewer",
            "record_quality_failure": "record_quality_failure",
        },
    )
    workflow.add_edge("report_reviewer", "synthesize_report_corrections")
    workflow.add_edge("synthesize_report_corrections", "generate_report")

    return workflow.compile()

graph = build_graph()
