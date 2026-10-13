import os
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver

from src.schemas import InvestmentAgentState
from src.nodes.core import (
    load_inputs,
    select_company,
    build_questions,
    retrieve_evidence,
    validate_evidence,
    web_research,
    rewrite_query,
    record_unknown,
    score_common,
    plan_worker_tasks,
    synthesize_company,
    save_company_result,
    compare_companies,
    generate_report,
    validate_report
)
from src.agents.profile import profile_agent_node

PROFILE_CRITICAL_ITEMS = frozenset({
    # 사업성·성장·안정형이 점수 산출에 반드시 요구하는 루브릭 항목의 합집합
    "fin_01", "fin_02", "team_01", "mkt_01", "rsk_01", "rsk_02",
})


def route_after_common_score(
    state: InvestmentAgentState,
) -> Literal["plan_worker_tasks", "web_research"]:
    """실제 채점 결과가 부족할 때만 누락 항목을 1회 외부 검색합니다."""
    questions = state.get("evaluation_questions", [])
    if (not questions or not os.getenv("TAVILY_API_KEY")
            or state.get("web_research_attempts", 0) >= 1):
        return "plan_worker_tasks"
    item_ids = {question.item_id for question in questions}
    unscored_items = item_ids.intersection(state.get("unknown_items", []))
    coverage = float(state.get("evidence_coverage", 0.0)) / 100.0
    if coverage < 0.75 or PROFILE_CRITICAL_ITEMS.intersection(unscored_items):
        return "web_research"
    return "plan_worker_tasks"

def route_to_profiles(state: InvestmentAgentState):
    """Orchestrator가 State에 기록한 작업 목록만 동적 병렬 실행합니다."""
    return [
        Send("profile_agent", {
            "profile_name": task.profile_name,
            "task_id": task.task_id,
            "task_reason": task.reason,
            "domains": task.domains,
            "domain_scores": state.get("domain_scores", {}),
            "validated_evidence": state.get("validated_evidence", []),
            "current_company": state.get("current_company")
        })
        for task in state.get("worker_tasks", [])
    ]

def route_next_company(state: InvestmentAgentState) -> Literal["select_company", "compare_companies"]:
    """남은 기업이 있는지 확인합니다."""
    idx = state.get("current_index", 0)
    companies = state.get("selected_companies", [])
    if idx < len(companies):
        return "select_company"
    else:
        return "compare_companies"

def route_report_validation(state: InvestmentAgentState) -> Literal["generate_report", "end"]:
    quality = state.get("report_quality")
    if quality and not quality.overall_passed and state.get("report_revision_count", 0) < 2:
        return "generate_report"
    return "end"

def build_graph() -> StateGraph:
    workflow = StateGraph(InvestmentAgentState)

    # 1. 노드 등록
    workflow.add_node("load_inputs", load_inputs)
    workflow.add_node("select_company", select_company)
    workflow.add_node("build_questions", build_questions)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("validate_evidence", validate_evidence)
    workflow.add_node("web_research", web_research)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("record_unknown", record_unknown)
    workflow.add_node("score_common", score_common)
    workflow.add_node("plan_worker_tasks", plan_worker_tasks)
    workflow.add_node("profile_agent", profile_agent_node)
    workflow.add_node("synthesize_company", synthesize_company)
    workflow.add_node("save_company_result", save_company_result)
    workflow.add_node("compare_companies", compare_companies)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("validate_report", validate_report)

    # 2. 엣지 연결
    workflow.set_entry_point("load_inputs")
    workflow.add_edge("load_inputs", "select_company")
    workflow.add_edge("select_company", "build_questions")
    workflow.add_edge("build_questions", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "validate_evidence")
    workflow.add_edge("validate_evidence", "score_common")

    workflow.add_edge("web_research", "validate_evidence")
    workflow.add_edge("rewrite_query", "retrieve_evidence")
    workflow.add_edge("record_unknown", "score_common")

    # Orchestrator 계획 이후 State 기반 Dynamic Fan-out
    # raw 청크 수가 아니라 실제 인용 검증까지 통과한 공통 채점 결과로 검색 여부를 판단합니다.
    workflow.add_conditional_edges(
        "score_common",
        route_after_common_score,
        {
            "web_research": "web_research",
            "plan_worker_tasks": "plan_worker_tasks",
        },
    )
    workflow.add_conditional_edges("plan_worker_tasks", route_to_profiles, ["profile_agent"])
    workflow.add_edge("profile_agent", "synthesize_company")
    
    workflow.add_edge("synthesize_company", "save_company_result")

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

    # 조건부 엣지: 보고서 검증 오류?
    workflow.add_conditional_edges(
        "validate_report",
        route_report_validation,
        {
            "generate_report": "generate_report",
            "end": END,
        }
    )

    return workflow.compile(checkpointer=MemorySaver())

graph = build_graph()
