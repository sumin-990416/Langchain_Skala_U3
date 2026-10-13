from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.schemas import InvestmentAgentState
from src.nodes.core import (
    load_inputs,
    select_company,
    orchestrate_research,
    validate_plan,
    research_worker,
    synthesize_research,
    validate_evidence,
    rewrite_query,
    record_unknown,
    score_common,
    synthesize_company,
    save_company_result,
    compare_companies,
    generate_report,
    validate_report,
    save_report,
)
from src.agents.profile import profile_agent_node, get_all_profiles

def route_evidence_check(state: InvestmentAgentState) -> Literal["score_common", "rewrite_query", "record_unknown"]:
    """검증된 근거와 재시도 한도에 따라 다음 단계를 정한다."""
    synthesis = state.get("research_synthesis")
    sufficient = bool(synthesis and synthesis.is_sufficient)
    sufficient = sufficient and state.get("evidence_coverage", 0) >= 80
    sufficient = sufficient and len({e.source_document for e in state.get("validated_evidence", [])
                                     if e.source_document}) >= 2
    if sufficient:
        return "score_common"
    if state.get("evidence_retry_count", 0) < state.get("max_evidence_retries", 2):
        return "rewrite_query"
    return "record_unknown"

def route_to_profiles(state: InvestmentAgentState):
    """공통 점수 후 5개 프로필 에이전트를 병렬 실행합니다."""
    if not (state.get("research_synthesis") and
            state["research_synthesis"].is_sufficient and
            state.get("evidence_coverage", 0) >= 80 and
            len({e.source_document for e in state.get("validated_evidence", [])
                 if e.source_document}) >= 2):
        return "synthesize_company"
    return [
        Send("profile_agent", {
            "profile_name": p,
            "domain_scores": state.get("domain_scores", {}),
            "validated_evidence": state.get("validated_evidence", []),
            "current_company": state.get("current_company")
        })
        for p in get_all_profiles()
    ]

def assign_research_workers(state: InvestmentAgentState):
    """Orchestrator가 만든 작업 수만큼 Research Worker를 병렬 fan-out합니다."""
    company = state.get("current_company")
    return [
        Send(
            "research_worker",
            {"current_company": company, "research_task": task,
             "max_worker_attempts": 2},
        )
        for task in state.get("research_tasks", [])
    ]

def route_next_company(state: InvestmentAgentState) -> Literal["select_company", "compare_companies"]:
    """남은 기업이 있는지 확인합니다."""
    idx = state.get("current_index", 0)
    companies = state.get("selected_companies", [])
    if idx < len(companies):
        return "select_company"
    else:
        return "compare_companies"

def route_report_validation(state: InvestmentAgentState) -> Literal["generate_report", "save_report"]:
    """품질 미달 보고서를 정해진 횟수만큼 수정한다."""
    evaluation = state.get("report_evaluation")
    if evaluation and evaluation.passed:
        return "save_report"
    if state.get("report_retry_count", 0) < state.get("max_report_retries", 2):
        return "generate_report"
    return "save_report"

def build_graph(checkpointer=None):
    workflow = StateGraph(InvestmentAgentState)

    # 1. 노드 등록
    workflow.add_node("load_inputs", load_inputs)
    workflow.add_node("select_company", select_company)
    workflow.add_node("orchestrator", orchestrate_research)
    workflow.add_node("validate_plan", validate_plan)
    workflow.add_node("research_worker", research_worker)
    workflow.add_node("synthesize_research", synthesize_research)
    workflow.add_node("validate_evidence", validate_evidence)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("record_unknown", record_unknown)
    workflow.add_node("score_common", score_common)
    workflow.add_node("profile_agent", profile_agent_node)
    workflow.add_node("synthesize_company", synthesize_company)
    workflow.add_node("save_company_result", save_company_result)
    workflow.add_node("compare_companies", compare_companies)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("validate_report", validate_report)
    workflow.add_node("save_report", save_report)

    # 2. 엣지 연결
    workflow.set_entry_point("load_inputs")
    workflow.add_edge("load_inputs", "select_company")
    workflow.add_edge("select_company", "orchestrator")
    workflow.add_edge("orchestrator", "validate_plan")
    workflow.add_conditional_edges("validate_plan", assign_research_workers,
                                   ["research_worker"])
    workflow.add_edge("research_worker", "synthesize_research")
    workflow.add_edge("synthesize_research", "validate_evidence")

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

    workflow.add_edge("rewrite_query", "orchestrator")
    workflow.add_edge("record_unknown", "score_common")

    # 병렬 에이전트 실행
    workflow.add_conditional_edges("score_common", route_to_profiles,
                                   ["profile_agent", "synthesize_company"])
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
            "save_report": "save_report"
        }
    )

    workflow.add_edge("save_report", END)

    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()
