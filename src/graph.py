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
from src.agents.profile import profile_agent_node, get_all_profiles

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
    """공통 점수 후 5개 프로필 에이전트를 병렬 실행합니다."""
    return [
        Send("profile_agent", {
            "profile_name": p,
            "domain_scores": state.get("domain_scores", {}),
            "validated_evidence": state.get("validated_evidence", []),
            "current_company": state.get("current_company")
        })
        for p in get_all_profiles()
    ]

def route_next_company(state: InvestmentAgentState) -> Literal["select_company", "compare_companies"]:
    """남은 기업이 있는지 확인합니다."""
    idx = state.get("current_index", 0)
    companies = state.get("selected_companies", [])
    if idx < len(companies):
        return "select_company"
    else:
        return "compare_companies"

def route_report_validation(state: InvestmentAgentState) -> str:
    """검증 오류는 결과에 남기고 종료하여 무한 재생성 루프를 방지합니다."""
    return END

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

    # 병렬 에이전트 실행
    workflow.add_conditional_edges("score_common", route_to_profiles, ["profile_agent"])
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
            END: END
        }
    )

    return workflow.compile()

graph = build_graph()
