import nbformat as nbf
import os

def create_nb(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for ctype, content in cells_data:
        if ctype == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        elif ctype == "code":
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

create_nb("04-LangGraph/01-Main-Graph.ipynb", [
    ("md", "# 04. LangGraph 워크플로우 (신규 설계안 반영)\n사용자가 제공한 직렬 파이프라인 및 조건부 루프 워크플로우 구조입니다."),
    ("code", """from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class InvestmentAgentState(TypedDict):
    company_name: str
    decision: str
    retry_count: int
    max_retries: int

def search_startup_node(state: InvestmentAgentState):
    print("-> 노드 실행: search_startup")
    return {"retry_count": state.get("retry_count", 0) + 1}

def summarize_tech_rag_node(state: InvestmentAgentState):
    print("-> 노드 실행: summarize_tech_rag")
    return {}

def evaluate_market_node(state: InvestmentAgentState):
    print("-> 노드 실행: evaluate_market")
    return {}

def compare_competitors_node(state: InvestmentAgentState):
    print("-> 노드 실행: compare_competitors")
    return {}

def judge_investment_node(state: InvestmentAgentState):
    print("-> 노드 실행: judge_investment")
    decision = "RECOMMEND" if state.get("retry_count", 0) >= 2 else "HOLD"
    return {"decision": decision}

def generate_report_node(state: InvestmentAgentState):
    print("-> 노드 실행: generate_report")
    return {}

def route_after_judgment(state: InvestmentAgentState):
    decision = state.get("decision", "HOLD")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if decision == "RECOMMEND":
        return "generate_report"

    if retry_count < max_retries:
        return "search_startup"
    else:
        return "generate_report"

workflow = StateGraph(InvestmentAgentState)

workflow.add_node("search_startup", search_startup_node)
workflow.add_node("summarize_tech", summarize_tech_rag_node)
workflow.add_node("evaluate_market", evaluate_market_node)
workflow.add_node("compare_competitors", compare_competitors_node)
workflow.add_node("judge_investment", judge_investment_node)
workflow.add_node("generate_report", generate_report_node)

workflow.set_entry_point("search_startup")

workflow.add_edge("search_startup", "summarize_tech")
workflow.add_edge("summarize_tech", "evaluate_market")
workflow.add_edge("evaluate_market", "compare_competitors")
workflow.add_edge("compare_competitors", "judge_investment")

workflow.add_conditional_edges(
    "judge_investment",
    route_after_judgment,
    {
        "generate_report": "generate_report",
        "search_startup": "search_startup",
    },
)

workflow.add_edge("generate_report", END)

app = workflow.compile()
""")
])
