"""Orchestrator-Workers: 계획 → 동적 fan-out → 집계 → 품질 평가 → 제한 재작업."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.orchestration.finalize import finalize_node
from src.orchestration.planner import initialize_node, orchestrator_node
from src.orchestration.quality import evaluate_report_node
from src.orchestration.reporting import generate_report_node, synthesizer_node
from src.orchestration.schemas import OrchestratorState
from src.orchestration.worker import worker_node


def route_to_workers(state: OrchestratorState):
    """Workers의 수와 입력은 사전에 저장된 plan에서만 결정한다."""
    if not state.get("plan"):
        raise ValueError("Worker 실행 전에 구조화 계획이 필요합니다.")
    return [Send("worker", {
        "task": task,
        "trace_id": state["trace_id"], "output_dir": state["output_dir"],
        "execution_mode": state["execution_mode"],
        "generator_model": state["generator_model"],
        "max_worker_attempts": state["max_worker_attempts"],
        "plan_revision": state["plan_revision"],
        "retrieval_mode": state.get("retrieval_mode", "faiss"),
    }) for task in state["plan"]]


def route_after_quality(state: OrchestratorState) -> str:
    if state["eval_result"].passed:
        return "finalize"
    if (state["quality_retry_count"] >= state["max_quality_retries"]
            or state["step_count"] + 5 >= state["max_steps"]):
        return "finalize"
    return "orchestrator"


def build_graph(checkpointer=None, interrupt_before=None):
    workflow = StateGraph(OrchestratorState)
    for name, node in [
        ("initialize", initialize_node), ("orchestrator", orchestrator_node),
        ("worker", worker_node), ("synthesizer", synthesizer_node),
        ("generate_report", generate_report_node),
        ("evaluate_report", evaluate_report_node), ("finalize", finalize_node),
    ]:
        workflow.add_node(name, node)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "orchestrator")
    workflow.add_conditional_edges("orchestrator", route_to_workers, ["worker"])
    workflow.add_edge("worker", "synthesizer")
    workflow.add_edge("synthesizer", "generate_report")
    workflow.add_edge("generate_report", "evaluate_report")
    workflow.add_conditional_edges("evaluate_report", route_after_quality,
                                   {"orchestrator": "orchestrator", "finalize": "finalize"})
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


graph = build_graph()
