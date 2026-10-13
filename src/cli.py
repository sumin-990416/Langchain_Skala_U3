"""비대화형 CLI와 SQLite 체크포인트 기반 재개."""

import argparse
import json
import os
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.graph import build_graph
from src.orchestration.checkpoints import checkpoint_serializer
from src.orchestration.planner import DEFAULT_COMPANIES
from src.orchestration.schemas import QualityVerdict
from src.orchestration.tracing import TraceSession
from src.schemas import Company


def _parser():
    parser = argparse.ArgumentParser(description="Orchestrator-Workers 기반 배터리 AI 다관점 평가")
    parser.add_argument("--demo", action="store_true", help="허구 fixture로 API 없이 실행")
    parser.add_argument("--company", action="append", help="평가 기업명, 반복 지정 가능 (최대 5개)")
    parser.add_argument("--domain", default="배터리 AI")
    parser.add_argument("--output-dir", type=Path, help="실행별 출력 폴더 (기본 output/runs/<trace_id>)")
    parser.add_argument("--skip-ingest", action="store_true", help="기존 로컬 FAISS 인덱스 사용")
    parser.add_argument("--retrieval", choices=["faiss", "keyword"], default="faiss",
                        help="faiss 또는 제공 PDF 근거카드의 TF-IDF 검색")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--quality-mode", choices=["hybrid", "deterministic"], default="hybrid")
    parser.add_argument("--generator-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-quality-retries", type=int, choices=range(4), default=2)
    parser.add_argument("--max-worker-attempts", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--resume", type=Path, help="해당 실행 폴더의 SQLite 체크포인트에서 재개")
    parser.add_argument("--pause-before-workers", action="store_true", help="계획 저장 후 일시 중단 (재개 실습용)")
    parser.add_argument("--trace", action="store_true", help="LangSmith trace 전송 활성화 (데모에서도 사용 가능)")
    parser.add_argument("--trace-project", help="LangSmith 프로젝트 이름 (.env 설정보다 우선)")
    return parser


def _print_final(state):
    verdict = QualityVerdict.model_validate(state["eval_result"])
    print(f"품질 평가: {'통과' if verdict.passed else '미달 - 추가 확인 필요'}")
    print(f"종료 사유: {state['termination_reason']}")
    print(f"작업 {len(state['task_registry'])}개 / 품질 재작업 {state['quality_retry_count']}회")
    print(f"보고서: {Path(state['output_dir']) / 'report.md'}")
    pdf = Path(state["output_dir"]) / "report.pdf"
    if pdf.is_file():
        print(f"PDF: {pdf}")
    for error in state.get("report_errors", []):
        print(f"확인 필요: {error}")
    return 0 if verdict.passed and not state.get("report_errors") else 2


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    if args.demo and args.company:
        raise SystemExit("--demo는 고정된 허구 기업을 사용합니다. --company와 함께 지정할 수 없습니다.")
    if args.resume:
        root = args.resume.resolve()
        if not (root / "checkpoint.sqlite").is_file() or not (root / "run-config.json").is_file():
            raise SystemExit("해당 실행의 checkpoint.sqlite와 run-config.json이 필요합니다.")
        saved = json.loads((root / "run-config.json").read_text(encoding="utf-8"))
        trace_id = saved["trace_id"]
        mode = saved["execution_mode"]
        initial = None
    else:
        trace_id = str(uuid.uuid4())
        root = (args.output_dir or Path("output/runs") / trace_id).resolve()
        if (root / "run-config.json").exists() or (root / "checkpoint.sqlite").exists():
            raise SystemExit("기존 실행 폴더입니다. --resume으로 재개하거나 새 출력 폴더를 지정하세요.")
        mode = "demo" if args.demo else "live"
        known = {company.name.casefold(): company for company in DEFAULT_COMPANIES}
        companies = [known.get(name.casefold(), Company(name=name)) for name in args.company or []]
        if mode == "live" and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("live 실행에는 OPENAI_API_KEY가 필요합니다. API 없는 실습은 --demo를 사용하세요.")
        if mode == "live" and args.retrieval == "faiss" and not args.skip_ingest:
            from src.rag.ingest import ingest_documents
            ingest_documents()
        initial = {
            "execution_mode": mode, "selected_companies": companies,
            "target_domain": args.domain, "trace_id": trace_id, "output_dir": str(root),
            "generator_model": args.generator_model, "judge_model": args.judge_model,
            "quality_mode": args.quality_mode, "generate_pdf": not args.no_pdf,
            "max_quality_retries": args.max_quality_retries,
            "max_worker_attempts": args.max_worker_attempts,
            "retrieval_mode": args.retrieval,
        }
    root.mkdir(parents=True, exist_ok=True)
    trace_session = TraceSession(str(root), mode, args.trace, args.trace_project)
    from langgraph.checkpoint.sqlite import SqliteSaver
    config = {
        "configurable": {"thread_id": trace_id},
        "recursion_limit": 60, "max_concurrency": 4,
        "run_name": "battery-orchestrator-workers",
        "tags": ["orchestrator-workers", mode],
        "metadata": {"trace_id": trace_id, "pattern": "orchestrator-workers"},
        **trace_session.graph_config(),
    }
    print(f"모드: {mode} / trace_id: {trace_id}")
    with trace_session, sqlite3.connect(str(root / "checkpoint.sqlite"), check_same_thread=False) as connection:
        saver = SqliteSaver(connection, serde=checkpoint_serializer())
        graph = build_graph(checkpointer=saver,
                            interrupt_before=["worker"] if args.pause_before_workers else None)
        if args.resume:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise SystemExit("저장된 그래프 State를 찾을 수 없습니다.")
            if not snapshot.next:
                return _print_final(snapshot.values)
        trace_session.mark_started()
        for event in graph.stream(initial, config=config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    continue
                if node == "orchestrator":
                    print(f"계획 r{update['plan_revision']}: {len(update['plan'])}개 작업을 동적으로 분할")
                elif node == "worker":
                    for result in update.get("worker_results", {}).values():
                        print(f"Worker {result.task_id}: {result.status} / 시도 {result.attempts}회")
                elif node == "evaluate_report":
                    print("품질 평가: " + ("통과" if update["eval_result"].passed else "미달"))
        snapshot = graph.get_state(config)
        if snapshot.next:
            print(f"체크포인트 저장 후 중단. 재개: python main.py --resume '{root}'")
            return 0
        return _print_final(snapshot.values)
