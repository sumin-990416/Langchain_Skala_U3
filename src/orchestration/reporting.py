"""Deterministic synthesis and neutral report assembly.

Workers author the analysis. These nodes preserve their citations, uncertainty,
and failures without introducing a score, a ranking, or a recommendation.
"""

from collections import defaultdict

from src.orchestration.observability import advance, log_decision, write_json
from src.orchestration.schemas import (
    OrchestratorState,
    PERSPECTIVE_LABELS,
    Perspective,
    PerspectiveSummary,
    ResearchTask,
    SourceReference,
    WorkerResult,
)


def _tasks(state: OrchestratorState) -> dict[str, ResearchTask]:
    return {
        key: ResearchTask.model_validate(value)
        for key, value in state.get("task_registry", {}).items()
    }


def _results(state: OrchestratorState) -> dict[str, WorkerResult]:
    return {
        key: WorkerResult.model_validate(value)
        for key, value in state.get("worker_results", {}).items()
    }


def _company_names(state: OrchestratorState, tasks: dict[str, ResearchTask]) -> list[str]:
    names = [company.name for company in state.get("selected_companies", [])]
    return list(dict.fromkeys(names + [task.company_name for task in tasks.values()]))


def synthesizer_node(state: OrchestratorState) -> dict:
    """Aggregate the latest results for every registered task, including failures."""
    update = advance(state, "synthesizer")
    tasks, results = _tasks(state), _results(state)
    grouped: dict[tuple[str, Perspective], list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        grouped[(task.company_name, task.perspective)].append(task_id)

    summaries = []
    for company in _company_names(state, tasks):
        for perspective in Perspective:
            task_ids = sorted(grouped[(company, perspective)])
            matching = [
                results[task_id]
                for task_id in task_ids
                if task_id in results
                and results[task_id].company_name == company
                and results[task_id].perspective == perspective
            ]
            completed = [result for result in matching if result.status == "completed" and result.claims]
            if completed and len(completed) == len(task_ids):
                status = "completed"
            elif completed:
                status = "partial"
            elif matching and len(matching) == len(task_ids) and all(
                result.status == "failed" for result in matching
            ):
                status = "failed"
            else:
                status = "insufficient_evidence"
            summaries.append(PerspectiveSummary(
                company_name=company,
                perspective=perspective,
                task_ids=task_ids,
                status=status,
            ))

    write_json(state["output_dir"], f"synthesis-r{state.get('plan_revision', 0)}.json", [
        summary.model_dump(mode="json") for summary in summaries
    ])
    log_decision(state, "synthesizer", "aggregate", "task_id별 최신 결과를 회사·관점별로 집계", task_count=len(tasks))
    return {**update, "synthesis": summaries}


_STATUS_LABELS = {
    "completed": "자료 확인",
    "partial": "일부 작업 미완료",
    "insufficient_evidence": "근거 부족",
    "failed": "작업 실패",
}
_CLAIM_LABELS = {"support": "확인 내용", "limitation": "한계·상충", "context": "적용 맥락"}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _display(value: str) -> str:
    return " ".join(value.split())


def generate_report_node(state: OrchestratorState) -> dict:
    """Build a Markdown draft; persistence and PDF export belong to finalize."""
    update = advance(state, "generate_report")
    tasks, results = _tasks(state), _results(state)
    summaries = [PerspectiveSummary.model_validate(item) for item in state.get("synthesis", [])]
    summary_map = {(item.company_name, item.perspective): item for item in summaries}
    companies = _company_names(state, tasks)
    references: dict[str, SourceReference] = {}
    for task_id in sorted(tasks):
        if task_id in results:
            for source in results[task_id].sources:
                references.setdefault(source.evidence_id, source)

    lines = [
        "# 배터리 AI 스타트업 다관점 기술 평가 보고서",
        "",
        "## SUMMARY",
        "",
        f"- 평가 도메인: {state.get('target_domain', '배터리 AI')}",
        f"- 검색 방식: {'제공 PDF 근거카드 TF-IDF 검색' if state.get('retrieval_mode') == 'keyword' else 'FAISS 벡터 검색'}",
        f"- 평가 대상: {', '.join(companies) if companies else '평가 대상 없음'}",
        "- 평가 관점: 기술 성숙도, 시장성, 이해관계자, 도메인 적용",
        "- 공개자료에서 확인한 내용과 한계·상충, 미확인 사항을 함께 정리했습니다.",
        f"- 실행 식별자: `{state.get('trace_id', '')}` · 계획 revision: {state.get('plan_revision', 0)}",
        "",
        "| 기업 | 기술 성숙도 | 시장성 | 이해관계자 | 도메인 적용 |",
        "|---|---|---|---|---|",
    ]
    for company in companies:
        statuses = [
            _STATUS_LABELS[summary_map[(company, perspective)].status]
            if (company, perspective) in summary_map else "근거 부족"
            for perspective in Perspective
        ]
        lines.append(f"| {company} | " + " | ".join(statuses) + " |")

    cited_ids = []
    for company in companies:
        lines.extend(["", f"## {company}"])
        for perspective in Perspective:
            summary = summary_map.get((company, perspective))
            task_ids = summary.task_ids if summary else []
            matched = [
                results[task_id] for task_id in task_ids
                if task_id in results and results[task_id].company_name == company
                and results[task_id].perspective == perspective
            ]
            status = summary.status if summary else "insufficient_evidence"
            lines.extend(["", f"### {PERSPECTIVE_LABELS[perspective]}", "", f"- 조사 상태: {_STATUS_LABELS[status]}"])
            seen_claims = set()
            for result in matched:
                for claim in result.claims:
                    key = (claim.text, tuple((citation.evidence_id, citation.quote) for citation in claim.citations))
                    if key in seen_claims:
                        continue
                    seen_claims.add(key)
                    ids = _unique([citation.evidence_id for citation in claim.citations])
                    citations = " ".join(f"[{evidence_id}]" for evidence_id in ids)
                    lines.append(f"- **{_CLAIM_LABELS[claim.kind]}**: {_display(claim.text)} {citations}".rstrip())
                    for citation in claim.citations:
                        cited_ids.append(citation.evidence_id)
                        source = references.get(citation.evidence_id)
                        location = f"{source.source_file}, {source.page}쪽" if source else "출처 위치 미확인"
                        lines.append(f"  - 근거 [{citation.evidence_id}] ({location}): “{_display(citation.quote)}”")
            if not seen_claims:
                lines.append("- 확인 내용: 인용 가능한 주장이 확보되지 않았습니다.")
            unknowns = _unique([value for result in matched for value in result.unknowns])
            questions = _unique([value for result in matched for value in result.follow_up_questions])
            limitations = _unique([value for result in matched for value in result.source_limitations])
            lines.extend(["", "#### 미확인 사항 및 추가 질문", ""])
            lines.extend(f"- 미확인: {value}" for value in unknowns)
            lines.extend(f"- 추가 질문: {value}" for value in questions)
            if not unknowns and not questions:
                lines.append("- 추가로 명시된 미확인 항목 없음. 이는 모든 사항의 검증 완료를 의미하지 않습니다.")
            lines.extend(["", "#### 출처 한계 및 작업 상태", ""])
            lines.extend(f"- 출처 한계: {value}" for value in limitations)
            if not limitations:
                lines.append("- 출처 한계: 입력 문서와 검색된 청크 범위에 한정되며, 문서 밖의 사실은 확인하지 않았습니다.")
            for task_id in task_ids:
                result = results.get(task_id)
                if not result:
                    lines.append(f"- 작업 미완료: `{task_id}` 결과가 없습니다.")
                elif result.status != "completed":
                    detail = f" · 오류 유형: {result.error_type}" if result.error_type else ""
                    lines.append(f"- Worker 상태: `{task_id}` · {_STATUS_LABELS[result.status]} · 시도 {result.attempts}회{detail}")

    lines.extend(["", "## REFERENCE", ""])
    for evidence_id in _unique(cited_ids):
        source = references.get(evidence_id)
        if source:
            url = f" · <{source.url}>" if source.url else ""
            lines.append(
                f"- [{evidence_id}] {source.source_id} · {source.source_file} · {source.page}쪽"
                f" · 출처 그룹: {source.source_group} · 유형: {source.source_type}{url}"
            )
        else:
            lines.append(f"- [{evidence_id}] 원문 위치 연결 실패")
    if not cited_ids:
        lines.append("- 연결된 근거가 없습니다. 이 보고서는 근거 부족 상태입니다.")
    lines.extend(["", "인용은 해당 문서의 진술을 추적하기 위한 것이며, 독립 검증을 대신하지 않습니다.", ""])
    report = "\n".join(lines)
    log_decision(state, "generate_report", "assemble", "Worker 주장·직접 인용·원문 위치·실패를 결정적으로 조립", cited_sources=len(_unique(cited_ids)))
    return {**update, "report_draft": report}
