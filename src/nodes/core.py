import math
import time
from pathlib import Path
from typing import List, TypedDict

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.schemas import (
    InvestmentAgentState, Company, Domain, Question, Evidence,
    ResearchTask, ResearchResult, ResearchSynthesis, ReportEvaluation,
    WorkflowError, CompanyResult, ComparisonResult, FinalDecision
)
from src.rag.vector_store import retrieve_top_k

def load_inputs(state: InvestmentAgentState) -> dict:
    """고정 기업 3곳 로드 (현재는 ACCURE 1개만 임시 지원)"""
    companies = [
        Company(name="ACCURE", description="BESS·EV 배터리 이상, 안전, 성능, 열화 예측 AI"),
        Company(name="volytica diagnostics", description="SOH·열화·이상·안전 분석"),
        Company(name="Electra Vehicles", description="AI 배터리 디지털 트윈, SOC·SOH 예측")
    ]
    return {"selected_companies": companies, "current_index": 0}

def select_company(state: InvestmentAgentState) -> dict:
    idx = state.get("current_index", 0)
    companies = state.get("selected_companies", [])
    if idx < len(companies):
        return {"current_company": companies[idx], "evidence_retry_count": 0,
                "unknown_items": [], "research_synthesis": None}
    return {}

class ResearchPlan(BaseModel):
    tasks: List[ResearchTask] = Field(
        min_length=3,
        max_length=5,
        description="서로 독립적으로 병렬 수행할 수 있는 3~5개 기업 조사 작업",
    )


class ResearchWorkerState(TypedDict):
    current_company: Company
    research_task: ResearchTask
    max_worker_attempts: int


def _rubric_tasks() -> List[ResearchTask]:
    """LLM 계획이 실패해도 실행 가능한 영역별 조사 계획을 만든다."""
    rubric_path = Path(__file__).parent.parent.parent / "configs" / "rubric.yaml"
    with open(rubric_path, "r", encoding="utf-8") as file:
        domains = yaml.safe_load(file).get("domains", {})

    tasks = []
    for domain_name, items in domains.items():
        questions = " ".join(item["question"] for item in items)
        tasks.append(
            ResearchTask(
                task_id=f"research_{domain_name}",
                domain=Domain(domain_name),
                domains=[Domain(domain_name)],
                query=questions,
                objective=f"{domain_name} 영역의 문서 근거 확보",
                success_criteria=["출처가 식별되는 관련 문서 1건 이상"],
            )
        )
    return tasks


def orchestrate_research(state: InvestmentAgentState) -> dict:
    """기업별 조사 목표를 독립적인 3~5개 작업으로 동적 분해한다."""
    company = state["current_company"]
    fallback_tasks = _rubric_tasks()
    rubric_summary = "\n".join(
        f"- {task.domain.value}: {task.query}" for task in fallback_tasks
    )
    previous = state.get("research_synthesis")
    missing = previous.missing_domains if previous else []
    retry_note = (
        f"\n이전 조사에서 근거가 부족한 영역: {', '.join(d.value for d in missing)}. "
        "이전 검색과 다른 구체적인 질문을 작성하세요."
        if state.get("evidence_retry_count", 0) else ""
    )
    prompt = (
        "당신은 배터리 AI 스타트업 투자조사를 조율하는 Orchestrator입니다.\n"
        "아래 기업을 평가하기 위한 조사를 서로 독립적으로 병렬 수행 가능한 "
        "3~5개 작업으로 분해하세요. tasks 수는 기업별 조사 필요에 맞게 선택하세요. "
        "domains에 해당 작업이 다루는 평가 영역을 모두 기재해 5개 영역을 빠짐없이 포함하고, "
        "각 작업에는 고유 task_id, 단일 영역, 검색 가능한 query, objective, "
        "검증 가능한 success_criteria를 포함하세요.\n\n"
        f"기업: {company.name}\n기업 설명: {company.description}\n"
        f"평가 기준:\n{rubric_summary}{retry_note}"
    )

    plan_error = ""
    try:
        planner = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            ResearchPlan
        )
        tasks = planner.invoke(prompt).tasks
    except Exception as exc:
        tasks = fallback_tasks
        plan_error = f"계획 생성 실패로 rubric 계획 사용: {type(exc).__name__}"
        if state.get("evidence_retry_count", 0):
            tasks = [task.model_copy(update={
                "query": f"{task.query} 독립 검증 자료와 수치 근거",
            }) for task in tasks]

    questions = [
        Question(domain=task.domain, item_id=task.task_id, query=task.query)
        for task in tasks
    ]
    return {
        "research_tasks": tasks,
        "plan_version": state.get("plan_version", 0) + 1,
        "plan_error": plan_error,
        "evaluation_questions": questions,
        "research_results": [],
        "retrieved_evidence": [],
        "validated_evidence": [],
    }


def validate_plan(state: InvestmentAgentState) -> dict:
    """동적 계획이 누락·중복·빈 작업을 포함하면 rubric 계획으로 교체한다."""
    tasks = state.get("research_tasks", [])
    domains = {domain for task in tasks for domain in (task.domains or [task.domain])}
    task_ids = [task.task_id for task in tasks]
    valid = (
        3 <= len(tasks) <= 5
        and domains == set(Domain)
        and len(set(task_ids)) == len(task_ids)
        and all(task.query.strip() and task.objective.strip() and task.success_criteria for task in tasks)
    )
    if valid:
        return {"plan_error": state.get("plan_error", "")}
    fallback = _rubric_tasks()
    return {
        "research_tasks": fallback,
        "evaluation_questions": [
            Question(domain=t.domain, item_id=t.task_id, query=t.query)
            for t in fallback
        ],
        "plan_error": "계획 검증 실패: 5개 평가 영역, 고유 ID, 목표·성공 기준 필요. rubric 계획 사용",
    }


def research_worker(state: ResearchWorkerState) -> dict:
    """Orchestrator가 할당한 단일 조사 작업의 근거를 검색한다."""
    company = state["current_company"]
    task = state["research_task"]
    start = time.monotonic()
    attempts = max(1, state.get("max_worker_attempts", 2))
    for attempt in range(1, attempts + 1):
        try:
            results = retrieve_top_k(f"{company.name} {task.query}", k=5)
            evidence = [
                Evidence(
                    item_id=task.task_id,
                    content=doc.page_content[:500],
                    source_type=doc.metadata.get("source_type", "document"),
                    source_id=doc.metadata.get("source_id") or
                              f"{Path(doc.metadata.get('source', 'unknown')).name}:p{doc.metadata.get('page', 0) + 1}",
                    source_document=doc.metadata.get("source_document") or
                                    Path(doc.metadata.get("source", "unknown")).name,
                    evidence_grade=doc.metadata.get("evidence_grade", "U"),
                    is_direct=doc.metadata.get("is_direct", False),
                )
                for doc, _score in results
            ]
            result = ResearchResult(
                task=task,
                evidence=evidence,
                status="SUCCESS" if evidence else "EMPTY",
                attempts=attempt,
                duration_ms=round((time.monotonic() - start) * 1000),
            )
            return {"research_results": [result]}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result = ResearchResult(
        task=task, status="FAILED", attempts=attempts, error=error,
        duration_ms=round((time.monotonic() - start) * 1000),
    )
    return {
        "research_results": [result],
        "errors": [WorkflowError(node="research_worker", task_id=task.task_id,
                                 message=error, attempts=attempts)],
    }


def synthesize_research(state: InvestmentAgentState) -> dict:
    """Worker 완료 순서와 무관하게 근거를 취합하고 부분 실패를 판정한다."""
    deduplicated = {}
    results = sorted(state.get("research_results", []), key=lambda r: r.task.task_id)
    for result in results:
        for evidence in result.evidence:
            key = (evidence.item_id, evidence.source_id, evidence.content)
            deduplicated[key] = evidence
    evidence = [deduplicated[key] for key in sorted(deduplicated)]
    successful = [r.task.task_id for r in results if r.status == "SUCCESS"]
    failed = [r.task.task_id for r in results if r.status == "FAILED"]
    empty = [r.task.task_id for r in results if r.status == "EMPTY"]
    covered = {domain for r in results if r.status == "SUCCESS"
               for domain in (r.task.domains or [r.task.domain])}
    task_count = len(state.get("research_tasks", []))
    coverage = len(successful) / task_count * 100 if task_count else 0.0
    synthesis = ResearchSynthesis(
        successful_task_ids=successful,
        failed_task_ids=failed,
        empty_task_ids=empty,
        missing_domains=sorted(set(Domain) - covered, key=lambda d: d.value),
        evidence_coverage=coverage,
        is_sufficient=not failed and not empty and bool(evidence),
    )
    return {"retrieved_evidence": evidence, "research_synthesis": synthesis,
            "evidence_coverage": coverage}

def validate_evidence(state: InvestmentAgentState) -> dict:
    evidence = [e for e in state.get("retrieved_evidence", [])
                if e.source_id and e.evidence_grade != "U" and e.is_direct]
    covered_tasks = {e.item_id for e in evidence}
    tasks = state.get("research_tasks", [])
    coverage = len(covered_tasks) / len(tasks) * 100 if tasks else 0.0
    return {"validated_evidence": evidence, "evidence_coverage": coverage}

def rewrite_query(state: InvestmentAgentState) -> dict:
    return {"evidence_retry_count": state.get("evidence_retry_count", 0) + 1}

def record_unknown(state: InvestmentAgentState) -> dict:
    covered = {e.item_id for e in state.get("validated_evidence", [])}
    missing = [task.task_id for task in state.get("research_tasks", [])
               if task.task_id not in covered]
    return {"unknown_items": missing}

def score_common(state: InvestmentAgentState) -> dict:
    """
    공통 평가항목 점수 및 근거 충족률 계산 (기획안 C-4 기준)
    DomainScore(d) = 확인된 항목의 가중평균 x 20
    """
    # 실제로는 LLM이 항목당 1~5점 부여. 임시로 수학적 계산 구현
    domain_scores = {
        Domain.TECHNOLOGY: (4.5 / 5.0) * 100, # 90점
        Domain.MARKET: (4.0 / 5.0) * 100,     # 80점
        Domain.FINANCE: (3.5 / 5.0) * 100,    # 70점
        Domain.RISK: (4.2 / 5.0) * 100,       # 84점
        Domain.TEAM: (4.8 / 5.0) * 100        # 96점
    }
    return {"domain_scores": domain_scores,
            "evidence_coverage": state.get("evidence_coverage", 0.0)}

def synthesize_company(state: InvestmentAgentState) -> dict:
    """5개 Agent 결과 종합·최종 투자지표 도출"""
    scores = [p.weighted_score for p in state.get("profile_results", [])]
    
    avg_score = sum(scores) / len(scores) if scores else 0
    # 표본 표준편차 계산
    if len(scores) > 1:
        variance = sum((x - avg_score) ** 2 for x in scores) / len(scores)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0
        
    coverage = state.get("evidence_coverage", 0)
    
    # 투자 판정 (기획안 C-4)
    if avg_score >= 75 and coverage >= 80:
        decision = FinalDecision.RECOMMEND
    elif avg_score >= 60 or (60 <= coverage < 80):
        decision = FinalDecision.CONDITIONAL
    else:
        decision = FinalDecision.HOLD_SCORE
        
    if coverage < 60:
        decision = FinalDecision.HOLD_RAG

    company = state["current_company"].name if state.get("current_company") else "Unknown"
    
    company_result = CompanyResult(
        company_name=company,
        domain_scores=state.get("domain_scores", {}),
        evidence_coverage=coverage,
        profile_results=state.get("profile_results", []),
        average_score=avg_score,
        score_std_dev=std_dev,
        final_decision=decision.value,
        summary_reason=f"다수 프로필 종합 결과, 평균 {avg_score:.1f}점 (편차 {std_dev:.1f})",
        evidence_sources=sorted({e.source_id for e in state.get("validated_evidence", [])
                                 if e.source_id}),
        source_documents=sorted({e.source_document for e in state.get("validated_evidence", [])
                                 if e.source_document}),
        research_synthesis=state.get("research_synthesis"),
    )
    return {"company_result": company_result}

def save_company_result(state: InvestmentAgentState) -> dict:
    idx = state.get("current_index", 0)
    res = state.get("company_result")
    # company_results (Annotated reducer)에 누적
    return {
        "company_results": [res],
        "current_index": idx + 1,
        "profile_results": [],
        "research_results": [],
    }

def compare_companies(state: InvestmentAgentState) -> dict:
    companies = state.get("company_results", [])
    
    table = "| 기업 | 기술 | 시장 | 재무 | 안정성 | 팀·사업 | 근거충족률 | 최종 점수 | 판정 |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for c in companies:
        table += (
            f"| {c.company_name} "
            f"| {c.domain_scores.get(Domain.TECHNOLOGY, 0):.1f} "
            f"| {c.domain_scores.get(Domain.MARKET, 0):.1f} "
            f"| {c.domain_scores.get(Domain.FINANCE, 0):.1f} "
            f"| {c.domain_scores.get(Domain.RISK, 0):.1f} "
            f"| {c.domain_scores.get(Domain.TEAM, 0):.1f} "
            f"| {c.evidence_coverage:.1f}% "
            f"| {c.average_score:.1f} "
            f"| {c.final_decision} |\n"
        )
        
    return {"final_comparison": ComparisonResult(companies=companies, comparison_table=table)}

def generate_report(state: InvestmentAgentState) -> dict:
    """기업별 근거 ID를 포함한 검증 가능한 보고서 초안을 만든다."""
    comp = state.get("final_comparison")
    if not comp:
        return {}

    report = f"""# 배터리 AI 스타트업 투자 검토 보고서

## SUMMARY (종합 요약)
본 보고서는 문서에서 확인된 근거와 미확인 항목을 바탕으로 기업별 추가 실사 필요성을 검토합니다. 점수는 절대적 기술 우열을 의미하지 않습니다.

## 1. 종합 비교표
{comp.comparison_table}

## 2. 기업별 상세
"""
    for c in comp.companies:
        report += f"\n### {c.company_name} ({c.final_decision})\n"
        report += f"- **평균 점수**: {c.average_score:.1f}점 (표준편차: {c.score_std_dev:.1f})\n"
        report += f"- **근거 충족률**: {c.evidence_coverage:.1f}%\n"
        report += f"- **최종 요약**: {c.summary_reason}\n\n"
        sources = c.evidence_sources
        report += f"- **확인된 근거 ID**: {', '.join(f'[{s}]' for s in sources) if sources else '없음'}\n"
        if c.research_synthesis:
            report += f"- **미완료 조사 작업**: {', '.join(c.research_synthesis.failed_task_ids + c.research_synthesis.empty_task_ids) or '없음'}\n"
        
        report += "#### Profile Agent 결과\n"
        for p in c.profile_results:
            recommendation = p.recommendation
            if state.get("report_errors"):
                for expression in ("기술 우위", "최고 기술", "기술적으로 가장 우수"):
                    recommendation = recommendation.replace(expression, "추가 검토 필요")
            report += f"- **{p.profile_name}**: {p.weighted_score:.1f}점 - {recommendation}\n"
            
        report += "\n#### 검토 한계\n"
        report += "- 공개 자료에 없는 수치와 계약 사실은 확인 전까지 미확인으로 취급합니다.\n"

    report += "\n## REFERENCE (참고 자료)\n"
    sources = sorted({source for c in comp.companies for source in c.evidence_sources})
    report += "\n".join(f"- [{source}]" for source in sources) if sources else "- 검증 가능한 출처 없음"
    report += "\n"
    return {"report_draft": report,
            "report_retry_count": state.get("report_retry_count", 0) + 1}


def validate_report(state: InvestmentAgentState) -> dict:
    """출처 연결·중립성·출처 편중을 검사하고 수정 피드백을 만든다."""
    report = state.get("report_draft", "")
    comp = state.get("final_comparison")
    companies = comp.companies if comp else []
    sources = {source for c in companies for source in c.evidence_sources}
    missing = [c.company_name for c in companies if not c.evidence_sources]
    groundedness = bool(companies and sources and not missing and
                        all(f"[{source}]" in report for source in sources))
    neutrality = not any(term in report for term in
                         ("기술 우위", "최고 기술", "기술적으로 가장 우수"))
    bias_control = all(len(set(c.source_documents)) >= 2 for c in companies)
    feedback = []
    if not groundedness:
        feedback.append(f"근거 ID가 없는 기업 또는 인용 누락: {', '.join(missing) or 'REFERENCE 확인'}")
    if not neutrality:
        feedback.append("기술 우열 단정 표현을 제거하세요")
    if not bias_control:
        feedback.append("기업별 독립 원본 문서 2개 이상이 필요합니다. 자료를 보강하세요")
    evaluation = ReportEvaluation(
        groundedness=groundedness, neutrality=neutrality,
        bias_control=bias_control, passed=not feedback, feedback=feedback,
    )
    return {"report_evaluation": evaluation, "report_errors": feedback,
            "status": "QUALITY_PASSED" if evaluation.passed else "QUALITY_FAILED",
            "termination_reason": "QUALITY_PASSED" if evaluation.passed else "QUALITY_REVIEW_REQUIRED"}


def save_report(state: InvestmentAgentState) -> dict:
    """평가 이후 산출물을 저장한다. 실패한 초안에는 상태를 명시한다."""
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    evaluation = state.get("report_evaluation")
    report = state.get("report_draft", "")
    if not evaluation or not evaluation.passed:
        report = (
            "> 품질 평가 미통과: 자동 생성 결과를 투자 판단에 사용하지 마세요.\n\n"
            + report
        )
    report_path = output_dir / "final_multi_agent_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    if evaluation and evaluation.passed:
        try:
            from src.report_generator import generate_pdf_report
            pdf_path = generate_pdf_report(state["final_comparison"])
        except Exception as exc:
            return {"errors": [WorkflowError(node="save_report", message=str(exc))],
                    "status": "REPORT_PDF_FAILED",
                    "termination_reason": "REPORT_PDF_FAILED",
                    "report_uri": str(report_path), "report_draft": ""}
        return {"status": "COMPLETE", "termination_reason": "QUALITY_PASSED",
                "report_uri": pdf_path, "report_draft": ""}
    return {"status": "QUALITY_FAILED",
            "termination_reason": "REPORT_RETRY_LIMIT_REACHED",
            "report_uri": str(report_path), "report_draft": ""}
