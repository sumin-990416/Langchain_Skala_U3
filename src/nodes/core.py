import math
import re
from pathlib import Path
import yaml
from langchain_openai import ChatOpenAI
from src.schemas import (
    InvestmentAgentState, Company, Domain, Question, Evidence, 
    CompanyResult, ComparisonResult, FinalDecision, ResultCategory
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
        return {"current_company": companies[idx]}
    return {}

def build_questions(state: InvestmentAgentState) -> dict:
    """rubric.yaml의 전체 평가 항목을 검색 질문으로 연결합니다."""
    rubric_path = Path(__file__).parent.parent.parent / "configs" / "rubric.yaml"
    with rubric_path.open(encoding="utf-8") as handle:
        domains = yaml.safe_load(handle)["domains"]
    questions = [
        Question(domain=Domain(domain_name), item_id=item["id"], query=item["question"])
        for domain_name, items in domains.items()
        for item in items
    ]
    return {"evaluation_questions": questions}


def _source_id_from_document(doc) -> str:
    """본문의 공개 근거 ID를 우선하고 없으면 파일·페이지 기반 ID를 만듭니다."""
    public_ids = re.findall(r"\b[AVEG]-?\d{2}\b", doc.page_content)
    if public_ids:
        return public_ids[0]
    metadata = doc.metadata
    if metadata.get("source_id"):
        return str(metadata["source_id"])
    source_name = Path(str(metadata.get("source", "document"))).stem
    page = int(metadata.get("page", 0)) + 1
    return f"{source_name}:p{page}"

def retrieve_evidence(state: InvestmentAgentState) -> dict:
    company_name = state["current_company"].name if state.get("current_company") else ""
    retrieved = []
    
    for q in state.get("evaluation_questions", []):
        search_query = f"{company_name} {q.query}"
        # 설계안에 맞춰 Top-K 5 설정
        results = retrieve_top_k(search_query, k=5)
        
        for doc, score in results:
            ev = Evidence(
                item_id=q.item_id,
                content=doc.page_content,
                source_type=doc.metadata.get("source_type", "document"),
                source_id=_source_id_from_document(doc),
                evidence_grade=doc.metadata.get("evidence_grade", "B"),
                is_direct=True
            )
            retrieved.append(ev)

    return {"retrieved_evidence": retrieved}

def validate_evidence(state: InvestmentAgentState) -> dict:
    # 관련성, 출처등급, 직접성, 범위, 상충 판정 (임시 전부 PASS)
    return {"validated_evidence": state.get("retrieved_evidence", [])}

def rewrite_query(state: InvestmentAgentState) -> dict:
    return {}

def record_unknown(state: InvestmentAgentState) -> dict:
    return {"unknown_items": ["일부 재무 매출 미공개"]}

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
    return {"domain_scores": domain_scores, "evidence_coverage": 85.0}

def synthesize_company(state: InvestmentAgentState) -> dict:
    """5개 Agent 결과 종합·최종 투자지표 도출"""
    profiles = state.get("profile_results", [])
    scores = [p.weighted_score for p in profiles if p.weighted_score is not None]
    unscored_profiles = [p.profile_name for p in profiles if p.weighted_score is None]
    rag_holds = [p.profile_name for p in profiles if p.decision == FinalDecision.HOLD_RAG]
    gate_holds = [p.profile_name for p in profiles if p.decision == FinalDecision.HOLD_GATE]
    errored_profiles = [
        p.profile_name for p in profiles
        if p.result_category == ResultCategory.EXECUTION_ERROR
    ]
    
    # 미산출 점수를 0점으로 대체하지 않고, 평가가 완료된 프로필만 집계한다.
    avg_score = sum(scores) / len(scores) if scores else None
    # 기존과 동일하게 평가된 프로필 점수의 모집단 표준편차를 계산한다.
    if len(scores) > 1:
        variance = sum((x - avg_score) ** 2 for x in scores) / len(scores)
        std_dev = math.sqrt(variance)
    elif scores:
        std_dev = 0.0
    else:
        std_dev = None
        
    profile_coverages = [
        p.evidence_coverage for p in profiles if p.evidence_coverage is not None
    ]
    coverage = (
        sum(profile_coverages) / len(profile_coverages)
        if profile_coverages
        else state.get("evidence_coverage", 0)
    )
    
    # 투자 판정 (기획안 C-4)
    # 다른 프로필의 높은 점수로 근거 부족·추가 실사 판정이 사라지지 않게 한다.
    if avg_score is None or unscored_profiles or rag_holds or errored_profiles or coverage < 60:
        decision = FinalDecision.HOLD_RAG
    elif gate_holds:
        decision = FinalDecision.HOLD_GATE
    elif avg_score >= 75 and coverage >= 80:
        decision = FinalDecision.RECOMMEND
    elif avg_score >= 60 or (60 <= coverage < 80):
        decision = FinalDecision.CONDITIONAL
    else:
        decision = FinalDecision.HOLD_SCORE

    if errored_profiles:
        result_category = ResultCategory.EXECUTION_ERROR
    elif avg_score is None:
        result_category = ResultCategory.INSUFFICIENT_EVIDENCE
    elif unscored_profiles or rag_holds or any(
        p.result_category == ResultCategory.PARTIAL_EVIDENCE for p in profiles
    ):
        result_category = ResultCategory.PARTIAL_EVIDENCE
    elif gate_holds:
        result_category = ResultCategory.NEEDS_DUE_DILIGENCE
    else:
        result_category = ResultCategory.COMPLETE
        
    company = state["current_company"].name if state.get("current_company") else "Unknown"

    if avg_score is None:
        summary = "평가 가능한 프로필 점수가 없어 점수 없음 / 판단 유보"
    elif unscored_profiles:
        summary = (
            f"전체 {len(profiles)}개 중 평가 완료 {len(scores)}개 프로필의 참고 평균 "
            f"{avg_score:.1f}점 (편차 {std_dev:.1f}); 전체 평가가 완료되지 않아 판단 유보"
        )
    else:
        summary = f"다수 프로필 종합 결과, 평균 {avg_score:.1f}점 (편차 {std_dev:.1f})"
    if unscored_profiles:
        summary += f". 점수 미산출 {len(unscored_profiles)}개: {', '.join(unscored_profiles)}"
    if rag_holds:
        summary += f". 근거 부족 판단 유보 프로필: {', '.join(rag_holds)}"
    if gate_holds:
        summary += f". 추가 실사 필요 프로필: {', '.join(gate_holds)}"
    if errored_profiles:
        summary += f". 평가 오류 프로필: {', '.join(errored_profiles)}"
    if coverage < 60:
        summary += f". 공통 근거 충족률 {coverage:.1f}%로 판단 유보"
    
    company_result = CompanyResult(
        company_name=company,
        domain_scores=state.get("domain_scores", {}),
        evidence_coverage=coverage,
        profile_results=profiles,
        average_score=avg_score,
        score_std_dev=std_dev,
        final_decision=decision.value,
        result_category=result_category,
        summary_reason=summary
    )
    return {"company_result": company_result}

def save_company_result(state: InvestmentAgentState) -> dict:
    idx = state.get("current_index", 0)
    res = state.get("company_result")
    # company_results (Annotated reducer)에 누적
    return {"company_results": [res], "current_index": idx + 1, "profile_results": []} # 리스트 클리어 트릭 필요하지만 지금은 reducer가 처리

def _format_score(score: float | None, suffix: str = "") -> str:
    """미산출은 숫자나 0점 대신 명시적인 문구로 표시한다."""
    return "점수 없음" if score is None else f"{score:.1f}{suffix}"


def _has_partial_scores(company: CompanyResult) -> bool:
    return company.average_score is not None and any(
        profile.weighted_score is None for profile in company.profile_results
    )


def compare_companies(state: InvestmentAgentState) -> dict:
    companies = state.get("company_results", [])
    
    table = "| 기업 | 기술 | 시장 | 재무 | 안정성 | 팀·사업 | 근거충족률 | 최종 점수 | 결과 구분 | 판정 |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    for c in companies:
        average_label = _format_score(c.average_score)
        if _has_partial_scores(c):
            average_label += " (일부 프로필 참고 평균)"
        table += (
            f"| {c.company_name} "
            f"| {_format_score(c.domain_scores.get(Domain.TECHNOLOGY))} "
            f"| {_format_score(c.domain_scores.get(Domain.MARKET))} "
            f"| {_format_score(c.domain_scores.get(Domain.FINANCE))} "
            f"| {_format_score(c.domain_scores.get(Domain.RISK))} "
            f"| {_format_score(c.domain_scores.get(Domain.TEAM))} "
            f"| {c.evidence_coverage:.1f}% "
            f"| {average_label} "
            f"| {c.result_category.value} "
            f"| {c.final_decision} |\n"
        )
        
    return {"final_comparison": ComparisonResult(companies=companies, comparison_table=table)}

def generate_report(state: InvestmentAgentState) -> dict:
    """5쪽 이내의 마크다운 보고서 작성 및 최종 PDF 리포트 생성"""
    comp = state.get("final_comparison")
    if not comp:
        return {"report_draft": "", "report_errors": ["종합 비교 결과가 없습니다."]}

    report = f"""# Battery Intelligence 스타트업 투자검토 보고서

## SUMMARY

{comp.comparison_table}

## 기업별 투자검토
"""

    source_ids = []
    for company in comp.companies:
        average_title = "일부 프로필 참고 평균" if _has_partial_scores(company) else "최종 점수"
        report += f"\n### {company.company_name}\n\n"
        report += f"- **결과 구분**: {company.result_category.value}\n"
        report += f"- **판정**: {company.final_decision}\n"
        report += f"- **{average_title}**: {_format_score(company.average_score, '점')}\n"
        report += f"- **근거 충족률**: {company.evidence_coverage:.1f}%\n"
        report += f"- **종합 의견**: {company.summary_reason}\n"

        report += "\n#### Profile Agent 결과\n"
        for profile in company.profile_results:
            score = _format_score(profile.weighted_score, "점")
            decision = profile.decision.value if profile.decision else "판정 미확인"
            coverage = (
                "미산출" if profile.evidence_coverage is None
                else f"{profile.evidence_coverage:.1f}%"
            )
            report += (
                f"\n- **{profile.profile_name}**: {score} / {profile.result_category.value} "
                f"/ {decision} / 근거 {coverage}\n"
            )
            if profile.unknown_items:
                report += "  - 근거 부족: " + "; ".join(profile.unknown_items) + "\n"
            if profile.error_message:
                report += f"  - 평가 오류: {profile.error_message}\n"
            if profile.due_diligence_questions:
                report += "  - 추가 실사: " + "; ".join(profile.due_diligence_questions) + "\n"
            source_ids.extend(profile.supporting_evidence_ids)
            source_ids.extend(profile.contrary_evidence_ids)

    report += "\n## REFERENCE\n"
    unique_source_ids = list(dict.fromkeys(source_ids))
    if unique_source_ids:
        report += "".join(f"- [{source_id}] 검색·검증된 입력 문서 근거\n" for source_id in unique_source_ids)
    else:
        report += "- 보고서 판정에 연결된 유효 근거 ID 없음\n"

    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "final_multi_agent_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    errors = []
    try:
        from src.report_generator import generate_pdf_report
        pdf_path = generate_pdf_report(comp)
        if pdf_path:
            print(f"🎉 고품질 PDF 보고서가 생성되었습니다: {pdf_path}")
    except Exception as e:
        error_message = f"PDF 생성 실패: {type(e).__name__}: {' '.join(str(e).split())[:240]}"
        errors.append(error_message)
        print(error_message)

    return {"report_draft": report, "report_errors": errors}

def validate_report(state: InvestmentAgentState) -> dict:
    """산출물 누락을 기록하되 그래프가 무한 재생성 루프에 빠지지 않게 합니다."""
    errors = list(state.get("report_errors", []))
    output_dir = Path(__file__).parent.parent.parent / "output"
    if not state.get("report_draft", "").strip():
        errors.append("마크다운 보고서 내용이 비어 있습니다.")
    if not (output_dir / "final_multi_agent_report.md").exists():
        errors.append("마크다운 보고서 파일이 생성되지 않았습니다.")
    if not (output_dir / "final_report.pdf").exists():
        errors.append("PDF 보고서 파일이 생성되지 않았습니다.")
    return {"report_errors": list(dict.fromkeys(errors))}
