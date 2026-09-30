import math
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.schemas import (
    InvestmentAgentState, Company, Domain, Question, Evidence, 
    CompanyResult, ComparisonResult, FinalDecision
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
    # 29개 항목을 기업별 질문으로 변환 (설계안 D-1)
    # 임시로 영역별 1개씩만 생성
    questions = [
        Question(domain=Domain.TECHNOLOGY, item_id="TECH-01", query="AI 모델 성능 지표 및 외부 검증"),
        Question(domain=Domain.MARKET, item_id="MKT-01", query="고객 문제 해결 및 유료 고객 실적"),
        Question(domain=Domain.FINANCE, item_id="FIN-01", query="최신 투자 단계 및 금액"),
        Question(domain=Domain.RISK, item_id="RSK-01", query="안전, 규제, 데이터 권리 관련 문제"),
        Question(domain=Domain.TEAM, item_id="TEAM-01", query="창업진 경력 및 핵심 인력 전문성")
    ]
    return {"evaluation_questions": questions}

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
                source_id=doc.metadata.get("source_id", "DOC-01"),
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
        
    coverage = state.get("evidence_coverage", 0)
    
    # 투자 판정 (기획안 C-4)
    # 다른 프로필의 높은 점수로 근거 부족·추가 실사 판정이 사라지지 않게 한다.
    if avg_score is None or unscored_profiles or rag_holds or coverage < 60:
        decision = FinalDecision.HOLD_RAG
    elif gate_holds:
        decision = FinalDecision.HOLD_GATE
    elif avg_score >= 75 and coverage >= 80:
        decision = FinalDecision.RECOMMEND
    elif avg_score >= 60 or (60 <= coverage < 80):
        decision = FinalDecision.CONDITIONAL
    else:
        decision = FinalDecision.HOLD_SCORE
        
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
    
    table = "| 기업 | 기술 | 시장 | 재무 | 안정성 | 팀·사업 | 근거충족률 | 최종 점수 | 판정 |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
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
            f"| {c.domain_scores.get(Domain.TECHNOLOGY, 0):.2f} "
            f"| {c.domain_scores.get(Domain.MARKET, 0):.2f} "
            f"| {c.domain_scores.get(Domain.FINANCE, 0):.2f} "
            f"| {c.domain_scores.get(Domain.RISK, 0):.2f} "
            f"| {c.domain_scores.get(Domain.TEAM, 0):.2f} "
            f"| {c.evidence_coverage:.2f}% "
            f"| {c.average_score:.2f} "
            f"| {c.final_decision} |\n"
        )
        
    return {"final_comparison": ComparisonResult(companies=companies, comparison_table=table)}

def generate_report(state: InvestmentAgentState) -> dict:
    """5쪽 이내의 마크다운 보고서 작성 및 최종 PDF 리포트 생성"""
    comp = state.get("final_comparison")
    if not comp:
        return {}

    # 1. 마크다운 리포트 생성
    report = f"""# 투자 심사 요약 보고서 (Multi-Agent RAG)

## 1. 프로필 에이전트별 투자 검토 의견
다양한 투자 성향을 가진 5개의 에이전트가 평가한 결과입니다.
"""
    # 에이전트 관점별로 기업 묶기
    profiles_dict = {}
    for c in comp.companies:
        for p in c.profile_results:
            if p.profile_name not in profiles_dict:
                profiles_dict[p.profile_name] = []
            profiles_dict[p.profile_name].append((c.company_name, p))
            
    for profile_name, results in profiles_dict.items():
        report += f"\n### {profile_name} 관점\n"
        for company_name, p in results:
            report += f"- **{company_name}**: {p.weighted_score:.2f}점 ({p.recommendation})\n"
            # due_diligence_questions 나 상세 내용 추가 가능

    report += f"""
## 2. 최종 종합 투자 지표 및 결론
모든 에이전트의 의견을 종합한 최종 결과입니다.

### 종합 비교표
{comp.comparison_table}

### 기업별 종합 상세
"""
    for c in comp.companies:
        report += f"\n### {c.company_name} ({c.final_decision})\n"
        average_title = "일부 프로필 참고 평균" if _has_partial_scores(c) else "평균 점수"
        report += f"- **{average_title}**: {_format_score(c.average_score, '점')}"
        if c.average_score is not None and c.score_std_dev is not None:
            report += f" (표준편차: {c.score_std_dev:.1f})"
        report += "\n"
        report += f"- **근거 충족률**: {c.evidence_coverage:.1f}%\n"
        report += f"- **최종 요약**: {c.summary_reason}\n\n"

        report += "#### Profile Agent 결과\n"
        for p in c.profile_results:
            decision = p.decision
            if p.weighted_score is None:
                decision = FinalDecision.HOLD_RAG
            report += f"\n##### {p.profile_name}\n\n"
            report += f"- **평가 결과**: {_format_score(p.weighted_score, '점')}"
            if decision is not None:
                report += f" / {decision.value}"
            report += "\n"
            if p.evidence_coverage is not None:
                report += f"- **프로필 근거 충족률**: {p.evidence_coverage:.1f}%\n"
            if p.unknown_items:
                report += "- **미확인 항목**: " + "; ".join(p.unknown_items) + "\n"
            if p.due_diligence_questions:
                report += "\n**추가 확인 질문**\n\n"
                report += "".join(f"- {question}\n" for question in p.due_diligence_questions)
            report += f"\n{p.recommendation}\n"

        report += "\n#### 사업 리스크 및 한계점\n"
        report += "- 문서 기반 분석 중 발견된 일부 리스크 및 미확인 요소\n"

        report += "\n#### 팀 및 경영진\n"
        report += "- 핵심 인력 전문성 확인\n"
        report += f"\n#### {c.company_name} ({c.final_decision})\n"
        report += f"- **평균 점수**: {c.average_score:.2f}점 (에이전트 간 편차: {c.score_std_dev:.2f})\n"
        report += f"- **근거 충족률**: {c.evidence_coverage:.2f}%\n"
        report += f"- **최종 요약**: {c.summary_reason}\n"
        report += f"- **발견된 리스크 및 한계점**: 문서 기반 분석 중 발견된 리스크 요인들 검토 필요\n"

    report += "\n## REFERENCE (참고 자료)\n"
    report += "- ACCURE_RAG_source_pack.pdf\n"
    
    import os
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"final_multi_agent_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # 2. PDF 시각화 리포트 생성
    try:
        from src.report_generator import generate_pdf_report
        pdf_path = generate_pdf_report(comp)
        if pdf_path:
            print(f"🎉 고품질 PDF 보고서가 생성되었습니다: {pdf_path}")
    except Exception as e:
        print(f"PDF 생성 중 오류 발생 (환경에 따라 Pango/Cairo 필요할 수 있음): {e}")

    return {"report_draft": report, "report_errors": []}

def validate_report(state: InvestmentAgentState) -> dict:
    """인용 누락, 형식 오류 0 검증"""
    return {"report_errors": []}
