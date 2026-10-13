"""10쪽 이내 중립 평가 요약본. 전체 주장·원문은 Markdown/근거 bundle에 보존."""

import os
from pathlib import Path
from xml.sax.saxutils import escape

from src.orchestration.schemas import PERSPECTIVE_LABELS, Perspective


def _font_name():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        os.environ.get("REPORT_FONT_PATH", ""),
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont("ReportKorean", candidate))
            return "ReportKorean"
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    return "HYSMyeongJo-Medium"


def generate_pdf(state, output_path: str) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from pypdf import PdfReader

    font = _font_name()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = {}

    def shortened(value: str, limit: int) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[:limit - 3] + "..."

    for size in [9.5, 9.0, 8.5]:
        base = ParagraphStyle("body", fontName=font, fontSize=size,
                              leading=size * 1.5, spaceAfter=5, wordWrap="CJK")
        title = ParagraphStyle("title", parent=base, fontSize=17, leading=23,
                               spaceAfter=15, textColor=colors.HexColor("#17324d"))
        heading = ParagraphStyle("heading", parent=base, fontSize=12, leading=17,
                                 spaceBefore=9, spaceAfter=6, textColor=colors.HexColor("#17324d"))
        small = ParagraphStyle("small", parent=base, fontSize=size - 1, leading=size + 3,
                               textColor=colors.HexColor("#485663"))
        story = []
        sources.clear()

        def paragraph(text: str, style=base):
            story.append(Paragraph(escape(text), style))

        paragraph("배터리 AI 기업·기술 다관점 평가", title)
        paragraph("SUMMARY", heading)
        if state["execution_mode"] == "demo":
            paragraph("허구 데모: 회사·기관·문서·수치는 교육용 fixture입니다. 실제 검색 결과가 아닙니다.")
        paragraph("기술 성숙도·시장성·이해관계자·도메인 적용을 제공된 자료 범위에서 검토합니다.")
        verdict = state["eval_result"]
        paragraph("보고서 품질: " + ("통과" if verdict.passed else "미달 - 추가 확인 필요"))
        for name, label in [("groundedness", "Groundedness"), ("neutrality", "중립성"),
                            ("bias_control", "편향 통제"), ("coverage", "관점 커버리지")]:
            check = getattr(verdict, name)
            paragraph(f"{label}: {'충족' if check.passed else '미달'} / {shortened(check.reason, 230)}")
        paragraph(f"평가 모드: {state['execution_mode']} / 품질 평가: {verdict.evaluator}")
        paragraph(f"작업 수: {len(state['task_registry'])} / 품질 재작업: {state['quality_retry_count']}회")
        paragraph(f"Trace: {state['trace_id']}", small)
        paragraph("요약본의 문장은 길이를 제한했습니다. 전체 주장, 인용, 미확인 항목은 동일 실행의 report.md와 근거 JSON에서 확인하세요.")

        for company in state["selected_companies"]:
            story.append(PageBreak())
            paragraph(company.name, title)
            results = [result for result in state["worker_results"].values()
                       if result.company_name == company.name]
            for perspective in Perspective:
                paragraph(PERSPECTIVE_LABELS[perspective], heading)
                workers = [result for result in results if result.perspective == perspective]
                claims = [claim for result in workers for claim in result.claims]
                selected = []
                for kinds in [{"support", "context"}, {"limitation"}]:
                    candidate = next((claim for claim in claims if claim.kind in kinds), None)
                    if candidate:
                        selected.append(candidate)
                if not selected:
                    paragraph("확인 가능한 인용 근거가 부족합니다.")
                for claim in selected:
                    refs = " ".join(f"[{citation.evidence_id}]" for citation in claim.citations)
                    paragraph(shortened(claim.text, 170))
                    paragraph(refs, small)
                    for citation in claim.citations:
                        for worker in workers:
                            source = next((source for source in worker.sources
                                           if source.evidence_id == citation.evidence_id), None)
                            if source:
                                sources[source.evidence_id] = (source, citation.quote)
                unknowns = [value for result in workers for value in result.unknowns]
                if unknowns:
                    paragraph("미확인: " + shortened(unknowns[0], 95), small)
                failed = [result for result in workers if result.status == "failed"]
                if failed:
                    paragraph("Worker 실패: " + ", ".join(f"{r.task_id} ({r.error_type})" for r in failed), small)

        story.append(PageBreak())
        paragraph("REFERENCE", title)
        paragraph("아래 출처는 이 요약본에 인용된 입력 자료입니다. 전체 자료 목록은 report.md에서 확인할 수 있습니다.", small)
        for evidence_id, (source, quote) in sources.items():
            paragraph(f"[{evidence_id}] {source.source_file} / p.{source.page}", small)
            paragraph("인용: " + shortened(quote, 120), small)
            if source.url:
                paragraph(shortened(source.url, 150), small)
            story.append(Spacer(1, 3))
        if not sources:
            paragraph("유효한 인용 출처를 확보하지 못했습니다.")

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont(font, 8)
            canvas.setFillColor(colors.HexColor("#657381"))
            canvas.drawString(40, 24, "Orchestrator-Workers | " + state["execution_mode"])
            canvas.drawRightString(A4[0] - 40, 24, str(doc.page))
            canvas.restoreState()

        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=40, rightMargin=40,
                                topMargin=38, bottomMargin=38)
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        if len(PdfReader(path).pages) <= 10:
            return str(path)
    path.unlink(missing_ok=True)
    raise ValueError("요약 PDF가 10쪽 상한을 초과했습니다. 입력 기업/내용을 줄여 주세요.")
