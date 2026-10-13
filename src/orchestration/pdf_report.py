"""근거와 품질 판정을 보존하는 10쪽 이내 다관점 평가 요약본.

선택한 주장과 직접 인용은 잘라 쓰지 않으며 전체 내용은 report.md에 남는다.
"""

import json
import os
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from src.orchestration.schemas import (
    PERSPECTIVE_LABELS, Perspective, PerspectiveSummary,
    QualityVerdict, ResearchTask, WorkerResult,
)

_NAVY = "#182C48"
_BLUE = "#2A79B8"
_INK = "#263C55"
_MUTED = "#667B92"
_LIGHT = "#EFF3F8"
_LINE = "#CED8E6"
_STATUS = {"completed": "자료 확인", "partial": "일부 근거 부족",
           "insufficient_evidence": "근거 부족", "failed": "작업 실패"}
_TYPE_LABELS = {"original_pdf_page": "입력 PDF 원문 페이지",
                "supplied_evidence_summary": "사용자 제공 근거 요약",
                "source_card": "입력 자료의 출처 카드",
                "fictional_demo": "허구 데모 자료", "document": "입력 문서"}
_CHECKS = [("groundedness", "Groundedness"), ("neutrality", "중립성"),
           ("bias_control", "편향 통제"), ("coverage", "관점 커버리지")]


def _font_name():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    if "ReportKorean" in pdfmetrics.getRegisteredFontNames():
        return "ReportKorean"
    candidates = [os.environ.get("REPORT_FONT_PATH", ""),
                  "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "C:/Windows/Fonts/malgun.ttf"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont("ReportKorean", candidate))
            pdfmetrics.registerFontFamily("ReportKorean", normal="ReportKorean", bold="ReportKorean")
            return "ReportKorean"
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    pdfmetrics.registerFontFamily("HYSMyeongJo-Medium", normal="HYSMyeongJo-Medium", bold="HYSMyeongJo-Medium")
    return "HYSMyeongJo-Medium"


def _text(value):
    """Collapse display whitespace without truncating claims, quotes, or URLs."""
    return " ".join(str(value or "").split())


def _field(value, name, default=""):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _evaluation_reason(value):
    """Show identical static/Judge reasons once without dropping their meaning."""
    static, separator, judge = value.partition(" / LLM Judge: ")
    if separator and static.startswith("정적 검사: ") and static.removeprefix("정적 검사: ") == judge:
        return "정적 검사 및 LLM Judge: " + judge
    return value


def _history(state):
    root = Path(state.get("output_dir", "."))
    rows = []
    for revision in range(state.get("plan_revision", 0) + 1):
        verdict = _read_json(root / f"quality-r{revision}.json")
        plan = _read_json(root / f"plan-r{revision}.json")
        if not verdict and not plan:
            continue
        failed = [label for key, label in _CHECKS if verdict and not verdict.get(key, {}).get("passed", False)]
        targets = verdict.get("retry_targets", []) if verdict else []
        categories = _unique(PERSPECTIVE_LABELS[Perspective(target["perspective"])]
                             for target in targets if target.get("perspective") in {item.value for item in Perspective})
        rows.append({"revision": revision, "tasks": len(plan) if isinstance(plan, list) else None,
                     "passed": verdict.get("passed") if verdict else None,
                     "failed": failed, "targets": len(targets), "categories": categories})
    return rows


def _selected_claims(workers):
    """Keep representative context and limitations for every split-domain task."""
    selected, seen = [], set()
    for kinds in ({"support", "context"}, {"limitation"}):
        for worker in workers:
            candidates = [item for item in worker.claims if item.kind in kinds]
            # Supplied financial-risk evidence must not disappear behind a benign
            # market statement in a neutral summary. Preserve its actual wording.
            def priority(item):
                ids = {citation.evidence_id for citation in item.citations}
                financial = any(source.evidence_id in ids and source.source_id.replace("-", "").upper() == "E09"
                                for source in worker.sources)
                return 0 if financial and item.kind == "limitation" else 1
            candidates.sort(key=priority)
            claim = candidates[0] if candidates else None
            if claim:
                key = (claim.text, tuple((item.evidence_id, item.quote) for item in claim.citations))
                if key not in seen:
                    seen.add(key)
                    selected.append((worker, claim))
    # A separate domain-context sentence can distinguish BESS collaboration
    # from EV deployment without replacing either the finding or its limitation.
    if workers and workers[0].perspective == Perspective.APPLICATION:
        for worker in workers:
            context = next((item for item in worker.claims if item.kind == "context"), None)
            if not context:
                continue
            key = (context.text, tuple((item.evidence_id, item.quote) for item in context.citations))
            if key not in seen:
                position = sum(claim.kind != "limitation" for _, claim in selected)
                selected.insert(position, (worker, context))
                break
    return selected


def generate_pdf(state, output_path: str) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from pypdf import PdfReader

    font = _font_name()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 88
    companies = state.get("selected_companies", [])
    names = [_field(company, "name") for company in companies]
    tasks = {key: ResearchTask.model_validate(value) for key, value in state.get("task_registry", {}).items()}
    results = {key: WorkerResult.model_validate(value) for key, value in state.get("worker_results", {}).items()}
    verdict = QualityVerdict.model_validate(state["eval_result"])
    summaries = [PerspectiveSummary.model_validate(value) for value in state.get("synthesis", [])]
    summary_map = {(item.company_name, item.perspective): item.status for item in summaries}
    history = _history(state)
    editorial = state.get("editorial_review") or {}
    editorial_quality = _read_json(Path(state.get("output_dir", ".")) / "quality-editorial.json") if editorial else None
    status_counts = Counter(result.status for result in results.values())
    sections, references = {}, {}
    all_sources = {source.evidence_id: source for result in results.values() for source in result.sources}
    for name in names:
        for perspective in Perspective:
            workers = [result for result in results.values()
                       if result.company_name == name and result.perspective == perspective]
            selected = _selected_claims(workers)
            sections[(name, perspective)] = (workers, selected)
            for worker, claim in selected:
                for citation in claim.citations:
                    source = next((item for item in worker.sources if item.evidence_id == citation.evidence_id), None)
                    source = source or all_sources.get(citation.evidence_id)
                    entry = references.setdefault(citation.evidence_id, {
                        "alias": f"R{len(references) + 1:02d}", "source": source, "quotes": []})
                    if citation.quote not in entry["quotes"]:
                        entry["quotes"].append(citation.quote)

    def status(name, perspective):
        if (name, perspective) in summary_map:
            return summary_map[(name, perspective)]
        matching_tasks = [task for task in tasks.values() if task.company_name == name and task.perspective == perspective]
        matching = sections[(name, perspective)][0]
        completed = [result for result in matching if result.status == "completed" and result.claims]
        if completed and len(completed) == len(matching_tasks):
            return "completed"
        if completed:
            return "partial"
        if matching and all(result.status == "failed" for result in matching):
            return "failed"
        return "insufficient_evidence"

    class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved_pages = []

        def showPage(self):
            self.saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self.saved_pages)
            for page_state in self.saved_pages:
                self.__dict__.update(page_state)
                self.saveState()
                self.setFont(font, 7)
                self.setFillColor(colors.HexColor(_MUTED))
                self.drawString(44, 27, "SKALA 4기 · 울산 3반 · 4조")
                self.drawRightString(A4[0] - 44, 27, f"{self._pageNumber} / {total}")
                self.restoreState()
                Canvas.showPage(self)
            Canvas.save(self)

    for size in (9.3, 8.9, 8.5):
        body = ParagraphStyle("body", fontName=font, fontSize=size, leading=size * 1.46,
                              textColor=colors.HexColor(_INK), spaceAfter=5, wordWrap="CJK")
        title = ParagraphStyle("title", parent=body, fontSize=18, leading=25,
                               textColor=colors.HexColor(_NAVY), spaceAfter=14)
        heading = ParagraphStyle("heading", parent=body, fontSize=11.5, leading=17,
                                 textColor=colors.HexColor(_NAVY), spaceBefore=10, spaceAfter=7, keepWithNext=True)
        small = ParagraphStyle("small", parent=body, fontSize=size - 1, leading=size * 1.28,
                               textColor=colors.HexColor(_MUTED), spaceAfter=4)
        tiny = ParagraphStyle("tiny", parent=small, fontSize=size - 1.3, leading=size * 1.24)
        cell = ParagraphStyle("cell", parent=small, textColor=colors.HexColor(_INK), spaceAfter=0)
        white = ParagraphStyle("white", parent=cell, textColor=colors.white)
        card_label = ParagraphStyle("card-label", parent=tiny, spaceAfter=7)
        card_value = ParagraphStyle("card-value", parent=body, fontSize=18, leading=22, spaceAfter=0)
        ref_style = ParagraphStyle("ref", parent=body, fontSize=size - 0.9, leading=size * 1.25, spaceAfter=3)
        story = []

        def p(value, style=body):
            return Paragraph(escape(_text(value)), style)

        def para(value, style=body):
            story.append(p(value, style))

        def table(rows, widths, header=True, compact=False):
            rendered = [[p(value, white if header and row_index == 0 else cell)
                         for value in row] for row_index, row in enumerate(rows)]
            result = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
            padding = 5 if compact else 7
            style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                     ("TOPPADDING", (0, 0), (-1, -1), padding), ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                     ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.HexColor(_LINE)),
                     ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor(_LINE))]
            if header:
                style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
                          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(_LIGHT)])]
            result.setStyle(TableStyle(style))
            return result

        def cards(items):
            card_width = (width - 24) / len(items)
            row = []
            for index, (label, value) in enumerate(items):
                content = Table([[p(label, card_label)], [p(value, card_value)]], colWidths=[card_width])
                content.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_LIGHT)),
                    ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor([_BLUE, "#38977D", "#7852AE", "#DC8548"][index % 4])),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
                row.append(content)
                if index < len(items) - 1:
                    row.append("")
            widths = [card_width if i % 2 == 0 else 24 / (len(items) - 1) for i in range(len(row))]
            result = Table([row], colWidths=widths, hAlign="LEFT")
            result.setStyle(TableStyle([(name, (0, 0), (-1, -1), 0) for name in
                                       ["LEFTPADDING", "RIGHTPADDING", "TOPPADDING", "BOTTOMPADDING"]]))
            story.extend([result, Spacer(1, 16)])

        def citation_text(claim):
            return " ".join(f"[{references[evidence_id]['alias']} · {evidence_id}]"
                            for evidence_id in _unique(item.evidence_id for item in claim.citations))

        def page_header(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.HexColor(_NAVY))
            canvas.setFont(font, 10)
            canvas.drawString(44, A4[1] - 44, "배터리 AI 스타트업 다관점 기술 평가")
            canvas.setFont(font, 6.5)
            canvas.setFillColor(colors.HexColor(_MUTED))
            canvas.drawRightString(A4[0] - 44, A4[1] - 44, "기준일 2026-10-13")
            canvas.setStrokeColor(colors.HexColor(_NAVY))
            canvas.setLineWidth(0.9)
            canvas.line(44, A4[1] - 54, A4[0] - 44, A4[1] - 54)
            canvas.restoreState()

        # Overview has status counts, never investment scores or rankings.
        para("SUMMARY", title)
        para("기술 성숙도·시장성·이해관계자·도메인 적용을 공개 입력 자료 범위에서 살펴봅니다. 확인 내용과 한계·상충 근거를 함께 기록합니다.")
        if state.get("execution_mode") == "demo":
            para("허구 데모: 회사·기관·문서·수치는 교육용 fixture이며 실제 기업 조사 결과가 아닙니다.")
        story.append(Spacer(1, 7))
        cards([("평가 기업", f"{len(names)}개"), ("평가 관점", "4개"),
               ("동적 조사 작업", f"{len(tasks)}개"),
               ("자동 초안 재작업" if editorial else "품질 재작업", f"{state.get('quality_retry_count', 0)}회")])
        quality_text = "품질 기준 충족" if verdict.passed else "품질 미달 · 추가 확인 필요"
        quality_message = ("평가 항목을 충족했습니다. 이는 입력 문서 밖의 사실이나 기술 우열을 확정하지 않습니다."
                           if verdict.passed else "아래 미달 항목을 남긴 상태로 종료했습니다. 입력 근거의 한계와 재작업 결과는 품질 평가 쪽에서 확인할 수 있습니다.")
        if editorial:
            quality_message = "자동 초안 후 원문 대조 편집을 거쳤습니다. 아래 최종 품질 판정은 편집본 기준이며, 자동 초안의 Loop 이력은 별도로 보존했습니다."
        callout = Table([[p("최종 보고서 품질", white)],
                         [p(quality_text, ParagraphStyle("callout", parent=white, fontSize=14, leading=20))],
                         [p(quality_message, white)]], colWidths=[width])
        callout.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_NAVY)),
            ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
        story.append(callout)
        para("기업별 조사 상태", heading)
        rows = [["기업", "기술 성숙도", "시장성", "이해관계자", "도메인 적용", "작업"]]
        for name in names:
            rows.append([name] + [_STATUS[status(name, view)] for view in Perspective]
                        + [str(sum(task.company_name == name for task in tasks.values()))])
        story.append(table(rows, [120, 81, 70, 83, 83, width - 437]))
        para("자료 해석의 기준", heading)
        para("‘자료 확인’은 해당 관점의 인용 가능한 Worker 결과를 확보했다는 뜻입니다. 장기 운영 성능·시장 성과·독립 검증의 완료 여부는 개별 주장과 한계에서 확인해야 합니다.")
        para("품질 항목", heading)
        quality_rows = [["항목", "판정", "평가 목적"]]
        for key, label, purpose in [
            ("groundedness", "Groundedness", "주장과 검색 출처의 연결"),
            ("neutrality", "중립성", "기술 추천·우열 판정 통제"),
            ("bias_control", "편향 통제", "단일 출처·유리한 근거 편중 통제"),
            ("coverage", "관점 커버리지", "네 관점의 근거와 한계 포함")]:
            quality_rows.append([label, "충족" if getattr(verdict, key).passed else "미달", purpose])
        story.append(table(quality_rows, [120, 60, width - 180]))
        story.append(Spacer(1, 9))
        para("PDF는 대표 주장 요약본입니다. 전체 주장·미확인 사항·원문 인용은 동일 실행 폴더의 report.md와 evidence/ JSON에 보존했습니다.", small)

        story.append(PageBreak())
        para("조사 범위와 읽는 방법", title)
        para("1. 분석 대상", heading)
        scope_rows = [["기업", "입력 기술·적용 범위", "조사 작업"]]
        for company in companies:
            name = _field(company, "name")
            scope_rows.append([name, _field(company, "description"), str(sum(task.company_name == name for task in tasks.values()))])
        story.append(table(scope_rows, [120, width - 180, 60]))
        para("2. Orchestrator-Workers 실행", heading)
        para("Orchestrator > 동적 Worker 병렬 조사 > Synthesizer > 보고서 생성 > 품질 평가 > 미달 관점 재계획")
        para("Orchestrator가 기업별 쟁점과 적용 도메인에 맞춰 작업을 나눕니다. Worker는 같은 네 관점을 기준으로 각 질문의 근거·한계·미확인 사항을 조사합니다. Synthesizer는 task_id별 최신 결과를 집계합니다.")
        if editorial:
            para("작성 절차: 자동 초안 후 원문 대조 편집을 수행했습니다. 최종 품질 평가는 편집본 기준이며, 초안의 잘못된 일반화·출처 누락·인용 범위를 점검한 뒤 다시 평가했습니다.", small)
        task_rows = [["관점", "핵심 질문", "작업 수"]]
        for view, question in [
            (Perspective.TECHNOLOGY, "성능 검증 조건, 실증·성숙도, 데이터·일반화의 한계"),
            (Perspective.MARKET, "시장 수요, 고객·계약·수익모델, 공개 수치의 범위"),
            (Perspective.STAKEHOLDERS, "운영사·OEM·규제기관·사용자, 책임·데이터 권리"),
            (Perspective.APPLICATION, "ESS·EV 적용 조건, BMS 연동, 장기 운영 제약")]:
            task_rows.append([PERSPECTIVE_LABELS[view], question, str(sum(task.perspective == view for task in tasks.values()))])
        story.append(table(task_rows, [95, width - 155, 60]))
        para("3. 검색 및 출처의 경계", heading)
        retrieval = state.get("retrieval_metadata") or state.get("retrieval_method") or state.get("retrieval_mode")
        if retrieval:
            retrieval_text = " · ".join(f"{key}: {_text(value)}" for key, value in retrieval.items()) if isinstance(retrieval, dict) else _text(retrieval)
            if retrieval_text == "keyword":
                retrieval_text = "keyword · 로컬 입력 자료의 TF-IDF 키워드 검색"
            para("실행 검색 정보: " + retrieval_text, small)
        else:
            para("입력 문서에서 검색된 근거를 사용했습니다. 정확한 검색 방식은 실행 설정과 외부 로그를 확인하세요.", small)
        source_types = Counter(source.source_type for source in all_sources.values())
        if source_types:
            para("확보 근거 유형(고유 근거 ID): " + " · ".join(f"{_TYPE_LABELS.get(kind, kind)} {count}개" for kind, count in sorted(source_types.items())), small)
        para("입력 PDF의 요약·출처 카드는 원문 발표나 독립 실증 자료와 구분합니다. REFERENCE의 URL은 입력 자료에 기재된 출처 위치이며, 이 실행에서 원 웹페이지의 내용까지 검증했다는 뜻은 아닙니다.")
        para("논문과 IEA 등 산업 자료는 기술·시장 맥락이며, 해당 기업의 성능이나 고객 효과를 직접 검증하는 근거와 구분합니다.", small)
        para("4. 인용과 상태를 읽는 방법", heading)
        para("[R01 · 근거 ID]는 REFERENCE의 입력 파일·페이지·직접 인용으로 연결됩니다. ‘확인 내용’과 ‘적용 맥락’은 해당 자료의 진술이며, ‘한계·상충’과 ‘미확인’은 추가 검증이 필요한 범위입니다.")
        result_label = "편집본 관점별 결과" if editorial else "최신 Worker"
        para(f"{result_label}: 자료 확인 {status_counts['completed']}개 · 근거 부족 {status_counts['insufficient_evidence']}개 · 작업 실패 {status_counts['failed']}개. 빠진 결과 {max(0, len(tasks) - len(results))}개.", small)
        para("보고서 범위: " + state.get("target_domain", "배터리 AI") + " · 기술·기업 간 점수와 순위, 투자 권고는 산출하지 않습니다.", small)

        for company in companies:
            name = _field(company, "name")
            story.append(PageBreak())
            para(name, title)
            para(_field(company, "description"), small)
            para(f"조사 작업 {sum(task.company_name == name for task in tasks.values())}개 · 네 관점의 확인 내용과 한계", small)
            for index, perspective in enumerate(Perspective, 1):
                workers, selected = sections[(name, perspective)]
                para(f"{index}. {PERSPECTIVE_LABELS[perspective]}  |  {_STATUS[status(name, perspective)]}", heading)
                if not selected:
                    para("인용 가능한 주장이 확보되지 않았습니다. 추가 자료가 필요합니다.")
                for worker, claim in selected:
                    kind = {"support": "확인 내용", "context": "적용 맥락", "limitation": "한계·상충"}[claim.kind]
                    domain = ""
                    if len(workers) > 1:
                        task = tasks.get(worker.task_id)
                        question = task.question if task else ""
                        is_ev, is_ess = "EV" in question.upper(), "ESS" in question.upper()
                        domain = "EV · " if is_ev and not is_ess else "ESS · " if is_ess and not is_ev else f"분할 작업 {worker.task_id.rsplit('-', 1)[-1]} · "
                    rich = (f'<font color="{_BLUE if claim.kind != "limitation" else "#945D3E"}">'
                            f'{escape(domain + kind)}</font>  {escape(_text(claim.text))}')
                    story.append(KeepTogether([Paragraph(rich, body), p(citation_text(claim) or "출처 연결 없음", tiny)]))
                unknowns = _unique(value for worker in workers for value in worker.unknowns)
                if unknowns:
                    para("미확인: " + unknowns[0], small)
                incomplete = [worker for worker in workers if worker.status != "completed"]
                if incomplete:
                    para("작업 상태: " + " · ".join(f"{worker.task_id}: {_STATUS[worker.status]}" +
                         (f" ({worker.error_type})" if worker.error_type else "") for worker in incomplete), small)
            company_limits = _unique(value for result in results.values() if result.company_name == name for value in result.source_limitations)
            if company_limits:
                story.append(Spacer(1, 5))
                para("출처 한계: " + company_limits[0], tiny)

        story.append(PageBreak())
        para("보고서 품질 평가와 재작업", title)
        para(("편집본 최종 판정: " if editorial else "최종 판정: ") + quality_text, heading)
        para(f"평가 방식: {verdict.evaluator} · 설정: {state.get('quality_mode', '미기록')} · "
             f"{'자동 초안 재작업' if editorial else '품질 재작업'} {state.get('quality_retry_count', 0)}회 / 상한 {state.get('max_quality_retries', '미기록')}회", small)
        if editorial:
            para("아래 네 항목은 원문 대조 편집 후 보고서의 평가 결과입니다. 자동 초안의 회차별 판정과 구분합니다.", small)
        if state.get("quality_mode") == "hybrid" or "hybrid" in verdict.evaluator.lower():
            para("Hybrid 평가: 인용 연결·형식·네 관점 포함 여부를 결정적으로 검사하고, LLM Judge가 주장의 내용·중립성·출처 편중을 평가합니다. Judge의 내용 평가는 비결정적일 수 있습니다.")
        else:
            para("결정적 검사 결과입니다. 인용과 형식의 충족 여부만으로 내용의 정확성이나 독립 검증을 확정할 수는 없습니다.")
        quality_rows = [["항목", "판정", "실제 평가 사유"]]
        for key, label in _CHECKS:
            check = getattr(verdict, key)
            quality_rows.append([label, "충족" if check.passed else "미달", _evaluation_reason(check.reason)])
        story.append(table(quality_rows, [105, 48, width - 153], compact=True))
        para("정적 검사와 Judge가 같은 사유를 반환한 경우 함께 표기했습니다. 평가 사유의 원문은 quality-final.json에 보존했습니다.", tiny)
        para("커버리지 판정은 각 기업의 네 관점에 유효 인용 주장이 존재하는지 확인한 결과입니다. 개별 관점의 모든 쟁점을 조사했거나 내용상 함의를 완전히 검증했다는 뜻은 아닙니다.", tiny)
        para("자동 초안의 Loop 기록" if editorial else "Loop 기록", heading)
        if history:
            loop_rows = [["회차", "작업", "판정", "미달 항목 / 재작업 대상 관점"]]
            for row in history:
                detail = ", ".join(row["failed"]) or "미달 항목 없음"
                if row["targets"]:
                    detail += f" / 대상 {row['targets']}개: " + ", ".join(row["categories"])
                loop_rows.append([f"r{row['revision']}", str(row["tasks"]) if row["tasks"] is not None else "미기록",
                                  "충족" if row["passed"] else "미달" if row["passed"] is False else "미기록", detail])
            story.append(table(loop_rows, [40, 42, 45, width - 127], compact=True))
            para("각 행은 " + ("원래 자동 실행에서 보존한 " if editorial else "해당 실행의 ") +
                 "plan-rN.json과 quality-rN.json에서 읽었습니다. r0는 최초 계획, 이후 회차는 품질 평가의 미달 대상을 반영한 재계획입니다.", small)
        else:
            para("회차별 외부 로그가 없어 재작업 이력을 검증하지 못했습니다. 최종 State에 기록된 재작업 횟수만 표시합니다.", small)
        termination = state.get("termination_reason") or (
            "quality_passed" if verdict.passed else "step_limit_reached"
            if state.get("step_count", 0) + 5 >= state.get("max_steps", 30) else "quality_retry_limit_reached")
        if editorial:
            termination = _field(editorial, "original_termination_reason", termination)
        para(("자동 실행 종료 사유: " if editorial else "종료 사유: ") + termination, small)
        if editorial:
            if editorial_quality and isinstance(editorial_quality.get("passed"), bool):
                editorial_status = "충족" if editorial_quality["passed"] else "미달"
                para(f"편집 후 최종 검사: {editorial_status} · 평가 {editorial_quality.get('evaluator', '미기록')} · quality-editorial.json", small)
            else:
                para("편집 후 최종 검사는 위 최종 State의 판정을 표시했습니다. 별도 quality-editorial.json은 확인되지 않았습니다.", small)
            if _field(editorial, "original_run_id"):
                para("자동 초안 Run ID: " + _field(editorial, "original_run_id"), tiny)
            if _field(editorial, "note"):
                para("편집 이력: " + _field(editorial, "note"), tiny)
        if verdict.judge_error:
            para("Judge 오류: " + verdict.judge_error, small)
        para("원문 추적", heading)
        para("보고서 전체: report.md · 최신 평가: quality-final.json · 전체 결과: final-state.json · 검색된 원문: evidence/ · 의사결정: decisions.jsonl", small)
        para("Trace ID: " + state.get("trace_id", "미기록"), tiny)

        story.append(PageBreak())
        para("REFERENCE", title)
        para("이 PDF에 표시한 주장과 직접 연결된 입력 자료입니다. 원문 인용은 해당 자료의 진술을 추적하기 위한 것이며 독립 검증을 대신하지 않습니다.", small)
        para("출처 그룹은 작성 주체를 기준으로 묶었습니다. 같은 주체의 PDF·웹 발표는 독립 출처로 중복 계산하지 않으며, 파일 단위로만 묶인 자료는 작성 주체의 독립성이 불확실합니다.", tiny)
        para("보고서 생성에 사용한 입력 파일", heading)
        for source_file in _unique(source.source_file for source in all_sources.values()):
            para("• " + source_file, tiny)
        para("표시 주장별 근거와 직접 인용", heading)
        for evidence_id, entry in references.items():
            source, blocks = entry["source"], []
            if source:
                blocks.append(p(f"[{entry['alias']} · {evidence_id}] {source.source_file} · {source.page}쪽", ref_style))
                type_label = _TYPE_LABELS.get(source.source_type, source.source_type)
                proxy = source.source_group == source.source_file or source.source_group.startswith(("file:", "document:", "pdf:"))
                group_note = " · 파일 단위, 독립성 불확실" if proxy else ""
                blocks.append(p(f"자료 유형: {type_label} · 출처 ID: {source.source_id} · 그룹: {source.source_group}", tiny))
                if source.source_id.replace("-", "").upper() == "E09":
                    blocks.append(p("검증 범위: 제공 카드의 2025 회사 공시 재서술. 공시 원문 전체와 현재 재무상태는 미확인입니다.", tiny))
                if group_note:
                    blocks.append(p("출처 해석" + group_note, tiny))
                blocks.append(p("입력 자료의 원출처 URL: " + source.url if source.url else "원출처 URL: 입력 자료에 없음", tiny))
            else:
                blocks.append(p(f"[{entry['alias']} · {evidence_id}] 원문 위치 연결 실패", ref_style))
            for quote in entry["quotes"]:
                blocks.append(p("직접 인용: “" + quote + "”", ref_style))
            blocks.append(Spacer(1, 7))
            story.append(KeepTogether(blocks))
        if not references:
            para("연결 가능한 인용 출처를 확보하지 못했습니다.")
        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=44, rightMargin=44,
                                topMargin=78, bottomMargin=48,
                                title="배터리 AI 스타트업 다관점 기술 평가 보고서", author="SKALA 4기 울산 3반 4조")
        doc.build(story, onFirstPage=page_header, onLaterPages=page_header, canvasmaker=NumberedCanvas)
        reader = PdfReader(path)
        thin_overflow = any(len(page.extract_text() or "") < 200 for page in reader.pages)
        # A slightly smaller text profile is preferable to a separate page
        # containing only a trace identifier or an isolated metadata line.
        if 1 <= len(reader.pages) <= 10 and (not thin_overflow or size == 8.5):
            return str(path)
    path.unlink(missing_ok=True)
    raise ValueError("요약 PDF가 10쪽 상한을 초과했습니다. 전체 주장과 인용은 report.md에서 확인하세요.")
