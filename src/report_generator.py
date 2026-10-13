"""Ten-page evidence-aware PDF report, rendered with ReportLab."""

from datetime import date
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent.parent
W, H = A4
NAVY = colors.HexColor("#14243D")
BLUE = colors.HexColor("#2463B2")
PALE = colors.HexColor("#EAF1FB")
MUTED = colors.HexColor("#52627A")
RULE = colors.HexColor("#D9E2EF")


class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, preview_label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_label = preview_label

    def showPage(self):
        if self.preview_label:
            self.saveState()
            self.setFillColor(colors.HexColor("#D35353"))
            self.setFont("Helvetica-Bold", 9)
            self.drawRightString(W - 42, 52, self.preview_label)
            self.restoreState()
        super().showPage()


def _font():
    name = "ReportKorean"
    if name not in pdfmetrics.getRegisteredFontNames():
        candidates = [
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise RuntimeError("한글 PDF 글꼴이 없습니다. Noto Sans Gothic TTF를 설치하세요.")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    return name


def _text(pdf, font, text, x, y, width=W - 96, size=10, max_lines=5, lead=16):
    pdf.setFont(font, size)
    pdf.setFillColor(NAVY)
    lines = simpleSplit(str(text or "미확인"), font, size, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    for line in lines:
        pdf.drawString(x, y, line)
        y -= lead
    return y


def _page(pdf, font, number, title):
    pdf.setFillColor(NAVY)
    pdf.rect(0, H - 96, W, 96, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont(font, 9)
    pdf.drawString(42, H - 34, "BATTERY INTELLIGENCE / INVESTMENT REVIEW")
    pdf.setFont(font, 20)
    pdf.drawString(42, H - 69, title)
    pdf.setStrokeColor(RULE)
    pdf.line(42, 42, W - 42, 42)
    pdf.setFillColor(MUTED)
    pdf.setFont(font, 8)
    pdf.drawString(42, 27, "SKALA 4기 · 울산 3반 · 4조 | 근거 기반 투자 검토")
    pdf.drawRightString(W - 42, 27, f"{number:02d} / 10")


def _heading(pdf, font, label, y):
    pdf.setFillColor(BLUE)
    pdf.setFont(font, 12)
    pdf.drawString(42, y, label)
    pdf.setStrokeColor(RULE)
    pdf.line(42, y - 9, W - 42, y - 9)
    return y - 34


def _row(pdf, font, y, cells, widths, header=False):
    if header:
        pdf.setFillColor(PALE)
        pdf.rect(42, y - 30, sum(widths), 30, fill=1, stroke=0)
    x = 42
    for value, width in zip(cells, widths):
        _text(pdf, font, value, x + 7, y - 19, width - 14, 8, 1, 11)
        x += width
    pdf.setStrokeColor(RULE)
    pdf.line(42, y - 30, 42 + sum(widths), y - 30)
    return y - 30


def _score(company, domain):
    for key, value in company.domain_scores.items():
        if getattr(key, "value", key) == domain:
            return float(value)
    return 0.0


def _contributors():
    with (ROOT / "configs" / "contributors.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["contributors"]


def generate_pdf_report(comparison_result, output_path=None, preview_label=None):
    """Render three company dossiers in exactly ten A4 pages.

    The workflow's save_report node is responsible for enforcing quality checks.
    """
    companies = comparison_result.companies
    if len(companies) != 3:
        raise ValueError("10쪽 보고서에는 정확히 3개 기업 결과가 필요합니다.")
    contributors = _contributors()
    font = _font()
    destination = Path(output_path) if output_path else ROOT / "output" / "report.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = ReportCanvas(str(destination), pagesize=A4, preview_label=preview_label)
    pdf.setTitle("배터리 AI 스타트업 투자 검토 보고서")
    pdf.setAuthor("SKALA 4기 울산 3반 4조")

    # 1. Cover
    pdf.setFillColor(NAVY)
    pdf.rect(0, 0, W, H, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#72A8F2"))
    pdf.rect(42, H - 120, 90, 5, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont(font, 13)
    pdf.drawString(42, H - 155, "SKALA 4기 · 울산 3반 · 4조")
    pdf.setFont(font, 29)
    pdf.drawString(42, H - 232, "배터리 AI 스타트업")
    pdf.drawString(42, H - 276, "투자 검토 보고서")
    pdf.setFont(font, 10)
    pdf.drawString(42, 112, "대상: " + " / ".join(c.company_name for c in companies))
    pdf.drawString(42, 87, "작성일: " + date.today().isoformat())
    pdf.drawString(42, 62, "자동 투자 결정이 아닌 추가 실사 자료입니다.")
    pdf.showPage()

    # 2. Executive summary
    _page(pdf, font, 2, "핵심 요약")
    y = _heading(pdf, font, "기업별 판단", H - 126)
    for company in companies:
        y = _text(pdf, font, f"{company.company_name} | {company.average_score:.1f}점 | {company.final_decision}",
                  48, y, size=11, max_lines=2) - 8
        y = _text(pdf, font, f"근거 충족률 {company.evidence_coverage:.0f}% · 프로필 편차 {company.score_std_dev:.1f}점",
                  48, y, size=9, max_lines=2) - 28
    y = _heading(pdf, font, "읽는 방법", y - 10)
    _text(pdf, font, "점수는 서로 다른 투자 관점을 비교하기 위한 내부 지표입니다. 기술 우열이나 투자 결정을 자동으로 확정하지 않습니다.",
          48, y, max_lines=4)
    pdf.showPage()

    # 3. Method and controls
    _page(pdf, font, 3, "평가 방법과 통제")
    y = _heading(pdf, font, "Orchestrator - Workers", H - 126)
    steps = [
        "1. 기업별 조사 작업 3~5개를 생성하고 계획을 검증합니다.",
        "2. 작업별 Research Worker를 병렬 실행하고 결과를 취합합니다.",
        "3. 근거가 부족하면 제한 횟수 내에서 질의를 재계획합니다.",
        "4. 공통 근거를 5개 투자 프로필이 독립적으로 평가합니다.",
        "5. 보고서의 출처 연결·중립성·출처 편중을 검증합니다.",
    ]
    for step in steps:
        y = _text(pdf, font, step, 48, y, max_lines=3) - 22
    y = _heading(pdf, font, "해석 한계", y - 4)
    for item in (
        "영역별 공통 점수에는 예시 값이 포함되어 있어 실투자 판단에 사용할 수 없습니다.",
        "공개 자료로 확인할 수 없는 계약·매출·성능은 미확인으로 남깁니다.",
        "품질 검증을 통과한 실행에서만 PDF를 생성합니다.",
    ):
        y = _text(pdf, font, "• " + item, 48, y, max_lines=3) - 14
    pdf.showPage()

    # 4. Score comparison
    _page(pdf, font, 4, "기업 간 비교")
    y = _heading(pdf, font, "5개 영역 점수", H - 126)
    widths = [138, 66, 66, 66, 66, 66, 43]
    y = _row(pdf, font, y, ["기업", "기술", "시장", "재무", "리스크", "팀", "평균"], widths, True)
    for company in companies:
        values = [f"{_score(company, key):.0f}" for key in
                  ("technology", "market", "finance", "risk", "team")]
        y = _row(pdf, font, y, [company.company_name, *values, f"{company.average_score:.1f}"], widths)
    y = _heading(pdf, font, "관점 차이", y - 44)
    for company in companies:
        y = _text(pdf, font, f"{company.company_name}: 5개 프로필 점수 표준편차 {company.score_std_dev:.1f}점",
                  48, y, max_lines=2) - 18
    pdf.showPage()

    # 5-7. Company dossiers
    for number, company in enumerate(companies, 5):
        _page(pdf, font, number, company.company_name)
        y = _heading(pdf, font, "판정 및 핵심 판단", H - 126)
        y = _text(pdf, font, f"{company.final_decision} | 평균 {company.average_score:.1f}점 | 근거 충족률 {company.evidence_coverage:.0f}%",
                  48, y, size=11, max_lines=2) - 16
        y = _text(pdf, font, company.summary_reason, 48, y, max_lines=6) - 22
        y = _heading(pdf, font, "영역별 평가", y)
        for key, label in (("technology", "기술"), ("market", "시장"),
                           ("finance", "재무"), ("risk", "리스크"), ("team", "팀")):
            score = max(0, min(100, _score(company, key)))
            pdf.setFillColor(NAVY)
            pdf.setFont(font, 9)
            pdf.drawString(48, y, label)
            pdf.setFillColor(PALE)
            pdf.roundRect(125, y - 2, 300, 10, 5, fill=1, stroke=0)
            pdf.setFillColor(BLUE)
            pdf.roundRect(125, y - 2, score * 3, 10, 5, fill=1, stroke=0)
            pdf.drawRightString(465, y, f"{score:.0f}")
            y -= 31
        y = _heading(pdf, font, "출처와 확인 과제", y - 7)
        y = _text(pdf, font, "독립 원본: " + (", ".join(company.source_documents) or "없음"),
                  48, y, size=9, max_lines=3) - 8
        y = _text(pdf, font, "근거 ID: " + (", ".join(company.evidence_sources[:12]) or "없음"),
                  48, y, size=8, max_lines=3) - 19
        y = _heading(pdf, font, "5개 투자 관점", y)
        for profile in company.profile_results[:5]:
            y = _text(pdf, font,
                      f"{profile.profile_name} {profile.weighted_score:.1f}점 - {profile.recommendation}",
                      48, y, size=9, max_lines=2, lead=14) - 10
        pdf.showPage()

    # 8. Ownership and perspectives
    _page(pdf, font, 8, "담당 역할과 투자 관점")
    y = _heading(pdf, font, "5명별 담당 계획", H - 126)
    for person in contributors:
        y = _text(pdf, font,
                  f"{person['name']} · {person['profile']} · {person['responsibility']} · 기존 PR #{person['prior_pr']}",
                  48, y, size=9, max_lines=2) - 18
    y = _heading(pdf, font, "기업별 프로필 점수", y - 6)
    widths = [162, 116, 116, 117]
    y = _row(pdf, font, y, ["투자 관점", *[c.company_name for c in companies]], widths, True)
    for person in contributors:
        scores = [next((f"{p.weighted_score:.1f}" for p in c.profile_results
                        if p.profile_name == person["profile"]), "-") for c in companies]
        y = _row(pdf, font, y, [person["profile"], *scores], widths)
    _text(pdf, font, "기존 PR은 dev 브랜치 이력입니다. 역할표는 현재 main의 GitHub 기여 기록을 대신하지 않습니다.",
          48, y - 26, size=9, max_lines=2)
    pdf.showPage()

    # 9. Evidence audit
    _page(pdf, font, 9, "근거 품질 점검")
    y = _heading(pdf, font, "기업별 출처 현황", H - 126)
    widths = [161, 105, 105, 140]
    y = _row(pdf, font, y, ["기업", "원본 문서", "근거 ID", "충족률"], widths, True)
    for company in companies:
        y = _row(pdf, font, y, [company.company_name,
                                str(len(set(company.source_documents))),
                                str(len(set(company.evidence_sources))),
                                f"{company.evidence_coverage:.0f}%"], widths)
    y = _heading(pdf, font, "검증 기준", y - 43)
    for item in (
        "각 기업의 독립 원본 문서 2개 이상 확보",
        "근거 ID에서 원본 파일·페이지·청크로 역추적 가능",
        "출처 부족이나 상충 시 미확인 항목과 추가 실사를 우선",
    ):
        y = _text(pdf, font, "• " + item, 48, y, max_lines=2) - 18
    pdf.showPage()

    # 10. Diligence and source IDs
    _page(pdf, font, 10, "추가 실사와 참고 자료")
    y = _heading(pdf, font, "실사 질문", H - 126)
    for question in (
        "모델 성능은 어떤 데이터와 시험 조건에서 재현되었는가?",
        "유료 고객 계약과 반복 매출은 독립 자료로 확인되는가?",
        "안전·규제 준수 및 데이터 권리는 계약서로 확인되는가?",
        "기업별 다른 원본 자료가 기존 주장과 충돌하지 않는가?",
    ):
        y = _text(pdf, font, "• " + question, 48, y, max_lines=3) - 15
    y = _heading(pdf, font, "REFERENCE - 사용된 근거 ID", y - 7)
    sources = sorted({source for c in companies for source in c.evidence_sources})
    if sources:
        if len(sources) > 96:
            raise ValueError("참고 근거 ID가 10쪽 PDF의 참조 영역을 초과했습니다.")
        for index, source in enumerate(sources):
            label = "[" + source + "]"
            if pdfmetrics.stringWidth(label, font, 7) > 155:
                raise ValueError(f"근거 ID가 참조 열 너비를 초과했습니다: {source}")
            column, row = divmod(index, 32)
            pdf.setFillColor(NAVY)
            pdf.setFont(font, 7)
            pdf.drawString(48 + column * 169, y - row * 13, label)
    else:
        _text(pdf, font, "검증 가능한 근거 ID 없음", 48, y, size=9)
    pdf.showPage()

    pdf.save()
    return str(destination)
