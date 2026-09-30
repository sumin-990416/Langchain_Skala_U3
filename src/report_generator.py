import datetime
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

# GUI가 없는 터미널·CI에서도 차트를 생성할 수 있는 백엔드를 강제합니다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apple Silicon의 Homebrew 라이브러리를 WeasyPrint가 찾을 수 있게 합니다.
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    "/opt/homebrew/lib:/usr/local/lib:"
    + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
)

from weasyprint import HTML

from src.schemas import Domain


DOMAIN_LABELS = {
    Domain.TECHNOLOGY: "기술",
    Domain.MARKET: "시장",
    Domain.FINANCE: "재무·고객",
    Domain.RISK: "안정성",
    Domain.TEAM: "팀·사업",
}

COMPANY_DESCRIPTIONS = {
    "ACCURE": "BESS·EV 배터리의 이상·안전·성능·열화를 분석하는 Battery Intelligence 기업",
    "volytica diagnostics": "운영 데이터 기반 SOH·RUL·열화·안전 진단 소프트웨어 기업",
    "Electra Vehicles": "클라우드·임베디드 AI 배터리 디지털 트윈 및 제어 소프트웨어 기업",
}


def _score_text(score: Optional[float]) -> str:
    return "점수 없음" if score is None else f"{score:.1f}"


def _coverage_text(coverage: Optional[float]) -> str:
    return "미산출" if coverage is None else f"{coverage:.1f}%"


def _truncate(value: str, limit: int = 115) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _domain_score(company: Any, domain: Domain) -> Optional[float]:
    return company.domain_scores.get(domain, company.domain_scores.get(domain.value))


def _company_context(company: Any) -> Dict[str, Any]:
    profiles = []
    unknown_items = []
    due_diligence = []
    error_messages = []
    source_ids = []
    for profile in company.profile_results:
        profiles.append({
            "name": profile.profile_name,
            "score": _score_text(profile.weighted_score),
            "coverage": _coverage_text(profile.evidence_coverage),
            "category": profile.result_category.value,
            "decision": profile.decision.value if profile.decision else "판정 미확인",
            "recommendation": _truncate(profile.recommendation, 130),
        })
        unknown_items.extend(profile.unknown_items)
        due_diligence.extend(profile.due_diligence_questions)
        if profile.error_message:
            error_messages.append(f"{profile.profile_name}: {profile.error_message}")
        source_ids.extend(profile.supporting_evidence_ids)
        source_ids.extend(profile.contrary_evidence_ids)

    return {
        "name": company.company_name,
        "description": COMPANY_DESCRIPTIONS.get(company.company_name, "Battery Intelligence 분석 대상 기업"),
        "score": _score_text(company.average_score),
        "score_numeric": company.average_score,
        "std_dev": _score_text(company.score_std_dev),
        "coverage": f"{company.evidence_coverage:.1f}%",
        "coverage_numeric": company.evidence_coverage,
        "category": company.result_category.value,
        "decision": company.final_decision,
        "summary": _truncate(company.summary_reason, 185),
        "domains": {
            DOMAIN_LABELS[domain]: _score_text(_domain_score(company, domain))
            for domain in Domain
        },
        "profiles": profiles,
        "unknown_items": [_truncate(item, 100) for item in _unique(unknown_items)[:6]],
        "due_diligence": [_truncate(item, 105) for item in _unique(due_diligence)[:6]],
        "errors": [_truncate(item, 110) for item in _unique(error_messages)[:3]],
        "source_ids": _unique(source_ids),
    }


def _set_chart_font() -> None:
    import matplotlib.font_manager as font_manager

    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (font for font in candidates if font in available), "DejaVu Sans"
    )
    plt.rcParams["axes.unicode_minus"] = False


def _write_charts(company_rows: list[Dict[str, Any]], images_dir: Path) -> tuple[Path, Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    _set_chart_font()
    colors = ["#2f75b5", "#3e9b78", "#7552aa", "#dc7d2d", "#68768a"]

    bar_chart_path = images_dir / "bar_chart.png"
    names = [row["name"] for row in company_rows]
    scores = [row["score_numeric"] or 0 for row in company_rows]
    fig, ax = plt.subplots(figsize=(6.4, 2.25))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors[: len(names)], height=0.38)
    ax.set_yticks(y_pos, labels=names)
    ax.set_xlim(0, 108)
    ax.invert_yaxis()
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)
    for index, row in enumerate(company_rows):
        x = (row["score_numeric"] or 0) + 1.5
        ax.text(x, index, row["score"], va="center", fontsize=8, color="#1b2a41")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(bar_chart_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)

    donut_chart_path = images_dir / "donut_chart.png"
    count = max(1, len(company_rows))
    fig, axes = plt.subplots(1, count, figsize=(6.4, 2.25))
    if count == 1:
        axes = [axes]
    for index, row in enumerate(company_rows):
        ax = axes[index]
        coverage = max(0.0, min(100.0, row["coverage_numeric"]))
        ax.pie(
            [coverage, 100 - coverage],
            colors=[colors[index % len(colors)], "#e7ecf2"],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.28, "edgecolor": "white"},
        )
        ax.text(0, 0.04, f"{coverage:.0f}%", ha="center", va="center", fontsize=10)
        ax.set_title(row["name"], y=-0.12, fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(donut_chart_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return bar_chart_path, donut_chart_path


def generate_pdf_report(comparison_result: Any, output_path: Optional[Path | str] = None) -> str:
    """판단 유보·부분 근거 부족·Agent 오류를 포함해 항상 5쪽 보고서를 생성합니다."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    pdf_path = Path(output_path) if output_path else output_dir / "final_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    companies = comparison_result.companies
    if not companies:
        raise ValueError("PDF 보고서에 표시할 기업 결과가 없습니다.")

    company_rows = [_company_context(company) for company in companies]
    scored_rows = [row for row in company_rows if row["score_numeric"] is not None]
    best_company = max(scored_rows, key=lambda row: row["score_numeric"], default=None)
    if best_company is None:
        headline = "전체 기업 판단 유보"
        headline_detail = "산출 가능한 Profile 점수가 없어 우선검토 기업을 선정하지 않았습니다."
    else:
        headline = best_company["decision"]
        headline_detail = f"{best_company['name']} | {best_company['score']}점"

    images_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    bar_chart_path, donut_chart_path = _write_charts(company_rows, images_dir)

    profiles_dict: Dict[str, list[Dict[str, str]]] = {}
    source_ids = []
    for company in company_rows:
        for profile in company["profiles"]:
            profiles_dict.setdefault(profile["name"], []).append({
                "company": company["name"], **profile
            })
        source_ids.extend(company["source_ids"])

    raw_dir = base_dir / "data" / "raw"
    input_documents = sorted(path.name for path in raw_dir.iterdir() if path.is_file()) if raw_dir.exists() else []
    category_counts: Dict[str, int] = {}
    for company in company_rows:
        category_counts[company["category"]] = category_counts.get(company["category"], 0) + 1

    env = Environment(
        loader=FileSystemLoader(str(base_dir / "src" / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    html_content = template.render(
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        companies=company_rows,
        best_company=best_company,
        headline=headline,
        headline_detail=headline_detail,
        profiles_dict=profiles_dict,
        category_counts=category_counts,
        source_ids=_unique(source_ids)[:32],
        input_documents=input_documents,
        bar_chart_path=bar_chart_path.resolve().as_uri(),
        donut_chart_path=donut_chart_path.resolve().as_uri(),
    )
    HTML(string=html_content, base_url=str(base_dir)).write_pdf(pdf_path)
    return str(pdf_path)
