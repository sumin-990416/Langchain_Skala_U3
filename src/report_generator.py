import datetime
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Apple Silicon의 Homebrew 라이브러리를 WeasyPrint가 찾을 수 있게 합니다.
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    "/opt/homebrew/lib:/usr/local/lib:"
    + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
)

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


def _canonical_source_id(source_id: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", source_id.upper())


def build_report_summary(companies: list[Any]) -> str:
    """Return a concise investment conclusion, not an overview slide."""
    total = len(companies)
    on_hold = sum("유보" in company.final_decision for company in companies)
    if total and on_hold == total:
        opening = (
            f"검토 대상 {total}개 기업 모두 공개 근거만으로 투자 판단을 확정하기 어려워 "
            "현재 우선 투자 대상을 선정하지 않았으며, 추가 실사 후 재검토가 필요하다."
        )
    else:
        opening = f"검토 대상 {total}개 기업의 투자 판단과 근거 수준을 요약한다."
    company_summaries = []
    for company in companies:
        scored = sum(profile.weighted_score is not None for profile in company.profile_results)
        total_profiles = len(company.profile_results)
        company_summaries.append(
            f"{company.company_name}: {company.final_decision}, "
            f"근거충족률 {company.evidence_coverage:.1f}%, "
            f"점수 산출 {scored}/{total_profiles}개 관점"
        )
    return opening + (" " + " / ".join(company_summaries) if company_summaries else "")


def collect_reference_entries(
    source_ids: Iterable[str], raw_dir: Optional[Path] = None
) -> Dict[str, list[Dict[str, Any]]]:
    """Resolve only cited evidence IDs to bibliographic details present in input PDFs."""
    requested: Dict[str, list[str]] = {}
    for source_id in source_ids:
        if source_id.strip():
            canonical_id = _canonical_source_id(source_id)
            requested.setdefault(canonical_id, [])
            if source_id not in requested[canonical_id]:
                requested[canonical_id].append(source_id)
    if not requested:
        return {}

    documents_dir = raw_dir or Path(__file__).parent.parent / "data" / "raw"
    card_start = re.compile(r"(?m)^\s*([AVEG]\d{2})\s*\|\s*([^|\n]+)\s*\|\s*([^\n]+)")
    url_pattern = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
    resolved: Dict[str, Dict[str, str]] = {}

    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            matches = list(card_start.finditer(text))
            page_card: Optional[Dict[str, str]] = None
            for index, match in enumerate(matches):
                canonical_id = _canonical_source_id(match.group(1))
                if canonical_id in resolved:
                    continue
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                lines = [line.strip() for line in text[match.start():end].splitlines() if line.strip()]
                if len(lines) < 3:
                    continue
                title = lines[1]
                block = "\n".join(lines)
                author_match = re.search(r"작성\s*주체\s*[·:]\s*([^/\n|]+)", block)
                author = author_match.group(1).strip() if author_match else match.group(2).strip()
                date_match = re.search(
                    r"발행일\s*[·:]\s*([^|\n]+)(?:\|\s*확인일\s*[·:]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}))?",
                    block,
                )
                publication_date = date_match.group(1).strip() if date_match else "미상"
                checked_date = date_match.group(2).strip() if date_match and date_match.group(2) else ""
                urls = url_pattern.findall(block)
                url = urls[-1].rstrip(".,;") if urls else ""
                site = urlparse(url).hostname or match.group(2).strip()
                site = site.removeprefix("www.")
                card_kind = match.group(3).strip()

                if "논문" in card_kind or "연구" in card_kind:
                    kind = "학술 논문"
                    journal = ""
                    title_parts = title.rsplit(" | ", 1)
                    if len(title_parts) == 2:
                        title, journal = title_parts
                    citation_block = re.sub(
                        r"확인일\s*[·:]\s*\d{4}-\d{2}-\d{2}", "", block
                    )
                    year_match = re.search(r"\b(?:19|20)\d{2}\b", publication_date)
                    if year_match is None:
                        year_match = re.search(r"\b(?:19|20)\d{2}\b", citation_block[match.end():])
                    year = year_match.group(0) if year_match else ""
                    citation = (
                        f"{author}({year or '발행연도 미상'}). {title}."
                        + (f" *{journal}*." if journal else "")
                        + (f" {url}" if url else "")
                    )
                elif "보고서" in card_kind:
                    kind = "기관 보고서"
                    year_match = re.search(r"\b(?:19|20)\d{2}\b", publication_date)
                    date_label = year_match.group(0) if year_match else "발행연도 미상"
                    if author.strip() in {"기업", "기관", "개인", "미상"}:
                        author = match.group(2).strip()
                    citation = f"{author}({date_label}). *{title}*." + (f" {url}" if url else "")
                else:
                    kind = "웹페이지"
                    if author.strip() in {"기업", "기관", "개인", "미상"}:
                        author = match.group(2).strip()
                    date_label = publication_date if publication_date != "미상" else (
                        checked_date or "날짜 미상"
                    )
                    citation = f"{author}({date_label}). *{title}*. {site}." + (f" {url}" if url else "")

                entry = {
                    "kind": kind,
                    "citation": citation,
                    "source_id": match.group(1),
                }
                resolved[canonical_id] = entry
                if page_card is None:
                    page_card = entry
            if page_card is not None:
                page_alias = _canonical_source_id(f"{pdf_path.stem}:p{page_number}")
                resolved.setdefault(page_alias, page_card)

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for canonical_id, requested_ids in requested.items():
        entry = resolved.get(canonical_id)
        if entry is None:
            continue
        category_entries = grouped.setdefault(entry["kind"], [])
        matching_entry = next(
            (
                existing for existing in category_entries
                if existing["source_id"] == entry["source_id"]
            ),
            None,
        )
        if matching_entry is None:
            category_entries.append({**entry, "requested_ids": requested_ids.copy()})
        else:
            matching_entry["requested_ids"].extend(
                source_id
                for source_id in requested_ids
                if source_id not in matching_entry["requested_ids"]
            )
    return grouped


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
        "std_dev": _score_text(company.score_std_dev),
        "coverage": f"{company.evidence_coverage:.1f}%",
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


def generate_pdf_report(
    comparison_result: Any,
    output_path: Optional[Path | str] = None,
    report_summary: Optional[str] = None,
) -> str:
    """Generate a concise-summary PDF with cited references and a 10-page cap."""
    # WeasyPrint의 시스템 의존성이 없더라도 다른 모듈과 Markdown 생성은
    # 계속되도록 실제 PDF 생성 시점에만 불러옵니다.
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PDF 생성에 필요한 WeasyPrint 시스템 라이브러리(Pango/GLib)를 불러올 수 없습니다."
        ) from exc

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    pdf_path = Path(output_path) if output_path else output_dir / "final_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    companies = comparison_result.companies
    if not companies:
        raise ValueError("PDF 보고서에 표시할 기업 결과가 없습니다.")

    company_rows = [_company_context(company) for company in companies]
    profiles_dict: Dict[str, list[Dict[str, str]]] = {}
    source_ids = []
    for company in company_rows:
        for profile in company["profiles"]:
            profiles_dict.setdefault(profile["name"], []).append({
                "company": company["name"], **profile
            })
        source_ids.extend(company["source_ids"])

    reference_entries = collect_reference_entries(source_ids)
    env = Environment(
        loader=FileSystemLoader(str(base_dir / "src" / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    html_content = template.render(
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        report_summary=report_summary or build_report_summary(companies),
        companies=company_rows,
        profiles_dict=profiles_dict,
        reference_groups=reference_entries,
    )
    HTML(string=html_content, base_url=str(base_dir)).write_pdf(pdf_path)
    from pypdf import PdfReader
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > 10:
        pdf_path.unlink(missing_ok=True)
        raise ValueError(f"보고서가 최대 10페이지를 초과했습니다: {page_count}페이지")
    return str(pdf_path)
