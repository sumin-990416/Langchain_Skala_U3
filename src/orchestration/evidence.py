"""동적 조사 질문별 검색. 전체 원문은 Worker 로컬/외부 bundle에서만 보관."""

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from src.orchestration.schemas import Perspective, ResearchTask, SourceRecord


_DB = None
_DB_STAMP = None
_DB_LOCK = Lock()


def _get_db():
    global _DB, _DB_STAMP
    from src.rag.vector_store import VECTOR_DB_PATH, get_embeddings
    from langchain_community.vectorstores import FAISS

    index = VECTOR_DB_PATH / "index.faiss"
    if not index.is_file():
        raise FileNotFoundError("FAISS 인덱스가 없습니다. 문서 ingest를 먼저 실행하세요.")
    stamp = index.stat().st_mtime_ns
    with _DB_LOCK:
        if _DB is None or _DB_STAMP != stamp:
            # data/processed에서 이 프로젝트가 생성한 로컬 인덱스만 로드한다.
            _DB = FAISS.load_local(str(VECTOR_DB_PATH), get_embeddings(),
                                  allow_dangerous_deserialization=True)
            _DB_STAMP = stamp
    return _DB


def _company_matches(doc, company_name: str) -> bool:
    company = str(doc.metadata.get("company", "ALL")).casefold()
    target = company_name.casefold()
    if company not in {"all", "", "unknown"}:
        return company == target
    aliases = {"volytica diagnostics": "volytica", "electra vehicles": "electra"}
    needle = aliases.get(target, target)
    return needle in doc.page_content.casefold()


def source_from_document(doc) -> SourceRecord:
    metadata = doc.metadata
    source_file = str(metadata.get("source_file") or Path(str(metadata.get("source", "unknown"))).name)
    page = int(metadata.get("page", 0)) + 1
    content = doc.page_content.strip()[:2000]
    digest = hashlib.sha256(f"{source_file}|{page}|{content}".encode()).hexdigest()[:12]
    url = str(metadata.get("source_url") or metadata.get("url") or "")
    # 문서 본문에 URL이 여러 개 있으면 해당 청크의 원 출처를 추측하지 않는다.
    urls = re.findall(r"https?://[^\s<>]+", content)
    if not url and len(urls) == 1:
        url = urls[0].rstrip(".,;)")
    group = str(metadata.get("publisher") or metadata.get("source_group") or "")
    if not group:
        group = urlparse(url).hostname or source_file
    return SourceRecord(
        evidence_id=f"EV-{digest}",
        source_id=str(metadata.get("source_id") or f"{Path(source_file).stem}:p{page}"),
        source_file=source_file, page=page, url=url, source_group=group,
        source_type=str(metadata.get("source_type", "document")), content=content,
    )


def _tokens(value: str) -> Counter:
    """영문 토큰과 한글 부분어를 함께 사용해 조사형 검색어도 일치시킨다."""
    terms = re.findall(r"[a-z0-9]+|[가-힣]+", value.casefold())
    expanded = []
    for term in terms:
        expanded.append(term)
        if re.fullmatch(r"[가-힣]+", term) and len(term) >= 3:
            expanded.extend(term[index:index + 2] for index in range(len(term) - 1))
    return Counter(expanded)


def _card_blocks(text: str, filename: str) -> list[tuple[str, str]]:
    if "balanced_evidence" in filename:
        pattern = r"(?m)^\[([AVE]-\d{2})\]"
    else:
        pattern = r"(?m)^([AVEG]\d{2})\s*\|"
    starts = list(re.finditer(pattern, text))
    if not starts:
        return []
    return [(match.group(1), text[match.start():starts[i + 1].start() if i + 1 < len(starts) else len(text)].strip())
            for i, match in enumerate(starts)]


def _publisher(content: str, url: str, company: str) -> str:
    author = re.search(r"작성 주체\s*[·:]\s*([^\n/]+)", content)
    if author:
        publisher = author.group(1).strip()
        # SEC의 호스팅 도메인을 회사 발표의 독립 발행자로 세지 않는다.
        if publisher in {"기업", "ELECTRA", "ELECTRA/거래 당사자", "특허 출원인"} or "거래 당사자" in content:
            return company if company != "ALL" else "company_origin"
        return publisher
    if company != "ALL":
        return company
    return urlparse(url).hostname or "origin_not_identified"


@lru_cache(maxsize=4)
def _load_keyword_corpus(stamp: tuple) -> tuple:
    from pypdf import PdfReader
    records = []
    for filename, _modified in stamp:
        path = Path(filename)
        for page, item in enumerate(PdfReader(path).pages, start=1):
            text = (item.extract_text() or "").replace("\x00", " ")
            for source_id, content in _card_blocks(text, path.name):
                prefix = source_id[0]
                company = {"A": "ACCURE", "V": "volytica diagnostics", "E": "Electra Vehicles", "G": "ALL"}[prefix]
                urls = re.findall(r"https?://[^\s<>]+", content)
                url = urls[-1].rstrip(".,;)") if urls else ""
                digest = hashlib.sha256(f"{path.name}|{page}|{source_id}|{content}".encode()).hexdigest()[:12]
                source = SourceRecord(
                    evidence_id=f"EV-{digest}", source_id=source_id,
                    source_file=path.name, page=page, url=url,
                    source_group=_publisher(content, url, company),
                    source_type="supplied_evidence_summary" if "balanced_evidence" in path.name else "source_card",
                    content=content[:2000],
                )
                records.append((company, source, _tokens(content)))
    return tuple(records)


def retrieve_keyword_evidence(task: ResearchTask) -> list[SourceRecord]:
    """명시적으로 선택하는 PDF 근거카드 TF-IDF 검색; 임베딩을 가장하지 않는다."""
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    stamp = tuple((str(path), path.stat().st_mtime_ns) for path in sorted(raw.glob("*.pdf")))
    corpus = _load_keyword_corpus(stamp)
    candidates = [(company, record, tokens) for company, record, tokens in corpus
                  if company.casefold() == task.company_name.casefold() or company == "ALL"]
    if not candidates:
        return []
    query = _tokens(" ".join(task.queries) + " " + task.question)
    idf = {term: math.log((len(candidates) + 1) / (1 + sum(term in terms for _, _, terms in candidates))) + 1
           for term in query}
    ranked = []
    for company, record, terms in candidates:
        score = sum(min(terms.get(term, 0), 3) * weight for term, weight in idf.items()) / math.sqrt(max(1, sum(terms.values())))
        # 특정 기업의 기록이 먼저, 공통 자료는 산업/검증 맥락으로 별도 제공한다.
        ranked.append((score, company, record))
    own = sorted((entry for entry in ranked if entry[1] != "ALL"), key=lambda item: item[0], reverse=True)
    general = sorted((entry for entry in ranked if entry[1] == "ALL"), key=lambda item: item[0], reverse=True)
    selected = own[:11] + general[:3]
    # 공시의 불리한 재무·고객 집중 근거가 긍정 도입 사례 검색에서 누락되지 않게 한다.
    if task.company_name == "Electra Vehicles" and task.perspective in {Perspective.MARKET, Perspective.STAKEHOLDERS}:
        financial = next((entry for entry in own if entry[2].source_id == "E09"), None)
        if financial and financial not in selected:
            selected.append(financial)
    return [entry[2] for entry in selected[:15]]


def retrieve_task_evidence(task: ResearchTask, mode: str = "faiss") -> list[SourceRecord]:
    if mode == "keyword":
        return retrieve_keyword_evidence(task)
    if mode != "faiss":
        raise ValueError("지원하지 않는 검색 모드입니다.")
    db = _get_db()
    records = {}
    for query in task.queries:
        for doc, _score in db.similarity_search_with_score(f"{task.company_name} {query}", k=12):
            if doc.page_content.strip() and _company_matches(doc, task.company_name):
                record = source_from_document(doc)
                records[record.evidence_id] = record
                if len(records) >= 15:
                    return list(records.values())
    return list(records.values())
