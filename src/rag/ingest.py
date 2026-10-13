import os
import re
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.rag.vector_store import get_embeddings, VECTOR_DB_PATH
import yaml

load_dotenv()

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"
SOURCE_POLICY_PATH = Path(__file__).parent.parent.parent / "configs" / "source_policy.yaml"


def _source_metadata(page_content: str) -> dict:
    """PDF/TXT에 실제로 적힌 URL의 허용 도메인으로만 출처 등급을 정합니다."""
    policy = yaml.safe_load(SOURCE_POLICY_PATH.read_text(encoding="utf-8"))
    domain_tiers = {
        domain.casefold(): tier
        for tier, domains in policy.get("tiers", {}).items()
        for domain in domains
    }
    official_domains = [d.casefold() for d in policy.get("company_official_domains", [])]
    urls = list(dict.fromkeys(
        match.rstrip(".,;:)]}")
        for match in re.findall(r"https?://[^\s<>\"']+", page_content)
    ))
    hosts = [(url, (urlparse(url).hostname or "").casefold().removeprefix("www."))
             for url in urls]
    ids = list(dict.fromkeys(re.findall(r"\b[AVEG]-?\d{2}\b", page_content)))
    url_by_id = {}
    for evidence_id, url in re.findall(
        r"(?:(?:원문|original(?:\s+source)?)\s*)?\[([AVEG]-?\d{2})\]\s*[:：]?\s*(https?://[^\s<>\"']+)",
        page_content,
    ):
        url_by_id[evidence_id.upper()] = url.rstrip(".,;:)]}")
    if not url_by_id and len(ids) == len(urls) == 1:
        url_by_id[ids[0].upper()] = urls[0]
    tiers = []
    for _, host in hosts:
        tier = next((value for domain, value in domain_tiers.items()
                     if host == domain or host.endswith("." + domain)), None)
        tiers.append(tier)

    tier_by_url = {url: tier for (url, _), tier in zip(hosts, tiers)}
    url_hosts = {url: host for url, host in hosts}
    urls_by_id = {key: url for key, url in url_by_id.items() if url in url_hosts}
    tiers_by_id = {}
    publishers_by_id = {}
    for key, url in urls_by_id.items():
        host = url_hosts[url]
        tiers_by_id[key] = tier_by_url.get(url) or "U"
        publishers_by_id[key] = host

    # URL이 없거나 출처가 허용목록 밖이거나 티어가 섞이면 등급을 추정하지 않습니다.
    grade = tiers[0] if tiers and all(tier == tiers[0] and tier is not None for tier in tiers) else "U"
    unique_hosts = set(host for _, host in hosts)
    official = bool(hosts) and all(
        any(host == domain or host.endswith("." + domain) for domain in official_domains)
        for _, host in hosts
    )
    return {
        "evidence_grade": grade,
        "source_tier": grade if grade in {"A", "B", "C"} else "",
        "source_url": urls[0] if len(urls) == 1 else "",
        "publisher": next(iter(unique_hosts)) if len(unique_hosts) == 1 else "",
        "is_official": official,
        "source_urls_by_id": urls_by_id,
        "source_tiers_by_id": tiers_by_id,
        "source_publishers_by_id": publishers_by_id,
    }

def get_company_name_from_filename(filename: str) -> str:
    """파일명에서 현재 평가 대상 기업명을 일관되게 정규화합니다."""
    lower = filename.casefold()
    if "accure" in lower:
        return "ACCURE"
    if "volytica" in lower:
        return "volytica diagnostics"
    if "electra" in lower:
        return "Electra Vehicles"
    return "ALL"

def ingest_documents():
    print(f"데이터 스캔 시작... (경로: {RAW_DATA_PATH})")
    
    if not RAW_DATA_PATH.exists():
        RAW_DATA_PATH.mkdir(parents=True)
    
    docs = []
    
    # 1. 파일 읽기
    for file in RAW_DATA_PATH.iterdir():
        if file.is_file():
            company_name = get_company_name_from_filename(file.name)
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
                loaded_docs = loader.load()
            elif file.suffix.lower() == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
                loaded_docs = loader.load()
            else:
                continue
                
            # 메타데이터에 기업명 등 주입
            for page_index, d in enumerate(loaded_docs, start=1):
                d.metadata["company"] = company_name
                d.metadata["source_type"] = "document"
                d.metadata["source_file"] = file.name
                d.metadata["source_id"] = f"{file.stem}:p{page_index}"
                d.metadata.update(_source_metadata(d.page_content))
            
            docs.extend(loaded_docs)
            print(f"📄 읽은 파일: {file.name} (기업명: {company_name}) - {len(loaded_docs)} 페이지/섹션")

    if not docs:
        print("⚠️ data/raw/ 폴더에 읽을 수 있는 문서가 없습니다! (.pdf 또는 .txt를 넣어주세요)")
        return

    # 2. Text Splitter 설정
    # 리트리버 단위를 줄이고 속도/정확도를 올리기 위해 청크 크기를 500, 오버랩 50으로 세팅합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f"✂️ 청킹 완료: 총 {len(split_docs)} 개의 청크 생성 (크기: 500, 오버랩: 50)")

    # 3. FAISS 벡터 DB 저장
    print("🧠 OpenAI 임베딩 및 FAISS DB 생성 중...")
    embeddings = get_embeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    db.save_local(str(VECTOR_DB_PATH))
    
    print(f"✅ DB 저장 완료! (위치: {VECTOR_DB_PATH})")

if __name__ == "__main__":
    ingest_documents()
