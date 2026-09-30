import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.rag.vector_store import get_embeddings, VECTOR_DB_PATH

load_dotenv()

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"

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
                d.metadata["evidence_grade"] = "B" # 문서 내부 출처별 세분화 전 기본값
                d.metadata["is_direct"] = True
            
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
