import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "faiss_index"

# 속도와 다국어(한국어/영어) 처리 능력이 탁월한 최신 OpenAI 임베딩 모델 사용
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

def init_or_load_vector_db() -> FAISS:
    """FAISS 벡터 DB를 로드하거나, 없으면 임시로 빈 DB를 생성합니다."""
    embeddings = get_embeddings()
    
    if VECTOR_DB_PATH.exists():
        print(f"📦 기존 Vector DB 로드 중... ({VECTOR_DB_PATH})")
        return FAISS.load_local(
            str(VECTOR_DB_PATH), 
            embeddings,
            allow_dangerous_deserialization=True # 로컬에서 생성한 파일이므로 허용
        )
    else:
        print("⚠️ 기존 Vector DB가 없어 임시 DB를 생성합니다.")
        # 빈 DB를 만들기 위해 임시 문서 하나 생성
        dummy_docs = [
            Document(
                page_content="ACCURE Battery Intelligence는 배터리 데이터 분석 및 SOH/RUL 예측을 수행하는 독일 기반 스타트업입니다.",
                metadata={"source_type": "company_website", "evidence_grade": "C", "is_direct": True}
            ),
            Document(
                page_content="해당 기업은 최근 유럽 규제를 충족하는 배터리 안전 솔루션을 발표하여 제조물책임(PL) 위험을 관리하고 있습니다.",
                metadata={"source_type": "news_article", "evidence_grade": "B", "is_direct": True}
            )
        ]
        db = FAISS.from_documents(dummy_docs, embeddings)
        db.save_local(str(VECTOR_DB_PATH))
        return db

def retrieve_top_k(query: str, k: int = 3):
    """주어진 질문에 대해 가장 관련성 높은 문서를 검색합니다."""
    db = init_or_load_vector_db()
    # similarity_search_with_score는 문서와 점수를 튜플로 반환합니다
    results = db.similarity_search_with_score(query, k=k)
    return results
