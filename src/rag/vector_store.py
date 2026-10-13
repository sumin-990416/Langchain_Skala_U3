import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "faiss_index"

# 오픈소스 다국어 모델 사용 (교수님 요구사항 및 실습 노트북과 일치)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def init_or_load_vector_db() -> FAISS:
    """FAISS 벡터 DB를 로드합니다. 인덱스가 없으면 문서 적재를 요구합니다."""
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            f"FAISS 인덱스가 없습니다: {VECTOR_DB_PATH}. "
            "data/raw에 실제 PDF/TXT 자료를 넣고 문서 적재를 먼저 실행하세요."
        )

    print(f"📦 기존 Vector DB 로드 중... ({VECTOR_DB_PATH})")
    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True # 로컬에서 생성한 파일이므로 허용
    )

def retrieve_top_k(query: str, k: int = 3):
    """주어진 질문에 대해 가장 관련성 높은 문서를 검색합니다."""
    db = init_or_load_vector_db()
    # similarity_search_with_score는 문서와 점수를 튜플로 반환합니다
    results = db.similarity_search_with_score(query, k=k)
    return results
