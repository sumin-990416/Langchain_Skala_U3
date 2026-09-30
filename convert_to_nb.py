import nbformat as nbf
import os

# 디렉토리 생성
folders = [
    "01-Schema",
    "02-RAG",
    "03-Profile-Agents",
    "04-LangGraph"
]
for f in folders:
    os.makedirs(f, exist_ok=True)

def create_nb(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for ctype, content in cells_data:
        if ctype == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        elif ctype == "code":
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

# 01. Schema Notebook
create_nb("01-Schema/01-State-Schema.ipynb", [
    ("md", "# 01. State 및 Schema 정의\nLangGraph에서 사용할 데이터 모델과 Enum을 정의합니다."),
    ("code", """import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Domain(str, Enum):
    TECHNOLOGY = "technology"
    MARKET = "market"
    FINANCE = "finance"
    RISK = "risk"
    TEAM = "team"

class GateStatus(str, Enum):
    PASS = "PASS"
    HOLD_INSUFFICIENT_EVIDENCE = "HOLD_INSUFFICIENT_EVIDENCE"
    HOLD_CRITICAL_RISK = "HOLD_CRITICAL_RISK"
    HOLD_NO_DATA_RIGHTS = "HOLD_NO_DATA_RIGHTS"
    HOLD_MATERIAL_CONTRADICTION = "HOLD_MATERIAL_CONTRADICTION"
    HOLD_FINANCIAL_SURVIVAL = "HOLD_FINANCIAL_SURVIVAL"

class FinalDecision(str, Enum):
    RECOMMEND = "투자검토 추천"
    CONDITIONAL = "조건부 검토"
    HOLD_RAG = "근거 부족 판단 유보"
    HOLD_GATE = "투자 보류 및 추가 실사"
    HOLD_SCORE = "보류 (기준 미달)"
"""),
    ("code", """class CompanySnapshot(BaseModel):
    name: str = Field(description="기업명")
    description: str = Field(default="", description="기업 기본 설명")
    is_unlisted: bool = Field(default=True)
    investment_stage: str = Field(default="Seed")
    has_exited: bool = Field(default=False)

class Question(BaseModel):
    domain: Domain
    item_id: str
    query: str

class Evidence(BaseModel):
    item_id: str
    content: str
    source_type: str
    url: Optional[str] = None
    evidence_grade: str = "U"
    is_direct: bool = False
    has_contradiction: bool = False

class GateResult(BaseModel):
    status: GateStatus = GateStatus.PASS
    reason: Optional[str] = None

class ProfileResult(BaseModel):
    profile_name: str
    profile_score: float
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    questions_for_dd: List[str] = Field(default_factory=list)

class ConsensusResult(BaseModel):
    final_score: float
    score_std_dev: float
    is_high_disagreement: bool = False
    final_decision: FinalDecision
    summary_reason: str

class AgentState(BaseModel):
    current_company: Optional[CompanySnapshot] = None
    evaluation_questions: List[Question] = Field(default_factory=list)
    retrieved_evidence: List[Evidence] = Field(default_factory=list)
    validated_evidence: List[Evidence] = Field(default_factory=list)
    retry_count: Dict[str, int] = Field(default_factory=dict)
    unknown_items: List[str] = Field(default_factory=list)
    domain_scores: Dict[Domain, float] = Field(default_factory=dict)
    evidence_coverage: float = 0.0
    gate_result: Optional[GateResult] = None
    profile_results: Annotated[List[ProfileResult], operator.add] = Field(default_factory=list)
    consensus_result: Optional[ConsensusResult] = None
    report_errors: List[str] = Field(default_factory=list)
""")
])

# 02. RAG Notebook
create_nb("02-RAG/01-VectorDB-Retrieval.ipynb", [
    ("md", "# 02. 문서 검색 (RAG)\nFAISS와 HuggingFace Embedding을 이용한 검색 로직입니다."),
    ("code", """import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTOR_DB_PATH = "../data/processed/faiss_index"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def init_or_load_vector_db() -> FAISS:
    embeddings = get_embeddings()
    if os.path.exists(VECTOR_DB_PATH):
        print("기존 DB 로드")
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("임시 DB 생성")
        dummy_docs = [
            Document(page_content="ACCURE Battery Intelligence는 SOH/RUL 예측 기업입니다.", metadata={"source_type": "company_website", "evidence_grade": "C", "is_direct": True})
        ]
        db = FAISS.from_documents(dummy_docs, embeddings)
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        db.save_local(VECTOR_DB_PATH)
        return db

def retrieve_top_k(query: str, k: int = 3):
    db = init_or_load_vector_db()
    return db.similarity_search_with_score(query, k=k)
""")
])

# 03. Agent Notebook
create_nb("03-Profile-Agents/01-Profile-Agent.ipynb", [
    ("md", "# 03. Profile Agent\n각 투자 성향별로 가중치를 다르게 적용하고 LLM 의견을 묻는 에이전트"),
    ("code", """import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class LLMProfileOutput(BaseModel):
    pros: list[str] = Field(description="찬성 근거")
    cons: list[str] = Field(description="반대 근거")
    questions_for_dd: list[str] = Field(description="추가 실사 질문")

def run_profile_agent(state_dict, profile_id):
    # 실제로는 state에 들어있는 데이터를 기반으로 LLM을 호출합니다.
    print(f"[{profile_id}] 에이전트 실행 중...")
    # Mock return for Jupyter tutorial
    return {
        "profile_name": profile_id,
        "profile_score": 85.0,
        "pros": ["기술력 우수"],
        "cons": ["수익성 부족"],
        "questions_for_dd": ["구체적 매출 발생 시점은?"]
    }
""")
])

# 04. LangGraph Notebook
create_nb("04-LangGraph/01-Main-Graph.ipynb", [
    ("md", "# 04. LangGraph 워크플로우\n전체 과정을 연결하는 StateGraph 구성"),
    ("code", """from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import TypedDict

# State 정의 및 노드 함수는 앞선 노트북에서 정의한 내용들을 가져와서 조립합니다.
# 튜토리얼용으로 간략한 뼈대를 짭니다.

class DummyState(TypedDict):
    company_name: str
    gate_passed: bool
    results: list

def node_gate(state):
    print("Gate 판정")
    return {"gate_passed": True}

def node_profile(data):
    print(f"Profile: {data['profile']}")
    return {"results": [data['profile']]}

def route_gate(state):
    if state["gate_passed"]:
        profiles = ["tech", "market", "finance"]
        return [Send("node_profile", {"profile": p}) for p in profiles]
    return END

graph_builder = StateGraph(DummyState)
graph_builder.add_node("node_gate", node_gate)
graph_builder.add_node("node_profile", node_profile)

graph_builder.add_edge(START, "node_gate")
graph_builder.add_conditional_edges("node_gate", route_gate, ["node_profile", END])
graph_builder.add_edge("node_profile", END)

graph = graph_builder.compile()
""")
])
print("Notebooks created!")
