import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

def test_apis():
    print("=== API 연결 테스트 시작 ===\n")
    
    # 1. OpenAI 테스트
    try:
        from langchain_openai import ChatOpenAI
        print("1. OpenAI API 테스트 중...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        res = llm.invoke("Hello, who are you? (Please answer in 1 sentence)")
        print("✅ OpenAI 연결 성공! 응답:", res.content)
    except Exception as e:
        print("❌ OpenAI 연결 실패:", e)
        
    print("-" * 30)

    # 2. Tavily 테스트
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        print("2. Tavily Search API 테스트 중...")
        # langchain-community 0.3.x 이상일 경우 Tavily가 langchain-tavily 패키지로 분리되었을 수 있으나 
        # 일단 langchain-community 안에 있는지, 혹은 직접 요청으로 확인
        try:
            from langchain_tavily import TavilySearchResults
        except ImportError:
            pass # fallback to community if not installed
            
        tavily = TavilySearchResults(max_results=1)
        res = tavily.invoke("LangGraph release date")
        print("✅ Tavily 연결 성공! 검색 결과 갯수:", len(res))
        if res:
            print("   첫 번째 결과 url:", res[0].get('url', 'N/A'))
    except Exception as e:
        print("❌ Tavily 연결 실패:", e)

    print("-" * 30)

    # 3. LangSmith 환경변수 체크
    print("3. LangSmith 설정 확인...")
    tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    project = os.environ.get("LANGCHAIN_PROJECT")
    if tracing == "true":
        print(f"✅ LangSmith 트레이싱 활성화 됨 (프로젝트: {project})")
    else:
        print("⚠️ LangSmith 트레이싱 비활성화 상태")

    print("\n=== 테스트 종료 ===")

if __name__ == "__main__":
    test_apis()
