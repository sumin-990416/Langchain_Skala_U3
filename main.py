import warnings
warnings.filterwarnings("ignore")

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# GUI 파일 선택창을 위한 tkinter
import tkinter as tk
from tkinter import filedialog

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from src.graph import graph
from src.schemas import InvestmentAgentState
from src.rag.ingest import ingest_documents

load_dotenv()
console = Console()

def test_environment():
    """test_env.py 의 기능을 메인에 통합하여 환경 점검을 수행합니다."""
    console.print("\n[bold yellow]🔍 1단계: API 및 환경 변수 점검[/bold yellow]")
    
    # 1. OpenAI 체크
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # 가벼운 핑 테스트
        llm.invoke("ping")
        console.print("  [green]✔ OpenAI API 연결 성공[/green]")
    except Exception as e:
        console.print(f"  [red]✖ OpenAI API 연결 실패:[/red] {e}")
        console.print("  [red]=> .env 파일에 OPENAI_API_KEY를 설정해주세요.[/red]")
        sys.exit(1)
        
    # 2. LangSmith 체크
    tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    project = os.environ.get("LANGCHAIN_PROJECT")
    if tracing == "true":
        console.print(f"  [green]✔ LangSmith 트레이싱 활성화 (프로젝트: {project})[/green]")
    else:
        console.print("  [yellow]⚠ LangSmith 트레이싱 비활성화 상태[/yellow]")
        
    console.print("[green]=> 환경 점검 완료![/green]\n")

def select_files_via_gui() -> list:
    """Tkinter를 이용해 분석할 파일을 GUI로 선택합니다."""
    try:
        root = tk.Tk()
        root.withdraw() # 메인 창 숨김
        root.attributes('-topmost', True) # 창을 최상단으로
        
        file_paths = filedialog.askopenfilenames(
            title="분석할 스타트업 기업 문서(PDF, TXT 등)를 모두 선택하세요",
            filetypes=[("문서 파일", "*.pdf *.txt *.docx"), ("모든 파일", "*.*")]
        )
        root.destroy()
        return list(file_paths)
    except Exception as e:
        console.print(f"[red]GUI 창을 열 수 없습니다: {e}[/red]")
        return []

def prepare_data_directory():
    """GUI로 파일을 선택받아 data/raw/ 폴더를 준비합니다."""
    console.print("[bold yellow]📂 2단계: 분석 대상 문서 로드[/bold yellow]")
    
    raw_dir = Path(__file__).parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 파일 확인
    existing_files = list(raw_dir.glob("*.*"))
    if existing_files:
        if Confirm.ask("기존 data/raw 폴더에 파일이 있습니다. 새로 파일을 선택하시겠습니까? (기존 파일은 삭제됩니다)"):
            file_paths = select_files_via_gui()
            if not file_paths:
                console.print("[yellow]파일 선택이 취소되었습니다. 파이프라인을 종료합니다.[/yellow]")
                sys.exit(0)
            
            # 폴더 비우기
            for f in existing_files:
                f.unlink()
            
            # 새 파일 복사
            for fp in file_paths:
                shutil.copy(fp, raw_dir / Path(fp).name)
            console.print(f"[green]✔ 총 {len(file_paths)}개의 파일을 로드했습니다.[/green]\n")
        else:
            console.print("[green]✔ 기존 폴더의 파일을 그대로 사용합니다.[/green]\n")
    else:
        console.print("분석할 파일이 없습니다. 파일 선택 창을 엽니다...")
        file_paths = select_files_via_gui()
        if not file_paths:
            console.print("[yellow]파일 선택이 취소되었습니다. 파이프라인을 종료합니다.[/yellow]")
            sys.exit(0)
            
        for fp in file_paths:
            shutil.copy(fp, raw_dir / Path(fp).name)
        console.print(f"[green]✔ 총 {len(file_paths)}개의 파일을 로드했습니다.[/green]\n")


def main():
    console.print(Panel.fit("[bold cyan]배터리 스타트업 투자평가 Multi-Agent 파이프라인[/bold cyan] 🚀", border_style="cyan"))
    
    # 1. 환경 점검
    test_environment()
    
    # 2. 파일 선택 다이얼로그 (프로그램화)
    prepare_data_directory()
    
    # 3. 문서 임베딩 (Data Ingestion)
    console.print("[bold yellow]🧠 3단계: 문서 스캔 및 지식 베이스 구축[/bold yellow]")
    with console.status("[bold green]문서 청킹 및 FAISS 벡터 DB 구축 중...[/bold green]", spinner="dots"):
        ingest_documents()
    console.print("[green]✔ 문서 임베딩 완료[/green]\n")
    
    # 4. Multi-Agent Graph 구동
    console.print("[bold yellow]🤖 4단계: Multi-Agent 투자 심사 진행[/bold yellow]")
    initial_state = InvestmentAgentState(
        selected_companies=[],
        current_index=0,
        current_company=None,
        evaluation_questions=[],
        current_question=None,
        retrieved_evidence=[],
        validated_evidence=[],
        evidence_coverage=0.0,
        retry_count={},
        unknown_items=[],
        domain_scores={},
        profile_results=[],
        company_result=None,
        company_results=[],
        final_comparison=None,
        used_sources=[],
        report_draft="",
        report_errors=[],
        errors=[]
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task_id = progress.add_task("[cyan]에이전트 네트워크 준비 중...", total=None)
        
        for event in graph.stream(initial_state):
            for node_name, node_state in event.items():
                if node_name == "load_inputs":
                    companies = node_state.get("selected_companies", [])
                    c_names = ", ".join([c.name for c in companies])
                    progress.update(task_id, description=f"[cyan]평가 대상 로드: {c_names}[/cyan]")
                    console.print(f" ➔ 📂 분석 타겟: {c_names}")
                
                elif node_name == "select_company":
                    c = node_state.get("current_company")
                    if c:
                        progress.update(task_id, description=f"[bold yellow]심층 분석 진행 중: {c.name}[/bold yellow]")
                        console.print(f"\n🎯 [bold yellow]타겟 기업 설정됨:[/bold yellow] {c.name}")
                        
                elif node_name == "profile_agent":
                    console.print(f" ➔ 👥 5대 프로필 에이전트 병렬(Parallel) 분석 완료")
                    progress.update(task_id, description=f"[cyan]다중 관점 평가 종합 중...[/cyan]")
                    
                elif node_name == "synthesize_company":
                    res = node_state.get("company_result")
                    if res:
                        color = "green" if "추천" in res.final_decision or "RECOMMEND" in res.final_decision else "red"
                        score_text = "점수 없음" if res.average_score is None else f"{res.average_score:.1f}점"
                        if res.average_score is not None and any(
                            p.weighted_score is None for p in res.profile_results
                        ):
                            score_text += " · 일부 프로필 참고 평균"
                        console.print(f" ➔ ⚖️ [bold {color}]최종 투자 판정:[/bold {color}] {res.final_decision} ({score_text})")
                
                elif node_name == "compare_companies":
                    console.print(" ➔ 📊 기업 비교 및 최종 테이블 작성 완료")
                    progress.update(task_id, description="[magenta]종합 보고서 생성 중...[/magenta]")
                    
                elif node_name == "generate_report":
                    console.print(" ➔ 📑 투자 심사 마크다운 보고서 생성 완료")
                    progress.update(task_id, description="[magenta]보고서 저장 중...[/magenta]")
                    
                elif node_name in ["build_questions", "retrieve_evidence", "score_common"]:
                    progress.update(task_id, description=f"[magenta]에이전트 노드 통과: {node_name}[/magenta]")
                    console.print(f" ➔ ✨ Sub-Agent 작업 완료: {node_name}")
                    
        progress.update(task_id, description="[bold green]Multi-Agent 파이프라인 실행 완료![/bold green]")

    md_report_path = Path(__file__).parent / "output" / "final_multi_agent_report.md"
    pdf_report_path = Path(__file__).parent / "output" / "final_report.pdf"
    
    console.print("\n" + "="*50)
    
    if pdf_report_path.exists():
        console.print(Panel(f"[bold green]✅ 파이프라인 실행 완료[/bold green]\n최종 투자 심사 보고서가 [bold yellow]{pdf_report_path.name}[/bold yellow]에 저장되었습니다.", border_style="green"))
        console.print("\n[bold cyan]📄 생성된 PDF 보고서를 자동으로 엽니다...[/bold cyan]")
        os.system(f"open '{pdf_report_path}'")
    else:
        console.print(Panel(f"[bold green]✅ 파이프라인 실행 완료[/bold green]\n최종 투자 심사 보고서가 [bold yellow]{md_report_path.name}[/bold yellow]에 저장되었습니다.", border_style="green"))
        if md_report_path.exists():
            console.print("\n[bold cyan]📄 생성된 마크다운 보고서를 자동으로 엽니다...[/bold cyan]")
            os.system(f"open '{md_report_path}'")

if __name__ == "__main__":
    main()
