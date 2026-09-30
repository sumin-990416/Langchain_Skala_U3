import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from dotenv import load_dotenv

from src.graph import graph
from src.schemas import InvestmentAgentState
from src.rag.ingest import ingest_documents, get_company_name_from_filename

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()

def main():
    console.print(Panel.fit("[bold cyan]배터리 스타트업 투자평가 Multi-Agent 파이프라인[/bold cyan] 🚀", border_style="cyan"))
    
    # 1. 파이프라인 시작 전 문서 임베딩 (Data Ingestion)
    with console.status("[bold green]문서 스캔 및 FAISS 벡터 DB 구축 중...[/bold green]", spinner="dots"):
        ingest_documents()
    console.print("[green]✔ 문서 임베딩 완료[/green]\n")
    
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
        
        task_id = progress.add_task("[cyan]파이프라인 준비 중...", total=None)
        
        for event in graph.stream(initial_state):
            for node_name, node_state in event.items():
                if node_name == "load_inputs":
                    companies = node_state.get("selected_companies", [])
                    c_names = ", ".join([c.name for c in companies])
                    progress.update(task_id, description=f"[cyan]평가 대상 로드: {c_names}[/cyan]")
                    console.print(f" ➔ 📂 분석 대상: {c_names}")
                
                elif node_name == "select_company":
                    c = node_state.get("current_company")
                    if c:
                        progress.update(task_id, description=f"[bold yellow]진행 중: {c.name}[/bold yellow]")
                        console.print(f"\n🎯 [bold yellow]타겟 기업 설정됨:[/bold yellow] {c.name}")
                        
                elif node_name == "profile_agent":
                    console.print(f" ➔ 👥 프로필 에이전트 병렬 분석 완료")
                    progress.update(task_id, description=f"[cyan]프로필 병렬 평가 진행 중...[/cyan]")
                    
                elif node_name == "synthesize_company":
                    res = node_state.get("company_result")
                    if res:
                        color = "green" if "추천" in res.final_decision or "RECOMMEND" in res.final_decision else "red"
                        console.print(f" ➔ ⚖️ [bold {color}]투자 판정:[/bold {color}] {res.final_decision} ({res.average_score:.1f}점)")
                
                elif node_name == "compare_companies":
                    console.print(" ➔ 📊 기업 비교 및 최종 테이블 작성 완료")
                    progress.update(task_id, description="[magenta]종합 비교 진행 중...[/magenta]")
                    
                elif node_name == "generate_report":
                    console.print(" ➔ 📑 최종 마크다운 보고서 생성 완료")
                    progress.update(task_id, description="[magenta]보고서 저장 중...[/magenta]")
                    
                else:
                    progress.update(task_id, description=f"[magenta]노드 실행 중: {node_name}[/magenta]")
                    console.print(f" ➔ ✨ 노드 완료: {node_name}")
                    
        progress.update(task_id, description="[bold green]파이프라인 실행 완료![/bold green]")

    console.print("\n" + "="*50)
    console.print(Panel("[bold green]✅ 파이프라인 실행 완료[/bold green]\n최종 투자 심사 보고서가 [bold yellow]output/final_multi_agent_report.md[/bold yellow]에 저장되었습니다.", border_style="green"))

if __name__ == "__main__":
    main()
