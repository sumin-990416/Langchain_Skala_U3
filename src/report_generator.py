import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

# Mac(Apple Silicon)에서 Homebrew로 설치한 라이브러리를 WeasyPrint가 찾을 수 있도록 환경변수 강제 주입
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:/usr/local/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")

from weasyprint import HTML

def generate_pdf_report(comparison_result):
    # 경로 설정
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    images_dir = output_dir / "images"
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    companies = comparison_result.companies
    if not companies:
        return None
        
    # 미산출 점수는 순위에 포함하지 않는다.
    scored_companies = [c for c in companies if c.average_score is not None]
    best_company = max(scored_companies, key=lambda c: c.average_score, default=None)
    
    # 한글 폰트 설정 (Mac)
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Bar Chart 생성 (최종 투자지표)
    bar_chart_path = images_dir / "bar_chart.png"
    plt.figure(figsize=(6, 3))
    names = [c.company_name for c in companies]
    scores = [c.average_score if c.average_score is not None else 0 for c in companies]
    y_pos = np.arange(len(names))
    
    colors = ['#1976D2', '#4CAF50', '#9C27B0']
    plt.barh(y_pos, scores, color=colors, height=0.5)
    plt.yticks(y_pos, names)
    plt.xlim(0, 110)
    plt.gca().invert_yaxis()  # 상위 항목이 위에 오도록
    
    # 레이블 추가
    for i, company in enumerate(companies):
        score = company.average_score
        label = "점수 없음" if score is None else f"{score:.1f}"
        plt.text(1 if score is None else score + 1, i, label, va='center')
        
    plt.box(False)
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Donut Chart 생성 (근거충족률)
    donut_chart_path = images_dir / "donut_chart.png"
    fig, axes = plt.subplots(1, len(companies), figsize=(8, 3))
    if len(companies) == 1:
        axes = [axes]
        
    for i, c in enumerate(companies):
        ax = axes[i]
        coverage = c.evidence_coverage
        sizes = [coverage, max(0, 100 - coverage)]
        ax.pie(sizes, colors=[colors[i%len(colors)], '#EEEEEE'], startangle=90, 
               counterclock=False, wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{coverage:.0f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_title(c.company_name, y=-0.1)
        
    plt.tight_layout()
    plt.savefig(donut_chart_path, dpi=300, transparent=True)
    plt.close()
    
    # Jinja2 렌더링
    env = Environment(loader=FileSystemLoader(str(base_dir / "src" / "templates")))
    template = env.get_template("report.html")
    
    html_content = template.render(
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        best_company=best_company,
        companies=companies,
        bar_chart_path=f"file://{bar_chart_path.absolute()}",
        donut_chart_path=f"file://{donut_chart_path.absolute()}"
    )
    
    # PDF 변환
    pdf_path = output_dir / "final_report.pdf"
    HTML(string=html_content, base_url=str(base_dir)).write_pdf(pdf_path)
    
    return str(pdf_path)
