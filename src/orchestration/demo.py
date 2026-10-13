"""API 없이 실행하는 명시적인 허구 Orchestrator-Workers 예제.

모든 회사, 문서, 수치는 교육용 fixture다. 실제 기업의 투자 근거로
사용할 수 없으며, live 모드의 검색 결과와 섞지 않는다.
"""

import hashlib

from src.schemas import Company

from .schemas import (
    Citation,
    Claim,
    Perspective,
    ResearchPlan,
    ResearchTask,
    SourceRecord,
    WorkerAnalysis,
)


def demo_companies() -> list[Company]:
    """실제 회사와 혼동하지 않도록 이름에도 DEMO를 표시한다."""
    return [
        Company(
            name="DEMO Battery A",
            description="[허구 데모] EV와 ESS용 배터리를 연구하는 가상 회사",
        ),
        Company(
            name="DEMO Battery B",
            description="[허구 데모] ESS용 배터리를 연구하는 가상 회사",
        ),
    ]


def demo_plan(companies: list[Company], target_domain: str) -> ResearchPlan:
    """관점은 유지하면서 회사 설명에 따라 적용 분야 작업을 분해한다.

    task_id는 실제 planner가 동일한 규칙으로 부여한다. EV와 ESS가 모두
    명시된 회사는 두 적용 작업을 받아 회사별 작업 수가 4개 또는 5개다.
    """
    tasks: list[ResearchTask] = []
    for company in companies:
        for perspective, question in (
            (Perspective.TECHNOLOGY, "기술의 시험 수준과 재현성·양산 제약은 무엇인가?"),
            (Perspective.MARKET, "고객 수요와 사업화 가능성 및 매출 검증 한계는 무엇인가?"),
            (Perspective.STAKEHOLDERS, "주요 이해관계자와 권리·공급망 관련 위험은 무엇인가?"),
        ):
            tasks.append(
                ResearchTask(
                    company_name=company.name,
                    perspective=perspective,
                    question=question,
                    queries=[f"{company.name} {question}"],
                    rationale="[허구 데모] 서로 다른 관점을 독립 작업으로 조사한다.",
                )
            )
        description = company.description.upper()
        domains = ["EV", "ESS"] if "EV" in description and "ESS" in description else [target_domain]
        for domain in domains:
            question = f"{domain} 적용 가능성과 안전·운영 검증 한계는 무엇인가?"
            tasks.append(
                ResearchTask(
                    company_name=company.name,
                    perspective=Perspective.APPLICATION,
                    question=question,
                    queries=[f"{company.name} {domain} 적용 안전 실증"],
                    rationale=f"[허구 데모] 회사 설명과 목표 도메인에 따라 {domain} 적용을 조사한다.",
                )
            )
    return ResearchPlan(tasks=tasks)


def _fixture_statements(task: ResearchTask) -> tuple[str, str, str]:
    company = task.company_name
    is_a = company == "DEMO Battery A"
    if task.perspective == Perspective.TECHNOLOGY:
        cycles, retention = (500, 90) if is_a else (300, 88)
        return (
            f"{company}의 허구 시험에서는 25°C에서 셀 {cycles}회의 충·방전 후 초기 용량의 {retention}%가 유지되었다.",
            "실제 양산 수율과 다른 연구기관의 재현성 자료는 제공되지 않았다.",
            "다른 온도 및 대량 생산 조건에서 같은 결과가 재현되는가?",
        )
    if task.perspective == Perspective.MARKET:
        interested = 6 if is_a else 4
        return (
            f"{company}의 가상 고객 인터뷰 10건 중 {interested}건에서 시제품 평가에 관심을 보였다.",
            "가상 인터뷰는 유료 주문이 아니며 실제 매출·계약·고객 유지율 자료는 없다.",
            "유료 고객 계약과 검증 가능한 실제 매출이 있는가?",
        )
    if task.perspective == Perspective.STAKEHOLDERS:
        return (
            f"{company}의 가상 협력 목록에는 시험기관과 소재 공급업체가 각각 한 곳 포함되어 있다.",
            "허구 협력 목록에는 독점 계약, 특허 권리 범위와 공급업체 재무 상태가 검증되지 않았다.",
            "핵심 권리와 공급 계약을 독립 자료로 확인할 수 있는가?",
        )
    question = task.question.upper()
    application = "EV" if "EV" in question and "ESS" not in question else "ESS" if "ESS" in question and "EV" not in question else "목표 도메인"
    return (
        f"{company}의 허구 {application} 실증에서는 제한된 실내 조건에서 배터리 모듈 운전이 가능했다.",
        f"실제 {application} 현장의 장기 안전성, 인증과 총소유비용 자료는 확인되지 않았다.",
        f"{application} 현장 조건에서 장기 안전·인증·비용 기준을 충족하는가?",
    )


def demo_worker(task: ResearchTask, attempt: int = 1) -> tuple[WorkerAnalysis, list[SourceRecord]]:
    """관점마다 독립된 두 허구 출처에서 정확한 인용을 반환한다."""
    # attempt는 live worker와 동일한 주입 인터페이스를 위해 받는다.
    del attempt
    support, limitation, follow_up = _fixture_statements(task)
    fixture_id = hashlib.sha256(
        f"{task.company_name}|{task.perspective.value}|{task.question}".encode()
    ).hexdigest()[:12]
    records = [
        SourceRecord(
            evidence_id=f"demo-{fixture_id}-support",
            source_id=f"fictional-lab-{fixture_id}",
            source_file=f"demo-fixtures/{fixture_id}-lab.txt",
            page=1,
            url="",
            source_group=f"fictional-lab-{fixture_id}",
            source_type="fictional_demo",
            content=f"[허구 데모 자료: 가상 시험·조사 기관] {support} 모든 내용은 교육용으로 만든 허구다.",
        ),
        SourceRecord(
            evidence_id=f"demo-{fixture_id}-limitation",
            source_id=f"fictional-review-{fixture_id}",
            source_file=f"demo-fixtures/{fixture_id}-review.txt",
            page=1,
            url="",
            source_group=f"fictional-review-{fixture_id}",
            source_type="fictional_demo",
            content=f"[허구 데모 자료: 독립 가상 검토 기관] {task.company_name}: {limitation} 실제 기업의 실적을 나타내지 않는다.",
        ),
    ]
    analysis = WorkerAnalysis(
        claims=[
            Claim(
                text=f"[허구 데모] {support}",
                kind="support",
                citations=[Citation(evidence_id=records[0].evidence_id, quote=support)],
            ),
            Claim(
                text=f"[허구 데모] {limitation}",
                kind="limitation",
                citations=[Citation(evidence_id=records[1].evidence_id, quote=limitation)],
            ),
        ],
        unknowns=[f"[허구 데모] {follow_up}"],
        follow_up_questions=[follow_up],
        source_limitations=[
            "회사·기관·문서·수치가 모두 교육용 허구이며 실제 검색 또는 투자 분석 결과가 아니다."
        ],
    )
    return analysis, records
