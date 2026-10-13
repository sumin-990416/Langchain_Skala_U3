import unittest
from unittest.mock import patch
from pathlib import Path

from pypdf import PdfReader
from src.agents.profile import get_all_profiles, plan_profile_tasks, profile_agent_node
from src.graph import (
    PROFILE_WORKER_NODES,
    build_graph,
    build_graph_run_config,
    route_report_quality,
    route_to_profiles,
)
from src.nodes.report_quality import (
    _rule_checks,
    evaluate_report_quality,
    plan_report_review,
    route_to_report_reviewers,
    synthesize_report_corrections,
)
from src.nodes.core import compare_companies, save_company_result, synthesize_company
from src.report_generator import (
    build_report_summary,
    collect_reference_entries,
    generate_pdf_report,
)
from src.schemas import (
    Company,
    ComparisonResult,
    Domain,
    Evidence,
    FinalDecision,
    ProfileResult,
    ReportCorrection,
    ReportEvidence,
    ReportQualityFinding,
    ReportQualityVerdict,
    ResultCategory,
)


DOMAIN_SCORES = {domain: 70.0 for domain in Domain}


def make_profile(
    name: str,
    score: float | None,
    category: ResultCategory,
    coverage: float | None,
    decision: FinalDecision = FinalDecision.CONDITIONAL,
    error: str = "",
) -> ProfileResult:
    return ProfileResult(
        profile_name=name,
        weighted_score=score,
        decision=decision,
        evidence_coverage=coverage,
        result_category=category,
        recommendation="입력 근거 범위에서 평가함",
        unknown_items=[] if category == ResultCategory.COMPLETE else ["반복매출 미확인"],
        due_diligence_questions=["ARR 원자료를 제출할 것"],
        supporting_evidence_ids=["A-01"] if score is not None else [],
        error_message=error,
    )


class PipelineResilienceTest(unittest.TestCase):
    def test_graph_run_config_has_searchable_run_name_and_trace_correlation(self):
        config = build_graph_run_config("trace-for-test")

        self.assertEqual(config["run_name"], "battery-startup-investment-report")
        self.assertIn("langgraph", config["tags"])
        self.assertEqual(config["metadata"]["trace_id"], "trace-for-test")

    def test_report_summary_is_a_concise_conclusion(self):
        profile = make_profile(
            "기술중심형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0,
            FinalDecision.HOLD_RAG,
        )
        company = synthesize_company({
            "profile_results": [profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 0.0,
            "current_company": Company(name="기업A"),
        })["company_result"]

        summary = build_report_summary([company])

        self.assertIn("우선 투자 대상을 선정하지 않았으며", summary)
        self.assertIn("기업A", summary)
        self.assertLess(len(summary), 300)

    def test_references_resolve_only_cited_card_and_page_alias(self):
        raw_dir = Path(__file__).parent.parent / "data" / "raw"

        references = collect_reference_entries(
            ["A-01", "A01", "E-03", "BESS_RAG_Source_Pack:p16"], raw_dir=raw_dir
        )
        entries = [entry for group in references.values() for entry in group]

        self.assertEqual(len(entries), 2)
        self.assertTrue(any("Battery Analytics Platform" in entry["citation"] for entry in entries))
        self.assertTrue(any("ELECTRA AI" in entry["citation"] for entry in entries))
        self.assertEqual(
            {
                source_id
                for entry in entries
                for source_id in entry["requested_ids"]
            },
            {"A-01", "A01", "E-03", "BESS_RAG_Source_Pack:p16"},
        )

    def test_orchestrator_creates_structured_profile_tasks_from_configuration(self):
        plan = plan_profile_tasks({"current_company": Company(name="계획 대상 기업")})
        tasks = plan["profile_task_plan"]

        self.assertEqual([task["profile_name"] for task in tasks], get_all_profiles())
        self.assertTrue(all(task["task_id"].startswith("계획 대상 기업:") for task in tasks))
        self.assertTrue(all(task["focus"] for task in tasks))
        self.assertEqual(plan["profile_results"], [])

    def test_worker_fanout_uses_only_tasks_in_the_orchestrator_plan(self):
        task = {
            "task_id": "기업A:balanced",
            "profile_name": "balanced",
            "focus": "균형 평가",
        }
        sends = route_to_profiles({
            "profile_task_plan": [task],
            "domain_scores": DOMAIN_SCORES,
            "validated_evidence": [],
            "current_company": Company(name="기업A"),
        })

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "worker_균형형")
        self.assertEqual(sends[0].arg["task_id"], task["task_id"])
        self.assertEqual(sends[0].arg["profile_name"], task["profile_name"])

    def test_langgraph_trace_exposes_orchestrator_and_named_profile_workers(self):
        workflow = build_graph()
        nodes = set(workflow.nodes)

        self.assertIn("orchestrator_plan_workers", nodes)
        self.assertIn("synthesizer_aggregate_company", nodes)
        self.assertEqual(
            {node for node in nodes if node.startswith("worker_")},
            {
                "worker_기술중심형",
                "worker_안정형",
                "worker_성장형",
                "worker_균형형",
                "worker_사업성중심형",
            },
        )
        self.assertEqual(len(set(PROFILE_WORKER_NODES.values())), 5)

    def test_hybrid_report_rules_check_references_neutrality_and_source_balance(self):
        profile = make_profile("기술중심형", 75, ResultCategory.COMPLETE, 90).model_copy(
            update={
                "supporting_evidence_ids": ["SRC-1", "SRC-2"],
                "contrary_evidence_ids": [],
            }
        )
        company = synthesize_company({
            "profile_results": [profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        comparison = ComparisonResult(companies=[company])
        evidence = [
            ReportEvidence(
                company_name="기업A", item_id="tech_01", source_id="SRC-1",
                source_type="independent_study", evidence_grade="A",
                content="독립 검증 결과를 요약한 발췌문",
            ),
            ReportEvidence(
                company_name="기업A", item_id="tech_02", source_id="SRC-2",
                source_type="company_filing", evidence_grade="B",
                content="회사가 공개한 제품 검증 정보",
            ),
        ]
        report = (
            "분석 의견 [SRC-1] [SRC-2]\n\n## REFERENCE\n"
            "- [SRC-1]\n- [SRC-2]\n"
        )

        checks, findings = _rule_checks(report, comparison, evidence)

        self.assertEqual(
            checks,
            {"groundedness": True, "neutrality": True, "source_bias": True},
        )
        self.assertEqual(findings, [])

        biased_profile = profile.model_copy(update={"supporting_evidence_ids": ["SRC-1"]})
        biased_company = synthesize_company({
            "profile_results": [biased_profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        failed_checks, failed_findings = _rule_checks(
            "가장 최고의 기술 [UNKNOWN]\n## REFERENCE\n- [UNKNOWN]",
            ComparisonResult(companies=[biased_company]),
            evidence,
        )
        self.assertEqual(
            failed_checks,
            {"groundedness": False, "neutrality": False, "source_bias": False},
        )
        self.assertEqual(
            {finding.criterion for finding in failed_findings},
            {"groundedness", "neutrality", "source_bias"},
        )

    def test_report_review_planner_selects_only_affected_profile(self):
        profiles = [
            make_profile("기술중심형", 75, ResultCategory.COMPLETE, 90),
            make_profile("안정형", 70, ResultCategory.COMPLETE, 90),
        ]
        company = synthesize_company({
            "profile_results": profiles,
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        verdict = ReportQualityVerdict(
            groundedness_pass=False,
            neutrality_pass=True,
            source_bias_pass=True,
            findings=[
                ReportQualityFinding(
                    criterion="groundedness",
                    reason="기술 평가 주장에 근거 연결이 필요합니다.",
                    company_names=["기업A"],
                    profile_names=["기술중심형"],
                )
            ],
        )

        plan = plan_report_review({
            "final_comparison": ComparisonResult(companies=[company]),
            "report_quality_verdict": verdict,
        })["report_review_plan"]

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["company_name"], "기업A")
        self.assertEqual(plan[0]["profile_name"], "기술중심형")
        self.assertEqual(plan[0]["criteria"], ["groundedness"])
        self.assertTrue(plan[0]["feedback"])
        self.assertTrue(plan[0]["revise_company_summary"])

        sends = route_to_report_reviewers({
            "final_comparison": ComparisonResult(companies=[company]),
            "report_review_plan": plan,
            "report_evidence": [
                ReportEvidence(
                    company_name="기업A", item_id="tech_01", source_id="A-01",
                    source_type="filing", evidence_grade="B", content="관련 근거 발췌",
                )
            ],
            "trace_id": "test-trace",
        })
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].arg["task"]["profile_name"], "기술중심형")
        self.assertEqual(sends[0].arg["trace_id"], "test-trace")

    def test_report_corrections_are_synthesized_into_both_report_inputs(self):
        profile = make_profile("기술중심형", 75, ResultCategory.COMPLETE, 90)
        company = synthesize_company({
            "profile_results": [profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        comparison = ComparisonResult(companies=[company])
        correction = ReportCorrection(
            company_name="기업A",
            profile_name="기술중심형",
            recommendation="확인된 근거와 한계를 함께 고려한 수정 의견",
            company_summary="공개된 자료 범위 내에서 검토하며 추가 확인이 필요하다.",
        )

        updated = synthesize_report_corrections({
            "final_comparison": comparison,
            "report_corrections": [correction],
        })["final_comparison"]

        self.assertEqual(
            updated.companies[0].profile_results[0].recommendation,
            correction.recommendation,
        )
        self.assertEqual(updated.companies[0].summary_reason, correction.company_summary)

    def test_quality_loop_is_bounded_and_routes_targeted_rework(self):
        failed = ReportQualityVerdict(
            groundedness_pass=False,
            neutrality_pass=True,
            source_bias_pass=True,
        )

        self.assertEqual(route_report_quality({
            "report_quality_verdict": failed,
            "quality_retry_count": 0,
            "quality_max_retries": 1,
        }), "plan_report_review")
        self.assertEqual(route_report_quality({
            "report_quality_verdict": failed,
            "quality_retry_count": 1,
            "quality_max_retries": 1,
        }), "record_quality_failure")

    def test_report_evidence_payload_is_bounded(self):
        profile = make_profile(
            "기술중심형", 75, ResultCategory.COMPLETE, 90
        ).model_copy(update={"supporting_evidence_ids": ["SRC-99"]})
        company = synthesize_company({
            "profile_results": [profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        evidence = [
            Evidence(
                item_id=f"item-{index}",
                content="x" * 1000,
                source_type="filing",
                source_id=f"SRC-{index}",
                evidence_grade="B",
                is_direct=True,
            )
            for index in range(100)
        ]

        saved = save_company_result({
            "current_index": 0,
            "company_result": company,
            "validated_evidence": evidence,
        })

        self.assertEqual(len(saved["report_evidence"]), 60)
        self.assertTrue(all(len(item.content) <= 800 for item in saved["report_evidence"]))
        self.assertIn("SRC-99", {item.source_id for item in saved["report_evidence"]})

    def test_quality_judge_failure_is_not_treated_as_pass(self):
        profile = make_profile("기술중심형", 75, ResultCategory.COMPLETE, 90)
        company = synthesize_company({
            "profile_results": [profile],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 90.0,
            "current_company": Company(name="기업A"),
        })["company_result"]
        with patch("src.nodes.report_quality.ChatOpenAI", side_effect=RuntimeError("offline")):
            result = evaluate_report_quality({
                "report_draft": "## REFERENCE\n- [A-01]",
                "final_comparison": ComparisonResult(companies=[company]),
                "report_evidence": [],
                "report_errors": [],
                "trace_id": "test-trace",
            })

        self.assertTrue(result["report_quality_verdict"].judge_error)
        self.assertFalse(result["report_quality_verdict"].groundedness_pass)
        self.assertTrue(result["report_errors"])

    def test_all_agents_return_hold_instead_of_stopping_without_evidence(self):
        state = {
            "domain_scores": DOMAIN_SCORES,
            "validated_evidence": [],
            "current_company": Company(name="자료 부족 기업"),
        }
        for profile_id in get_all_profiles():
            with self.subTest(profile_id=profile_id):
                result = profile_agent_node({**state, "profile_name": profile_id})["profile_results"][0]
                self.assertIsNone(result.weighted_score)
                self.assertEqual(result.decision, FinalDecision.HOLD_RAG)
                self.assertIn(
                    result.result_category,
                    {ResultCategory.INSUFFICIENT_EVIDENCE, ResultCategory.PARTIAL_EVIDENCE},
                )

    def test_agent_error_is_structured_and_nonblocking(self):
        result = profile_agent_node({
            "task_id": "기업A:broken-profile",
            "profile_name": "broken-profile",
            "domain_scores": DOMAIN_SCORES,
            "validated_evidence": [],
            "current_company": Company(name="연결 오류 기업"),
        })["profile_results"][0]
        self.assertEqual(result.result_category, ResultCategory.EXECUTION_ERROR)
        self.assertIsNone(result.weighted_score)
        self.assertTrue(result.error_message)
        self.assertEqual(result.task_id, "기업A:broken-profile")

        company_result = synthesize_company({
            "profile_results": [result],
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 0.0,
            "current_company": Company(name="연결 오류 기업"),
        })["company_result"]
        self.assertEqual(company_result.result_category, ResultCategory.EXECUTION_ERROR)
        self.assertIn("평가 오류", company_result.summary_reason)

    def test_partial_results_are_compared_without_none_format_errors(self):
        profiles = [
            make_profile("기술중심형", 78, ResultCategory.COMPLETE, 90),
            make_profile("안정형", None, ResultCategory.PARTIAL_EVIDENCE, 60, FinalDecision.HOLD_RAG),
        ]
        company = synthesize_company({
            "profile_results": profiles,
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 0.0,
            "current_company": Company(name="부분 근거 기업"),
        })["company_result"]
        self.assertEqual(company.result_category, ResultCategory.PARTIAL_EVIDENCE)
        self.assertEqual(company.final_decision, FinalDecision.HOLD_RAG.value)
        comparison = compare_companies({"company_results": [company]})["final_comparison"]
        lines = comparison.comparison_table.splitlines()
        header_cells = [cell for cell in lines[0].split("|") if cell.strip()]
        row_cells = [cell for cell in lines[2].split("|") if cell.strip()]
        self.assertEqual(len(header_cells), len(row_cells))
        self.assertIn("일부 근거 부족", lines[2])

    def test_pdf_respects_page_limit_and_supports_hold_and_error_categories(self):
        company_specs = [
            ("ACCURE", [
                make_profile("기술중심형", 79, ResultCategory.COMPLETE, 90),
                make_profile("안정형", None, ResultCategory.PARTIAL_EVIDENCE, 62, FinalDecision.HOLD_RAG),
                make_profile("성장형", 76, ResultCategory.COMPLETE, 90),
                make_profile("균형형", 74, ResultCategory.COMPLETE, 100),
                make_profile("사업성중심형", 77, ResultCategory.COMPLETE, 88),
            ]),
            ("volytica diagnostics", [
                make_profile("기술중심형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG),
                make_profile("안정형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG),
                make_profile("성장형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG),
                make_profile("균형형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG),
                make_profile("사업성중심형", None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG),
            ]),
            ("Electra Vehicles", [
                make_profile("기술중심형", 65, ResultCategory.PARTIAL_EVIDENCE, 82),
                make_profile("안정형", 61, ResultCategory.PARTIAL_EVIDENCE, 80),
                make_profile(
                    "성장형", None, ResultCategory.EXECUTION_ERROR, None,
                    FinalDecision.HOLD_RAG, "APIConnectionError: 모의 연결 오류",
                ),
                make_profile("균형형", 64, ResultCategory.COMPLETE, 100),
                make_profile("사업성중심형", 63, ResultCategory.PARTIAL_EVIDENCE, 84),
            ]),
        ]
        companies = [
            synthesize_company({
                "profile_results": profiles,
                "domain_scores": DOMAIN_SCORES,
                "evidence_coverage": 0.0,
                "current_company": Company(name=name),
            })["company_result"]
            for name, profiles in company_specs
        ]
        comparison = compare_companies({"company_results": companies})["final_comparison"]
        output_path = Path("tmp/pdfs/test-hold-output.pdf").resolve()
        generated = Path(
            generate_pdf_report(
                comparison,
                output_path,
                report_summary="테스트 요약: 공개 근거만으로 판단을 확정하기 어렵다.",
            )
        )
        self.assertTrue(generated.exists())
        self.assertGreater(generated.stat().st_size, 0)
        pages = PdfReader(str(generated)).pages
        self.assertLessEqual(len(pages), 10)
        self.assertIn("SUMMARY", pages[0].extract_text() or "")
        self.assertIn("테스트 요약", pages[0].extract_text() or "")
        self.assertIn("REFERENCE", pages[-1].extract_text() or "")

    def test_pdf_is_created_when_every_profile_is_on_hold(self):
        profiles = [
            make_profile(name, None, ResultCategory.INSUFFICIENT_EVIDENCE, 0, FinalDecision.HOLD_RAG)
            for name in ["기술중심형", "안정형", "성장형", "균형형", "사업성중심형"]
        ]
        company = synthesize_company({
            "profile_results": profiles,
            "domain_scores": DOMAIN_SCORES,
            "evidence_coverage": 0.0,
            "current_company": Company(name="전체 판단 유보 기업"),
        })["company_result"]
        comparison = compare_companies({"company_results": [company]})["final_comparison"]
        output_path = Path("tmp/pdfs/test-all-hold-output.pdf").resolve()
        generated = Path(generate_pdf_report(comparison, output_path))
        self.assertTrue(generated.exists())
        self.assertGreater(generated.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
