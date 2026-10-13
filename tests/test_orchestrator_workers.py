import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.graph import assign_research_workers, graph, route_evidence_check, route_report_validation
from src.nodes.core import (
    orchestrate_research, research_worker, synthesize_research, validate_plan,
    validate_report,
)
from src.schemas import (
    Company, CompanyResult, ComparisonResult,
    Domain,
    Evidence,
    ResearchResult,
    ResearchTask,
    reset_or_add_research,
)


class FailingPlanner:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        raise RuntimeError("planner unavailable")


class OrchestratorWorkersTests(unittest.TestCase):
    def test_orchestrator_falls_back_to_five_independent_domain_tasks(self):
        state = {
            "current_company": Company(name="ACCURE", description="battery analytics")
        }
        with patch("src.nodes.core.ChatOpenAI", FailingPlanner):
            update = orchestrate_research(state)

        self.assertEqual(len(update["research_tasks"]), 5)
        self.assertEqual(
            {task.domain for task in update["research_tasks"]}, set(Domain)
        )
        self.assertEqual(update["research_results"], [])

    def test_assign_workers_fans_out_one_send_per_task(self):
        tasks = [
            ResearchTask(task_id="a", domain=Domain.TECHNOLOGY, query="기술"),
            ResearchTask(task_id="b", domain=Domain.MARKET, query="시장"),
        ]
        sends = assign_research_workers(
            {"current_company": Company(name="A"), "research_tasks": tasks}
        )
        self.assertEqual(len(sends), 2)
        self.assertTrue(all(send.node == "research_worker" for send in sends))

    def test_worker_returns_reducer_compatible_result(self):
        task = ResearchTask(
            task_id="tech", domain=Domain.TECHNOLOGY, query="모델 성능"
        )
        docs = [(Document(page_content="근거", metadata={"source_id": "D1"}), 0.1)]
        with patch("src.nodes.core.retrieve_top_k", return_value=docs):
            update = research_worker(
                {"current_company": Company(name="A"), "research_task": task}
            )

        self.assertEqual(update["research_results"][0].task, task)
        self.assertEqual(update["research_results"][0].evidence[0].source_id, "D1")

    def test_synthesizer_deduplicates_worker_evidence(self):
        task = ResearchTask(task_id="tech", domain=Domain.TECHNOLOGY, query="기술")
        evidence = Evidence(
            item_id="tech",
            content="같은 근거",
            source_type="document",
            source_id="D1",
            evidence_grade="B",
            is_direct=True,
        )
        state = {
            "research_results": [
                ResearchResult(task=task, evidence=[evidence]),
                ResearchResult(task=task, evidence=[evidence]),
            ]
        }
        update = synthesize_research(state)
        self.assertEqual(update["retrieved_evidence"], [evidence])

    def test_reducer_merges_parallel_results_and_supports_reset(self):
        task = ResearchTask(task_id="tech", domain=Domain.TECHNOLOGY, query="기술")
        result = ResearchResult(task=task, evidence=[])
        self.assertEqual(reset_or_add_research([result], [result]), [result, result])
        self.assertEqual(reset_or_add_research([result], []), [])

    def test_compiled_graph_contains_orchestrator_worker_synthesis(self):
        nodes = set(graph.get_graph().nodes)
        self.assertTrue(
            {"orchestrator", "validate_plan", "research_worker",
             "synthesize_research", "validate_report", "save_report"} <= nodes
        )

    def test_invalid_plan_falls_back_before_fanout(self):
        one_task = ResearchTask(task_id="a", domain=Domain.TECHNOLOGY, query="기술")
        update = validate_plan({"research_tasks": [one_task]})
        self.assertEqual(len(update["research_tasks"]), 5)
        self.assertIn("계획 검증 실패", update["plan_error"])

    def test_valid_three_task_plan_keeps_dynamic_fanout(self):
        tasks = [
            ResearchTask(task_id="tech_team", domain=Domain.TECHNOLOGY,
                         domains=[Domain.TECHNOLOGY, Domain.TEAM],
                         query="기술과 팀", objective="기술과 팀 근거",
                         success_criteria=["출처 1건"]),
            ResearchTask(task_id="market_finance", domain=Domain.MARKET,
                         domains=[Domain.MARKET, Domain.FINANCE],
                         query="시장과 재무", objective="시장과 재무 근거",
                         success_criteria=["출처 1건"]),
            ResearchTask(task_id="risk", domain=Domain.RISK,
                         domains=[Domain.RISK], query="안전", objective="위험 근거",
                         success_criteria=["출처 1건"]),
        ]
        state = {"current_company": Company(name="A"), "research_tasks": tasks}
        self.assertEqual(validate_plan(state), {"plan_error": ""})
        self.assertEqual(len(assign_research_workers(state)), 3)

    def test_worker_failure_is_recorded_without_raising(self):
        task = ResearchTask(task_id="tech", domain=Domain.TECHNOLOGY, query="기술")
        with patch("src.nodes.core.retrieve_top_k", side_effect=RuntimeError("offline")) as search:
            update = research_worker({"current_company": Company(name="A"),
                                      "research_task": task, "max_worker_attempts": 2})
        self.assertEqual(search.call_count, 2)
        self.assertEqual(update["research_results"][0].status, "FAILED")
        self.assertEqual(update["errors"][0].task_id, "tech")

    def test_partial_failure_is_visible_to_synthesis(self):
        good = ResearchTask(task_id="tech", domain=Domain.TECHNOLOGY, query="기술")
        bad = ResearchTask(task_id="risk", domain=Domain.RISK, query="위험")
        evidence = Evidence(item_id="tech", content="근거", source_type="document",
                            source_id="D1", evidence_grade="B", is_direct=True)
        update = synthesize_research({
            "research_tasks": [good, bad],
            "research_results": [
                ResearchResult(task=bad, status="FAILED", error="offline"),
                ResearchResult(task=good, evidence=[evidence]),
            ],
        })
        self.assertEqual(update["retrieved_evidence"], [evidence])
        self.assertEqual(update["research_synthesis"].failed_task_ids, ["risk"])
        self.assertFalse(update["research_synthesis"].is_sufficient)

    def test_evidence_loop_stops_at_limit(self):
        self.assertEqual(route_evidence_check({"evidence_retry_count": 0,
                                               "max_evidence_retries": 2}), "rewrite_query")
        self.assertEqual(route_evidence_check({"evidence_retry_count": 2,
                                               "max_evidence_retries": 2}), "record_unknown")

    def test_report_quality_fails_when_claims_lack_sources(self):
        company = CompanyResult(
            company_name="A", domain_scores={}, evidence_coverage=0,
            profile_results=[], average_score=0, score_std_dev=0,
            final_decision="판단 유보", summary_reason="자료 부족",
        )
        update = validate_report({"report_draft": "# 보고서",
                                  "final_comparison": ComparisonResult(companies=[company])})
        self.assertFalse(update["report_evaluation"].passed)
        self.assertFalse(update["report_evaluation"].groundedness)
        self.assertEqual(route_report_validation({"report_evaluation": update["report_evaluation"],
                                                  "report_retry_count": 2,
                                                  "max_report_retries": 2}), "save_report")

    def test_report_quality_passes_with_traceable_diverse_sources(self):
        company = CompanyResult(
            company_name="A", domain_scores={}, evidence_coverage=100,
            profile_results=[], average_score=0, score_std_dev=0,
            final_decision="조건부 검토", summary_reason="추가 실사 필요",
            evidence_sources=["doc-1:p1:c0", "doc-2:p2:c1"],
            source_documents=["doc-1.pdf", "doc-2.pdf"],
        )
        report = "A의 근거는 [doc-1:p1:c0], [doc-2:p2:c1]입니다."
        update = validate_report({"report_draft": report,
                                  "final_comparison": ComparisonResult(companies=[company])})
        self.assertTrue(update["report_evaluation"].passed)
        self.assertEqual(route_report_validation({"report_evaluation": update["report_evaluation"],
                                                  "report_retry_count": 1,
                                                  "max_report_retries": 2}), "save_report")


if __name__ == "__main__":
    unittest.main()
