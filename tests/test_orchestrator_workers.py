"""네트워크·실제 API 없는 Orchestrator-Workers 회복성 검증."""

import json
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.orchestration.demo import demo_companies, demo_plan, demo_worker
from src.orchestration.planner import validate_plan
from src.orchestration.schemas import (
    Citation,
    Claim,
    Perspective,
    ResearchPlan,
    QualityCheck,
    QualityVerdict,
    RetryTarget,
    SourceRecord,
    WorkerAnalysis,
    WorkerResult,
    merge_by_task_id,
)
from src.orchestration.worker import validate_citations, worker_node


def _tasks():
    companies = demo_companies()
    return validate_plan(demo_plan(companies, "배터리 AI"), companies, 8)


def _worker_state(output_dir, task=None, mode="demo", attempts=2):
    return {
        "task": task or _tasks()[0],
        "trace_id": "offline-test",
        "output_dir": output_dir,
        "execution_mode": mode,
        "generator_model": "offline-model",
        "max_worker_attempts": attempts,
        "plan_revision": 0,
    }


class WorkerContractTest(unittest.TestCase):
    def test_company_context_dynamically_splits_four_or_five_tasks(self):
        tasks = _tasks()
        counts = Counter(task.company_name for task in tasks)
        self.assertEqual(counts, {"DEMO Battery A": 5, "DEMO Battery B": 4})
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        for company in demo_companies():
            self.assertEqual(
                {task.perspective for task in tasks if task.company_name == company.name},
                set(Perspective),
            )
        self.assertEqual([task.task_id for task in tasks], [task.task_id for task in _tasks()])

    def test_planner_rejects_missing_required_perspective(self):
        companies = demo_companies()
        tasks = [task for task in demo_plan(companies, "배터리 AI").tasks
                 if task.perspective != Perspective.MARKET]
        with self.assertRaises(ValueError):
            validate_plan(ResearchPlan(tasks=tasks), companies, 8)

    def test_each_fictional_worker_has_two_sources_and_exact_quotes(self):
        for task in _tasks():
            with self.subTest(task_id=task.task_id):
                analysis, records = demo_worker(task)
                self.assertEqual(len({record.source_group for record in records}), 2)
                self.assertEqual({claim.kind for claim in analysis.claims}, {"support", "limitation"})
                self.assertTrue(all(record.source_type == "fictional_demo" for record in records))
                validated = validate_citations(analysis, records)
                self.assertEqual(validated.claims, analysis.claims)

    def test_invalid_citation_removes_entire_claim_and_records_unknown(self):
        record = SourceRecord(
            evidence_id="valid", source_id="source", source_file="source.txt",
            page=1, source_group="group", content="Alpha  \n Beta is confirmed.",
        )
        analysis = WorkerAnalysis(claims=[
            Claim(text="valid claim", kind="context", citations=[Citation(evidence_id="valid", quote="Alpha Beta")]),
            Claim(text="forged quote", kind="support", citations=[Citation(evidence_id="valid", quote="fabricated")]),
            Claim(text="one valid plus one unknown", kind="support", citations=[
                Citation(evidence_id="valid", quote="confirmed"),
                Citation(evidence_id="unknown", quote="confirmed"),
            ]),
            Claim(text="no citation", kind="context", citations=[]),
            Claim(text="blank quote", kind="context", citations=[Citation(evidence_id="valid", quote=" ")]),
        ])
        result = validate_citations(analysis, [record])
        self.assertEqual([claim.text for claim in result.claims], ["valid claim"])
        self.assertEqual(len(result.unknowns), 4)
        self.assertTrue(all("인용 검증 실패" in unknown for unknown in result.unknowns))

    def test_demo_writes_raw_evidence_outside_state_without_model_calls(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state = _worker_state(output_dir)
            with patch("src.orchestration.worker.ChatOpenAI") as llm, patch(
                "src.orchestration.worker.retrieve_task_evidence"
            ) as retrieve:
                update = worker_node(state)
            llm.assert_not_called()
            retrieve.assert_not_called()
            self.assertEqual(set(update), {"worker_results"})
            result = update["worker_results"][state["task"].task_id]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.attempts, 1)
            self.assertTrue(all("content" not in source.model_dump() for source in result.sources))
            path = Path(result.evidence_path)
            self.assertEqual(path.parent.name, "evidence")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], state["task"].task_id)
            self.assertEqual(payload["execution_mode"], "demo")
            self.assertEqual(len(payload["sources"]), 2)
            self.assertTrue(all("content" in source for source in payload["sources"]))

    def test_invalid_claim_reason_survives_full_unknowns_list(self):
        task = _tasks()[0]
        _, records = demo_worker(task)
        analysis = WorkerAnalysis(
            claims=[Claim(text="forged", kind="support", citations=[Citation(evidence_id="missing", quote="invented")])],
            unknowns=[f"기존 미확인 {index}" for index in range(8)],
        )
        validated = validate_citations(analysis, records)
        self.assertEqual(validated.claims, [])
        self.assertEqual(len(validated.unknowns), 8)
        self.assertIn("인용 검증 실패", validated.unknowns[0])

    def test_empty_evidence_is_structured_without_llm_call(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state = _worker_state(output_dir, mode="live")
            with patch("src.orchestration.worker.retrieve_task_evidence", return_value=[]), patch(
                "src.orchestration.worker.ChatOpenAI"
            ) as llm:
                result = worker_node(state)["worker_results"][state["task"].task_id]
            llm.assert_not_called()
            self.assertEqual(result.status, "insufficient_evidence")
            self.assertEqual(result.attempts, 1)
            self.assertEqual(result.claims, [])
            self.assertTrue(result.unknowns)

    def test_live_analysis_disables_sdk_retry_and_uses_structured_schema(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state = _worker_state(output_dir, mode="live")
            analysis, records = demo_worker(state["task"])
            with patch("src.orchestration.worker.retrieve_task_evidence", return_value=records), patch(
                "src.orchestration.worker.ChatOpenAI"
            ) as llm:
                llm.return_value.with_structured_output.return_value.invoke.return_value = analysis
                result = worker_node(state)["worker_results"][state["task"].task_id]
            llm.assert_called_once_with(model="offline-model", temperature=0, timeout=45, max_retries=0)
            llm.return_value.with_structured_output.assert_called_once_with(
                WorkerAnalysis, method="function_calling"
            )
            self.assertEqual(result.status, "completed")

    def test_two_failures_are_bounded_and_only_error_type_is_saved(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state = _worker_state(output_dir, mode="live", attempts=2)
            secret = "sk-secret-do-not-log"
            with patch("src.orchestration.worker.retrieve_task_evidence", side_effect=RuntimeError(secret)) as retrieve:
                result = worker_node(state)["worker_results"][state["task"].task_id]
            self.assertEqual(retrieve.call_count, 2)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_type, "RuntimeError")
            self.assertEqual(result.attempts, 2)
            self.assertNotIn(secret, result.model_dump_json())
            self.assertNotIn(secret, (Path(output_dir) / "decisions.jsonl").read_text(encoding="utf-8"))

    def test_retry_repeats_retrieval_and_analysis_then_can_recover(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state = _worker_state(output_dir, mode="live", attempts=3)
            analysis, records = demo_worker(state["task"])
            with patch("src.orchestration.worker.retrieve_task_evidence", return_value=records) as retrieve, patch(
                "src.orchestration.worker._analyze", side_effect=[TimeoutError("private"), ValueError("private"), analysis]
            ) as analyze:
                result = worker_node(state)["worker_results"][state["task"].task_id]
            self.assertEqual(retrieve.call_count, 3)
            self.assertEqual(analyze.call_count, 3)
            self.assertEqual(result.attempts, 3)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.error_type, "")
            payload = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["attempt"], 3)

    def test_task_reducer_upserts_without_duplicate_results(self):
        left = {"task-a": "old", "task-b": "kept"}
        right = {"task-a": "new"}
        merged = merge_by_task_id(left, right)
        self.assertEqual(merged, {"task-a": "new", "task-b": "kept"})
        self.assertEqual(merge_by_task_id(merged, right), merged)
        self.assertEqual(left, {"task-a": "old", "task-b": "kept"})


class OrchestratorNodeContractTest(unittest.TestCase):
    def test_targeted_repair_only_dispatches_failed_company_and_perspective(self):
        from src.orchestration.planner import initialize_node, orchestrator_node

        with tempfile.TemporaryDirectory() as output_dir:
            state = initialize_node({"execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False})
            state.update(orchestrator_node(state))
            original_ids = set(state["task_registry"])
            passed = QualityCheck(passed=True, reason="confirmed")
            state["eval_result"] = QualityVerdict(
                groundedness=passed, neutrality=passed, bias_control=passed,
                coverage=QualityCheck(passed=False, reason="missing technology evidence"),
                passed=False, evaluator="deterministic",
                retry_targets=[RetryTarget(
                    company_name="DEMO Battery A", perspective=Perspective.TECHNOLOGY,
                    reason="기술 검증 자료를 다시 확인",
                )],
            )
            update = orchestrator_node(state)
            self.assertEqual(update["plan_revision"], 1)
            self.assertEqual(update["quality_retry_count"], 1)
            self.assertEqual(len(update["plan"]), 1)
            self.assertEqual(update["plan"][0].company_name, "DEMO Battery A")
            self.assertEqual(update["plan"][0].perspective, Perspective.TECHNOLOGY)
            self.assertIn("재작업", update["plan"][0].question)
            self.assertNotEqual(update["plan"][0].queries, state["task_registry"][update["plan"][0].task_id].queries)
            self.assertTrue(set(update["task_registry"]) <= original_ids)

    def test_demo_envelope_is_verified_by_quality_without_judge_api(self):
        from src.orchestration.planner import initialize_node, orchestrator_node
        from src.orchestration.quality import evaluate_report_node
        from src.orchestration.reporting import generate_report_node, synthesizer_node

        with tempfile.TemporaryDirectory() as output_dir:
            state = initialize_node({"execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False})
            state.update(orchestrator_node(state))
            for task in state["plan"]:
                worker_update = worker_node(_worker_state(output_dir, task))
                state["worker_results"] = merge_by_task_id(state["worker_results"], worker_update["worker_results"])
            state.update(synthesizer_node(state))
            state.update(generate_report_node(state))
            with patch("src.orchestration.quality._judge") as judge:
                verdict = evaluate_report_node(state)["eval_result"]
            judge.assert_not_called()
            self.assertTrue(verdict.passed, verdict.model_dump_json())
            self.assertTrue(verdict.groundedness.passed)
            self.assertTrue(verdict.neutrality.passed)
            self.assertTrue(verdict.bias_control.passed)
            self.assertTrue(verdict.coverage.passed)
            self.assertIn("## SUMMARY", state["report_draft"])
            self.assertIn("## REFERENCE", state["report_draft"])

    def test_each_quality_dimension_rejects_a_real_report_defect(self):
        from src.orchestration.planner import initialize_node, orchestrator_node
        from src.orchestration.quality import evaluate_report_node
        from src.orchestration.reporting import generate_report_node, synthesizer_node

        with tempfile.TemporaryDirectory() as output_dir:
            base = initialize_node({"execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False})
            base.update(orchestrator_node(base))
            for task in base["plan"]:
                base["worker_results"].update(worker_node(_worker_state(output_dir, task))["worker_results"])
            base.update(synthesizer_node(base))
            base.update(generate_report_node(base))
            first_id = base["plan"][0].task_id
            for dimension in ["groundedness", "neutrality", "bias_control", "coverage"]:
                with self.subTest(dimension=dimension):
                    state = copy.deepcopy(base)
                    result = state["worker_results"][first_id]
                    if dimension == "groundedness":
                        result.claims[0].citations[0].quote = "원문에 없는 위조 인용"
                    elif dimension == "neutrality":
                        state["report_draft"] += "\n이 기업에 투자검토 추천합니다."
                    elif dimension == "bias_control":
                        result.claims = [claim for claim in result.claims if claim.kind != "limitation"]
                        state.update(generate_report_node(state))
                    else:
                        del state["worker_results"][first_id]
                        state.update(synthesizer_node(state))
                        state.update(generate_report_node(state))
                    verdict = evaluate_report_node(state)["eval_result"]
                    self.assertFalse(getattr(verdict, dimension).passed)
                    self.assertFalse(verdict.passed)
                    self.assertTrue(verdict.retry_targets)

    def test_hybrid_judge_error_cannot_be_reported_as_quality_pass(self):
        from src.orchestration.planner import initialize_node, orchestrator_node
        from src.orchestration.quality import evaluate_report_node
        from src.orchestration.reporting import generate_report_node, synthesizer_node

        with tempfile.TemporaryDirectory() as output_dir:
            state = initialize_node({"execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False})
            state.update(orchestrator_node(state))
            for task in state["plan"]:
                state["worker_results"].update(worker_node(_worker_state(output_dir, task))["worker_results"])
            state.update(synthesizer_node(state))
            state.update(generate_report_node(state))
            state.update(execution_mode="live", quality_mode="hybrid")
            with patch("src.orchestration.quality._judge", side_effect=TimeoutError("secret-token")):
                verdict = evaluate_report_node(state)["eval_result"]
            self.assertFalse(verdict.passed)
            self.assertEqual(verdict.judge_error, "TimeoutError")
            self.assertNotIn("secret-token", verdict.model_dump_json())


def _failed_worker(state):
    task = state["task"]
    result = WorkerResult(
        task_id=task.task_id, company_name=task.company_name, perspective=task.perspective,
        status="failed", attempts=2, error_type="TimeoutError",
        unknowns=["모의 연결 오류로 확인하지 못한 항목"], follow_up_questions=[task.question],
    )
    return {"worker_results": {task.task_id: result}}


class OrchestratorGraphTest(unittest.TestCase):
    def test_demo_graph_completes_all_dynamic_workers(self):
        from src.graph import build_graph

        with tempfile.TemporaryDirectory() as output_dir:
            graph = build_graph()
            with patch("src.orchestration.worker.ChatOpenAI") as llm, patch(
                "src.orchestration.planner.ChatOpenAI"
            ) as planner_llm:
                final = graph.invoke({"execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False})
            llm.assert_not_called()
            planner_llm.assert_not_called()
            self.assertEqual(final["termination_reason"], "quality_passed")
            self.assertEqual(len(final["worker_results"]), 9)
            self.assertEqual(len(final["synthesis"]), 8)
            self.assertEqual(final["quality_retry_count"], 0)
            self.assertTrue(final["eval_result"].passed)
            self.assertTrue((Path(output_dir) / "report.md").is_file())

    def test_partial_worker_failure_does_not_stop_other_results_or_graph(self):
        from src.graph import build_graph

        calls = []

        def partial_worker(state):
            task = state["task"]
            calls.append(task.task_id)
            if task.company_name == "DEMO Battery A" and task.perspective == Perspective.TECHNOLOGY:
                return _failed_worker(state)
            return worker_node(state)

        with tempfile.TemporaryDirectory() as output_dir, patch("src.graph.worker_node", side_effect=partial_worker):
            graph = build_graph()
            final = graph.invoke({
                "execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False,
                "max_quality_retries": 1,
            })
            self.assertEqual(final["termination_reason"], "quality_retry_limit_reached")
            self.assertFalse(final["eval_result"].passed)
            self.assertEqual(final["quality_retry_count"], 1)
            self.assertEqual(len(final["worker_results"]), 9)
            self.assertEqual(Counter(result.status for result in final["worker_results"].values()), {"completed": 8, "failed": 1})
            failed_ids = [task_id for task_id, result in final["worker_results"].items() if result.status == "failed"]
            self.assertEqual(Counter(calls)[failed_ids[0]], 2)
            self.assertEqual(len(calls), 10)  # 최초 9개 + 대상 재작업 1개
            self.assertIn("작업 실패", final["report_draft"])
            self.assertIn("DEMO Battery B", final["report_draft"])
            self.assertIn("추가 확인 필요", final["report_draft"])

    def test_quality_repair_loop_is_bounded_and_results_do_not_grow(self):
        from src.graph import build_graph

        with tempfile.TemporaryDirectory() as output_dir, patch("src.graph.worker_node", side_effect=_failed_worker) as worker:
            graph = build_graph()
            final = graph.invoke({
                "execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False,
                "max_quality_retries": 2,
            }, {"recursion_limit": 80})
            self.assertEqual(worker.call_count, 27)
            self.assertEqual(final["quality_retry_count"], 2)
            self.assertEqual(final["plan_revision"], 2)
            self.assertEqual(len(final["task_registry"]), 9)
            self.assertEqual(len(final["worker_results"]), 9)
            self.assertEqual(final["termination_reason"], "quality_retry_limit_reached")
            self.assertLessEqual(final["step_count"], final["max_steps"])
            self.assertTrue((Path(output_dir) / "plan-r2.json").exists())
            self.assertFalse((Path(output_dir) / "plan-r3.json").exists())

    def test_step_budget_can_stop_repair_before_quality_retry_limit(self):
        from src.graph import build_graph

        with tempfile.TemporaryDirectory() as output_dir, patch("src.graph.worker_node", side_effect=_failed_worker) as worker:
            final = build_graph().invoke({
                "execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False,
                "max_quality_retries": 3, "max_steps": 8,
            })
            self.assertEqual(final["termination_reason"], "step_limit_reached")
            self.assertEqual(final["quality_retry_count"], 0)
            self.assertEqual(worker.call_count, 9)
            self.assertLessEqual(final["step_count"], final["max_steps"])

    def test_sqlite_checkpoint_resumes_in_new_graph_without_repeating_workers(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from src.graph import build_graph
        from src.orchestration.checkpoints import checkpoint_serializer

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "src.orchestration.worker.demo_worker", wraps=demo_worker
        ) as worker:
            database = str(Path(output_dir) / "checkpoints.sqlite")
            config = {"configurable": {"thread_id": "offline-resume"}, "recursion_limit": 80}
            with SqliteSaver.from_conn_string(database) as saver:
                saver.serde = checkpoint_serializer()
                first_graph = build_graph(checkpointer=saver, interrupt_before=["finalize"])
                paused = first_graph.invoke({
                    "execution_mode": "demo", "output_dir": output_dir, "generate_pdf": False,
                }, config)
                snapshot = first_graph.get_state(config)
                self.assertEqual(snapshot.next, ("finalize",))
                self.assertEqual(len(paused["worker_results"]), 9)
                self.assertTrue(paused["eval_result"].passed)
                self.assertFalse((Path(output_dir) / "report.md").exists())
            # DB 연결과 graph를 모두 새로 열어 메모리 saver에 의존하지 않음을 확인.
            with SqliteSaver.from_conn_string(database) as saver:
                saver.serde = checkpoint_serializer()
                resumed_graph = build_graph(checkpointer=saver)
                final = resumed_graph.invoke(None, config)
                self.assertEqual(resumed_graph.get_state(config).next, ())
            self.assertEqual(worker.call_count, 9)
            self.assertEqual(final["termination_reason"], "quality_passed")
            self.assertEqual(len(final["worker_results"]), 9)
            self.assertEqual(final["trace_id"], paused["trace_id"])
            self.assertTrue((Path(output_dir) / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
