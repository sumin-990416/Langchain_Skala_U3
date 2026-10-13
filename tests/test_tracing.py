"""네트워크 없이 LangSmith 연결·전송·재개 계약을 검증한다."""

import io
import json
import os
import tempfile
import unittest
import uuid
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import tracing_context
from langsmith.utils import LangSmithNotFoundError

from src.graph import build_graph
from src.orchestration.tracing import TraceSession, tracing_enabled


class TracingConfigurationTest(unittest.TestCase):
    def test_demo_requires_explicit_opt_in_and_does_not_mutate_environment(self):
        configured = {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"}
        with patch.dict(os.environ, configured, clear=True):
            self.assertFalse(tracing_enabled("demo", False))
            self.assertTrue(tracing_enabled("demo", True))
            self.assertEqual(dict(os.environ), configured)

    def test_modern_false_overrides_legacy_true(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "true"}, clear=True):
            self.assertFalse(tracing_enabled("live", False))
            self.assertTrue(tracing_enabled("live", True))

    def test_legacy_environment_still_enables_live_tracing(self):
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True):
            self.assertTrue(tracing_enabled("live", False))

    def test_unconfigured_live_tracing_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(tracing_enabled("live", False))

    def test_missing_key_fails_before_constructing_client(self):
        with patch.dict(os.environ, {}, clear=True), patch("src.orchestration.tracing.Client") as client:
            with self.assertRaisesRegex(ValueError, "LANGSMITH_API_KEY"):
                TraceSession("unused", "demo", requested=True)
            client.assert_not_called()

    def test_current_credentials_and_explicit_project_are_passed_directly(self):
        configured = {
            "LANGSMITH_API_KEY": "dummy-current-key",
            "LANGCHAIN_API_KEY": "dummy-legacy-key",
            "LANGSMITH_ENDPOINT": "https://trace.example.invalid",
            "LANGCHAIN_ENDPOINT": "https://legacy.example.invalid",
            "LANGSMITH_PROJECT": "current-project",
            "LANGCHAIN_PROJECT": "legacy-project",
            "LANGSMITH_WORKSPACE_ID": str(uuid.uuid4()),
        }
        with patch.dict(os.environ, configured, clear=True), patch("src.orchestration.tracing.Client") as client, patch(
            "src.orchestration.tracing.LangChainTracer"
        ) as tracer:
            session = TraceSession("unused", "demo", requested=True, project_name="selected-project")
            kwargs = client.call_args.kwargs
            self.assertEqual(kwargs["api_key"], configured["LANGSMITH_API_KEY"])
            self.assertEqual(kwargs["api_url"], configured["LANGSMITH_ENDPOINT"])
            self.assertEqual(kwargs["workspace_id"], configured["LANGSMITH_WORKSPACE_ID"])
            tracer.assert_called_once_with(project_name="selected-project", client=client.return_value)
            self.assertEqual(session.graph_config(), {"run_id": session.run_id, "callbacks": [tracer.return_value]})

    def test_legacy_credentials_endpoint_and_project_are_supported(self):
        configured = {
            "LANGCHAIN_API_KEY": "dummy-legacy-key",
            "LANGCHAIN_ENDPOINT": "https://legacy.example.invalid",
            "LANGCHAIN_PROJECT": "legacy-project",
        }
        with patch.dict(os.environ, configured, clear=True), patch("src.orchestration.tracing.Client") as client, patch(
            "src.orchestration.tracing.LangChainTracer"
        ) as tracer:
            TraceSession("unused", "demo", requested=True)
            self.assertEqual(client.call_args.kwargs["api_key"], configured["LANGCHAIN_API_KEY"])
            self.assertEqual(client.call_args.kwargs["api_url"], configured["LANGCHAIN_ENDPOINT"])
            self.assertEqual(tracer.call_args.kwargs["project_name"], "legacy-project")


class TraceDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output_dir = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict(os.environ, {"LANGSMITH_API_KEY": "dummy-test-key"}, clear=True))
        self.client_cls = self.stack.enter_context(patch("src.orchestration.tracing.Client"))
        self.tracer_cls = self.stack.enter_context(patch("src.orchestration.tracing.LangChainTracer"))
        self.context_factory = self.stack.enter_context(patch("src.orchestration.tracing.tracing_context"))
        self.client = self.client_cls.return_value
        self.tracer = self.tracer_cls.return_value
        self.project = SimpleNamespace(id=uuid.uuid4())
        self.local_run = SimpleNamespace(start_time=datetime.now(timezone.utc))
        self.tracer.latest_run = self.local_run
        self.server_run = SimpleNamespace()
        self.client.read_project.return_value = self.project
        self.client.read_run.return_value = self.server_run
        self.client.get_run_url.return_value = "https://smith.example.invalid/verified-run"
        self.stdout = io.StringIO()

    def _session(self):
        return TraceSession(self.output_dir, "demo", requested=True, project_name="offline-project")

    def _record(self):
        return json.loads((Path(self.output_dir) / "langsmith-trace.json").read_text(encoding="utf-8"))

    def test_flush_and_server_verification_precede_success_link(self):
        calls = MagicMock()
        for method in ("flush", "read_project", "read_run", "get_run_url"):
            calls.attach_mock(getattr(self.client, method), method)
        with redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
        self.assertEqual([call[0] for call in calls.mock_calls], ["flush", "read_project", "read_run", "get_run_url"])
        self.client.flush.assert_called_once_with(timeout=15)
        self.client.read_project.assert_called_once_with(project_name="offline-project")
        self.assertEqual(self.client.read_run.call_args.args, (session.run_id,))
        self.assertEqual(self.client.read_run.call_args.kwargs["project_id"], self.project.id)
        self.assertEqual(self.client.read_run.call_args.kwargs["start_time"], self.local_run.start_time)
        self.context_factory.assert_called_once_with(enabled=True, project_name="offline-project", client=self.client)
        self.context_factory.return_value.__enter__.assert_called_once()
        self.context_factory.return_value.__exit__.assert_called_once_with(None, None, None)
        record = self._record()
        self.assertTrue(record["uploaded"])
        self.assertEqual(record["run_id"], str(session.run_id))
        self.assertIn(record["url"], self.stdout.getvalue())

    def test_delivery_failure_is_saved_without_private_exception_message(self):
        private = "private-response-with-dummy-key"
        self.client.read_run.side_effect = RuntimeError(private)
        with redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
        record = self._record()
        self.assertFalse(record["uploaded"])
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertNotIn("url", record)
        self.assertNotIn(private, json.dumps(record))
        self.assertNotIn(private, self.stdout.getvalue())
        self.client.get_run_url.assert_not_called()

    def test_initial_not_found_is_retried_until_stored_run_becomes_visible(self):
        self.client.read_run.side_effect = [LangSmithNotFoundError("private-first-response"), self.server_run]
        with patch("src.orchestration.tracing.time.sleep") as sleep, redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
        self.assertEqual(self.client.read_run.call_count, 2)
        self.assertEqual(self.client.read_project.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertTrue(self._record()["uploaded"])
        self.assertIn(self._record()["url"], self.stdout.getvalue())
        self.assertNotIn("private-first-response", self.stdout.getvalue())

    def test_not_found_retries_are_bounded_and_never_log_response_details(self):
        private = "private-persistent-not-found"
        self.client.read_run.side_effect = LangSmithNotFoundError(private)
        with patch("src.orchestration.tracing.time.sleep") as sleep, redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
        self.assertEqual(self.client.read_run.call_count, 5)
        self.assertEqual(self.client.read_project.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertTrue(all(call.args == (2,) for call in sleep.call_args_list))
        record = self._record()
        self.assertFalse(record["uploaded"])
        self.assertEqual(record["error_type"], "LangSmithNotFoundError")
        self.assertNotIn("url", record)
        self.assertNotIn(private, json.dumps(record))
        self.assertNotIn(private, self.stdout.getvalue())
        self.client.get_run_url.assert_not_called()

    def test_finished_resume_with_no_graph_run_never_queries_server(self):
        with redirect_stdout(self.stdout), self._session():
            pass
        self.client.flush.assert_not_called()
        self.client.read_project.assert_not_called()
        self.client.read_run.assert_not_called()
        self.client.get_run_url.assert_not_called()
        self.assertFalse((Path(self.output_dir) / "langsmith-trace.json").exists())

    def test_missing_local_root_cannot_claim_upload_success(self):
        self.tracer.latest_run = None
        with redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
        self.assertFalse(self._record()["uploaded"])
        self.assertEqual(self._record()["error_type"], "RuntimeError")
        self.client.read_run.assert_not_called()
        self.client.get_run_url.assert_not_called()

    def test_graph_exception_is_not_suppressed_and_traces_are_flushed(self):
        with self.assertRaisesRegex(ValueError, "graph failed"), redirect_stdout(self.stdout), self._session() as session:
            session.mark_started()
            raise ValueError("graph failed")
        self.client.flush.assert_called_once_with(timeout=15)
        args = self.context_factory.return_value.__exit__.call_args.args
        self.assertEqual(args[0], ValueError)
        self.assertEqual(str(args[1]), "graph failed")

    def test_default_demo_constructs_no_external_client_or_trace_record(self):
        with redirect_stdout(self.stdout), TraceSession(self.output_dir, "demo") as session:
            session.mark_started()
            self.assertEqual(session.graph_config()["callbacks"], [])
        self.client_cls.assert_not_called()
        self.tracer_cls.assert_not_called()
        self.context_factory.assert_called_once_with(enabled=False, project_name="default", client=None)
        self.assertFalse((Path(self.output_dir) / "langsmith-trace.json").exists())


class GraphCallbackContractTest(unittest.TestCase):
    def test_actual_sdk_callback_records_orchestrator_and_nine_workers_under_one_root(self):
        # The real SDK serializes and propagates callbacks; every API operation is mocked.
        client = MagicMock()
        tracer = LangChainTracer(project_name="offline-contract", client=client)
        run_id = uuid.uuid4()
        with tempfile.TemporaryDirectory() as output_dir, patch.dict(os.environ, {}, clear=True):
            initial = {
                "execution_mode": "demo", "selected_companies": [],
                "target_domain": "배터리 AI", "trace_id": "offline-callback",
                "output_dir": output_dir, "generator_model": "offline", "judge_model": "offline",
                "quality_mode": "deterministic", "generate_pdf": False,
                "max_quality_retries": 0, "max_worker_attempts": 2,
            }
            with tracing_context(enabled=True, project_name="offline-contract", client=client):
                build_graph().invoke(initial, config={
                    "run_id": run_id, "run_name": "battery-orchestrator-workers",
                    "callbacks": [tracer], "max_concurrency": 4,
                })
        submitted = [call.kwargs for call in client.create_run.call_args_list]
        roots = [run for run in submitted if run.get("parent_run_id") is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["id"], run_id)
        self.assertEqual(roots[0]["name"], "battery-orchestrator-workers")
        orchestrators = [run for run in submitted if run["name"] == "orchestrator"]
        workers = [run for run in submitted if run["name"] == "worker"]
        self.assertEqual(len(orchestrators), 1)
        self.assertEqual(len(workers), 9)
        self.assertTrue(all(run["parent_run_id"] == run_id for run in orchestrators + workers))
        self.assertTrue(all(run["trace_id"] == run_id for run in submitted))
        self.assertTrue(all(run["session_name"] == "offline-contract" for run in submitted))
        self.assertTrue(any(call.kwargs["run_id"] == run_id for call in client.update_run.call_args_list))
        self.assertEqual(tracer.latest_run.id, run_id)


if __name__ == "__main__":
    unittest.main()
