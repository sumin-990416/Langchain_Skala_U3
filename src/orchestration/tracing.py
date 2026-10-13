"""명시적 LangSmith 연결, 종료 전 전송 완료 및 서버 저장 확인."""

import os
import time
import uuid
from contextlib import AbstractContextManager

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, tracing_context
from langsmith.utils import LangSmithNotFoundError
from urllib3.util import Retry

from src.orchestration.observability import write_json


def tracing_enabled(mode: str, requested: bool) -> bool:
    if requested:
        return True
    if mode == "demo":
        return False
    configured = os.environ.get("LANGSMITH_TRACING")
    if configured is None:
        configured = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    return configured.lower() in {"true", "1", "yes"}


class TraceSession(AbstractContextManager):
    def __init__(self, output_dir: str, mode: str, requested: bool = False, project_name: str | None = None):
        self.output_dir = output_dir
        self.enabled = tracing_enabled(mode, requested)
        self.project_name = (project_name or os.environ.get("LANGSMITH_PROJECT")
                             or os.environ.get("LANGCHAIN_PROJECT") or "default")
        self.run_id = uuid.uuid4()
        self.started = False
        self.client = None
        self.tracer = None
        if self.enabled:
            key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
            if not key:
                raise ValueError("LangSmith tracing에는 LANGSMITH_API_KEY가 필요합니다.")
            endpoint = (os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT")
                        or "https://api.smith.langchain.com")
            # 환경 변수 캐시에 의존하지 않고 .env 로드 후 값을 직접 전달한다.
            self.client = Client(
                api_key=key, api_url=endpoint,
                workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID"),
                timeout_ms=(5000, 10000), retry_config=Retry(total=1, backoff_factor=0.2),
            )
            self.tracer = LangChainTracer(project_name=self.project_name, client=self.client)

    def __enter__(self):
        self.context = tracing_context(enabled=self.enabled, project_name=self.project_name, client=self.client)
        self.context.__enter__()
        if self.enabled:
            print(f"LangSmith tracing: 활성 / 프로젝트: {self.project_name}")
        else:
            print("LangSmith tracing: 비활성 (데모 전송은 --demo --trace)")
        return self

    def graph_config(self) -> dict:
        return {"run_id": self.run_id, "callbacks": [self.tracer] if self.tracer else []}

    def mark_started(self):
        self.started = True

    def _read_saved_run(self, local_run):
        # flush는 SDK 큐를 비우지만 서버 조회 반영은 잠시 늦을 수 있다.
        # 실제 첫 전송에서 404 후 조회 성공을 확인해 제한된 조회 재시도를 둔다.
        for attempt in range(5):
            try:
                project = self.client.read_project(project_name=self.project_name)
                run = self.client.read_run(
                    self.run_id, project_id=project.id, start_time=local_run.start_time,
                )
                return project, run
            except LangSmithNotFoundError:
                if attempt == 4:
                    raise
                time.sleep(2)

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.__exit__(exc_type, exc_value, traceback)
        if not self.enabled or not self.started:
            return False
        record = {"run_id": str(self.run_id), "project_name": self.project_name, "uploaded": False}
        try:
            # explicit Client는 전역 wait_for_all_tracers() 대신 직접 flush한다.
            self.client.flush(timeout=15)
            local_run = self.tracer.latest_run
            if local_run is None:
                raise RuntimeError("GraphTraceMissing")
            project, run = self._read_saved_run(local_run)
            record.update(uploaded=True)
            # 서버에 run이 저장된 사실을 먼저 확인한 뒤 링크를 표시한다.
            record["url"] = self.client.get_run_url(run=run, project_id=project.id)
            print(f"LangSmith trace 저장 확인: {record['url']}")
        except Exception as error:
            record["error_type"] = type(error).__name__
            print(f"LangSmith trace 전송·조회 확인 실패: {record['error_type']}")
        write_json(self.output_dir, "langsmith-trace.json", record)
        return False
