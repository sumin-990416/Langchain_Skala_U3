"""결정과 사유는 외부 JSONL, State에는 trace_id와 최소 진행 정보만."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


_LOG_LOCK = Lock()


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"JSON으로 직렬화할 수 없음: {type(value).__name__}")


def write_json(output_dir: str, relative_path: str, data) -> str:
    root = Path(output_dir).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("산출물 경로는 실행 디렉터리 내부여야 합니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    # 병렬 Worker는 서로 다른 파일을 쓰고, 중단 시 반쪽 JSON이 남지 않게 교체한다.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        handle.write(body)
        temporary = handle.name
    os.replace(temporary, path)
    return str(path)


def log_decision(state, node: str, decision: str, reason: str, **details) -> None:
    root = Path(state["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": state["trace_id"],
        "node": node,
        "decision": decision,
        "reason": reason,
        **details,
    }
    with _LOG_LOCK, (root / "decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")


def advance(state, node: str) -> dict:
    statuses = dict(state.get("node_status", {}))
    statuses[node] = "completed"
    return {"step_count": state.get("step_count", 0) + 1, "node_status": statuses}
