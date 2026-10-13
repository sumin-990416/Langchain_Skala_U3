"""명시적으로 등록한 State 모델만 체크포인트에서 복원한다."""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def checkpoint_serializer():
    allowed = [("src.schemas", "Company")]
    allowed.extend(("src.orchestration.schemas", name) for name in [
        "Perspective", "ResearchTask", "ResearchPlan", "Citation", "Claim",
        "SourceReference", "SourceRecord", "WorkerAnalysis", "WorkerResult",
        "PerspectiveSummary", "RetryTarget", "QualityCheck", "JudgeVerdict", "QualityVerdict",
    ])
    return JsonPlusSerializer(allowed_msgpack_modules=allowed, allowed_json_modules=allowed)
