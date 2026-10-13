"""품질 판정 후에만 실행별 최종 산출물을 저장한다."""

from pathlib import Path

from src.orchestration.observability import advance, log_decision, write_json


def finalize_node(state) -> dict:
    verdict = state["eval_result"]
    if verdict.passed:
        reason = "quality_passed"
    elif state.get("step_count", 0) + 5 >= state["max_steps"]:
        reason = "step_limit_reached"
    else:
        reason = "quality_retry_limit_reached"
    errors = list(state.get("report_errors", []))
    if not verdict.passed:
        errors.append("보고서 품질 미달: 제한된 재작업 후 추가 확인이 필요합니다.")
    root = Path(state["output_dir"])
    quality_notice = "통과" if verdict.passed else "미달 - 추가 확인 필요"
    report = state["report_draft"] + (
        f"\n## 품질 평가 결과\n\n- 판정: **{quality_notice}**\n"
        f"- 평가 방식: {verdict.evaluator}\n- 품질 재작업: {state['quality_retry_count']}회\n"
        f"- 종료 사유: {reason}\n- trace_id: `{state['trace_id']}`\n"
    )
    for field, label in [("groundedness", "Groundedness"), ("neutrality", "중립성"),
                         ("bias_control", "편향 통제"), ("coverage", "관점 커버리지")]:
        check = getattr(verdict, field)
        report += f"- {label}: {'충족' if check.passed else '미달'} / {check.reason}\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    write_json(state["output_dir"], "quality-final.json", verdict)
    if state.get("generate_pdf", True):
        try:
            from src.orchestration.pdf_report import generate_pdf
            generate_pdf(state, str(root / "report.pdf"))
        except Exception as exc:
            errors.append(f"PDF 생성 실패: {type(exc).__name__}")
    update = {**advance(state, "finalize"), "report_draft": report,
              "report_errors": list(dict.fromkeys(errors)), "termination_reason": reason}
    write_json(state["output_dir"], "final-state.json", {**state, **update})
    log_decision(state, "finalize", reason, "품질 판정과 미확인 내용을 포함한 보고서 저장",
                 quality_passed=verdict.passed, report_errors=update["report_errors"])
    return update
