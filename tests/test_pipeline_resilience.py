import unittest
from pathlib import Path

from src.agents.profile import get_all_profiles, profile_agent_node
from src.nodes.core import compare_companies, synthesize_company
from src.report_generator import generate_pdf_report
from src.schemas import (
    Company,
    Domain,
    FinalDecision,
    ProfileResult,
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
            "profile_name": "broken-profile",
            "domain_scores": DOMAIN_SCORES,
            "validated_evidence": [],
            "current_company": Company(name="연결 오류 기업"),
        })["profile_results"][0]
        self.assertEqual(result.result_category, ResultCategory.EXECUTION_ERROR)
        self.assertIsNone(result.weighted_score)
        self.assertTrue(result.error_message)

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

    def test_five_page_pdf_supports_hold_and_error_categories(self):
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
        generated = Path(generate_pdf_report(comparison, output_path))
        self.assertTrue(generated.exists())
        self.assertGreater(generated.stat().st_size, 0)

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
