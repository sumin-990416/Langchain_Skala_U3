import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from src.report_generator import generate_pdf_report
from src.schemas import CompanyResult, ComparisonResult, Domain, ProfileResult


class ReportGeneratorTests(unittest.TestCase):
    def test_report_has_ten_pages_and_contributor_section(self):
        names = ["ACCURE", "volytica diagnostics", "Electra Vehicles"]
        profiles = ["기술중심형", "안정형", "성장형", "균형형", "사업성중심형"]
        companies = []
        for name in names:
            companies.append(CompanyResult(
                company_name=name,
                domain_scores={domain: 75.0 for domain in Domain},
                evidence_coverage=80.0,
                profile_results=[ProfileResult(
                    profile_name=profile, weighted_score=75.0,
                    recommendation="추가 실사", due_diligence_questions=[],
                ) for profile in profiles],
                average_score=75.0, score_std_dev=0.0,
                final_decision="조건부 검토", summary_reason="테스트용 요약",
                evidence_sources=[f"{name}-P1", f"{name}-P2"],
                source_documents=[f"{name}-1.pdf", f"{name}-2.pdf"],
            ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.pdf"
            self.assertEqual(generate_pdf_report(ComparisonResult(companies=companies), path), str(path))
            reader = PdfReader(path)
            self.assertEqual(len(reader.pages), 10)
            self.assertIn("담당 역할과 투자 관점", reader.pages[7].extract_text())
            self.assertIn("[ACCURE-P1]", reader.pages[9].extract_text())
            self.assertIn("[Electra Vehicles-P2]", reader.pages[9].extract_text())
            self.assertEqual(float(reader.pages[7].mediabox.width), float(reader.pages[0].mediabox.width))
            self.assertEqual(reader.metadata.title, "배터리 AI 스타트업 투자 검토 보고서")

    def test_requires_three_companies(self):
        with self.assertRaises(ValueError):
            generate_pdf_report(ComparisonResult(companies=[]))

    def test_insufficient_evidence_is_unscored_in_pdf(self):
        companies = [CompanyResult(
            company_name=name, domain_scores={}, evidence_coverage=40,
            profile_results=[], average_score=None, score_std_dev=None,
            scoring_eligible=False, final_decision="근거 부족 판단 유보",
            summary_reason="근거 부족: 추가 조사 필요",
        ) for name in ("A", "B", "C")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            generate_pdf_report(ComparisonResult(companies=companies), path,
                                preview_label="검토용 - 품질 평가 미통과")
            reader = PdfReader(path)
            self.assertIn("점수 미산출", reader.pages[1].extract_text())
            self.assertIn("미산출", reader.pages[3].extract_text())
            self.assertIn("근거 부족", reader.pages[4].extract_text())
            self.assertTrue(all("품질 평가 미통과" in page.extract_text()
                                for page in reader.pages))


if __name__ == "__main__":
    unittest.main()
