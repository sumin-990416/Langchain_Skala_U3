import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.profile import calculate_profile_score
from src.graph import route_after_common_score
from src.schemas import Domain, Question
from src.agents.profiles.business import (
    BusinessCriterionAssessment,
    BusinessCustomAssessment,
    calculate_business_custom_score,
)
from src.agents.profiles.stability import (
    StabilityCriterionAssessment,
    StabilityCustomAssessment,
    calculate_stability_custom_score,
)
from src.rag import ingest


class ScoringPolicyTest(unittest.TestCase):
    def test_web_search_uses_scored_coverage_and_profile_critical_items(self):
        questions = [
            Question(domain=Domain.FINANCE, item_id="fin_01", query="BM"),
            Question(domain=Domain.FINANCE, item_id="fin_02", query="계약"),
            Question(domain=Domain.TEAM, item_id="team_01", query="팀"),
            Question(domain=Domain.MARKET, item_id="mkt_01", query="시장"),
            Question(domain=Domain.RISK, item_id="rsk_01", query="안전"),
            Question(domain=Domain.RISK, item_id="rsk_02", query="권한"),
            Question(domain=Domain.TECHNOLOGY, item_id="tech_01", query="성능"),
            Question(domain=Domain.TECHNOLOGY, item_id="tech_02", query="연동"),
        ]
        base = {"evaluation_questions": questions, "web_research_attempts": 0}
        with patch("src.graph.os.getenv", return_value="test-key"):
            self.assertEqual(
                route_after_common_score({**base, "evidence_coverage": 62.5, "unknown_items": ["tech_02"]}),
                "web_research",
            )
            self.assertEqual(
                route_after_common_score({**base, "evidence_coverage": 87.5, "unknown_items": ["fin_01"]}),
                "web_research",
            )
            self.assertEqual(
                route_after_common_score({**base, "evidence_coverage": 87.5, "unknown_items": ["tech_02"]}),
                "plan_worker_tasks",
            )

    def test_common_score_renormalizes_available_domains(self):
        self.assertEqual(
            calculate_profile_score({"finance": 80, "team": 60}, {"finance": 0.2, "team": 0.3, "risk": 0.5}),
            68.0,
        )

    def test_business_custom_score_requires_item_linked_citations(self):
        assessments = BusinessCustomAssessment(
            revenue_model=BusinessCriterionAssessment(score=75, evidence_ids=["F-01"], rationale="매출 확인"),
            customer_conversion=BusinessCriterionAssessment(score=50, evidence_ids=["F-02"], rationale="파일럿"),
            team_execution=BusinessCriterionAssessment(score=100, evidence_ids=["T-01"], rationale="실행 이력"),
        )
        result = calculate_business_custom_score(
            assessments, {"fin_01": ["F-01"], "fin_02": ["F-02"], "team_01": ["T-01"]}
        )
        self.assertEqual(result[0], 72.5)
        self.assertEqual(result[1], 100.0)
        rejected = calculate_business_custom_score(assessments, {"fin_01": ["F-01"], "fin_02": ["WRONG"], "team_01": ["T-01"]})
        self.assertIsNone(rejected[0])
        self.assertEqual(rejected[1], 60.0)

    def test_stability_custom_score_renormalizes_only_validly_cited_criteria(self):
        assessments = StabilityCustomAssessment(
            safety_regulation=StabilityCriterionAssessment(score=75, evidence_ids=["R-01"], rationale="확인"),
            financial_continuity=StabilityCriterionAssessment(score=50, evidence_ids=["F-01"], rationale="확인"),
            data_rights=StabilityCriterionAssessment(score=None, evidence_ids=[], rationale="자료 없음"),
            product_liability=StabilityCriterionAssessment(score=75, evidence_ids=["R-01"], rationale="확인"),
            intellectual_property=StabilityCriterionAssessment(score=100, evidence_ids=["T-03"], rationale="확인"),
        )
        score = calculate_stability_custom_score(
            assessments,
            {"rsk_01": ["R-01"], "fin_01": ["F-01"], "rsk_02": ["R-02"], "tech_03": ["T-03"]},
        )
        self.assertEqual(score[0], 71.43)
        self.assertEqual(score[1], 82.35)

    def test_source_urls_are_mapped_to_exact_evidence_ids_and_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "source_policy.yaml"
            policy_path.write_text(
                "tiers:\n  A: [sec.gov]\n  B: [example.com]\ncompany_official_domains: [accure.com]\n",
                encoding="utf-8",
            )
            content = "원문 [A-01]: https://sec.gov/filing/1\n원문 [E-02]: https://example.com/news/2"
            with patch.object(ingest, "SOURCE_POLICY_PATH", policy_path):
                metadata = ingest._source_metadata(content)
        self.assertEqual(metadata["source_urls_by_id"], {
            "A-01": "https://sec.gov/filing/1", "E-02": "https://example.com/news/2"
        })
        self.assertEqual(metadata["source_tiers_by_id"], {"A-01": "A", "E-02": "B"})
        self.assertEqual(metadata["evidence_grade"], "U")

    def test_source_mapping_does_not_guess_when_page_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "source_policy.yaml"
            policy_path.write_text("tiers:\n  A: [sec.gov]\n", encoding="utf-8")
            with patch.object(ingest, "SOURCE_POLICY_PATH", policy_path):
                metadata = ingest._source_metadata(
                    "A-01 A-02 https://sec.gov/one https://sec.gov/two"
                )
        self.assertEqual(metadata["source_urls_by_id"], {})


if __name__ == "__main__":
    unittest.main()
