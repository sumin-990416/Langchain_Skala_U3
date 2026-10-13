"""제공 PDF 검색과 원문 인용 경계 검증. 네트워크와 LLM API를 쓰지 않는다."""

import unittest
from unittest.mock import patch

from src.orchestration.evidence import (
    _card_blocks,
    _publisher,
    _tokens,
    retrieve_keyword_evidence,
)
from src.orchestration.schemas import (
    Citation,
    Claim,
    Perspective,
    ResearchTask,
    SourceRecord,
    WorkerAnalysis,
)
from src.orchestration.worker import quotable_spans, resolve_quote_ids, validate_citations


def _task(company="Electra Vehicles", perspective=Perspective.MARKET):
    return ResearchTask(
        task_id="evidence-audit",
        company_name=company,
        perspective=perspective,
        question="고객 선정과 배포를 검토하고 실제 매출과 제한 사항을 확인한다.",
        queries=["고객 선정 배포 시장 계약"],
        rationale="긍정 도입 사례와 불리한 근거를 함께 검토한다.",
    )


def _record(evidence_id, content, source_id="E01", company="Electra Vehicles"):
    return SourceRecord(
        evidence_id=evidence_id,
        source_id=source_id,
        source_file="fixture.pdf",
        page=1,
        source_group=company,
        content=content,
    )


class EvidenceOriginsTest(unittest.TestCase):
    def test_card_split_keeps_inline_reference_with_its_claim(self):
        summary = (
            "ACCURE 평가 근거\n"
            "[A-01] 고객 사례 설명.\n"
            "원문 [A-01]: https://www.accure.net/example\n"
            "[A-03] 계약 조건 설명.\n"
            "원문 [A-03]: https://www.accure.net/contract\n"
        )
        cards = _card_blocks(summary, "ACCURE_balanced_evidence.pdf")
        self.assertEqual([source_id for source_id, _ in cards], ["A-01", "A-03"])
        self.assertIn("원문 [A-01]", cards[0][1])
        self.assertNotIn("[A-03]", cards[0][1])

        pack = "A03 | ACCURE | SOC\n회사 사례.\nG01 | 공통 | 시장\n산업 통계."
        self.assertEqual(
            [source_id for source_id, _ in _card_blocks(pack, "BESS_RAG_Source_Pack.pdf")],
            ["A03", "G01"],
        )

    def test_two_files_of_same_company_are_one_publisher(self):
        summary_origin = _publisher(
            "[A-03] 회사가 발표한 계약 사례.",
            "https://www.accure.net/contract",
            "ACCURE",
        )
        card_origin = _publisher(
            "A03 | ACCURE | 제품\n작성 주체 · 기업 / 기업 주장/공개",
            "https://cdn.example.com/accure-card.pdf",
            "ACCURE",
        )
        self.assertEqual(summary_origin, "ACCURE")
        self.assertEqual({summary_origin, card_origin}, {"ACCURE"})

    def test_sec_hosting_does_not_create_an_independent_company_publisher(self):
        sec_url = "https://www.sec.gov/Archives/edgar/company-announcement.htm"
        contents = {
            "E09": "E09 | ELECTRA AI | 재무\n작성 주체 · ELECTRA/거래 당사자 / 회사 재무 공시",
            "E-03": "[E-03] SEC에 제출된 회사 보도자료에서 배포 중이라고 밝혔다.",
        }
        for source_id, content in contents.items():
            with self.subTest(source_id=source_id):
                self.assertEqual(_publisher(content, sec_url, "Electra Vehicles"), "Electra Vehicles")
        self.assertEqual(
            _publisher("작성 주체 · IEA / 일반 연구·통계", "https://www.iea.org/example", "ALL"),
            "IEA",
        )


class KeywordRetrievalTest(unittest.TestCase):
    def test_company_scope_excludes_matching_competitor_but_keeps_general_context(self):
        corpus = [
            ("Electra Vehicles", _record("own", "고객 선정 배포", "E01"), _tokens("고객 선정 배포")),
            ("ACCURE", _record("other-a", "고객 선정 배포 시장 계약", "A01", "ACCURE"), _tokens("고객 선정 배포 시장 계약")),
            ("volytica diagnostics", _record("other-v", "고객 선정 배포 시장 계약", "V01", "volytica diagnostics"), _tokens("고객 선정 배포 시장 계약")),
            ("ALL", _record("context", "산업 전체 시장 통계", "G01", "IEA"), _tokens("산업 전체 시장 통계")),
        ]
        with patch("src.orchestration.evidence._load_keyword_corpus", return_value=tuple(corpus)), patch(
            "src.orchestration.evidence._get_db", side_effect=AssertionError("키워드 검색은 임베딩을 호출하면 안 된다")
        ):
            records = retrieve_keyword_evidence(_task())
        self.assertEqual({record.evidence_id for record in records}, {"own", "context"})

    def test_negative_financial_card_survives_positive_customer_query_limit(self):
        positive = [
            _record(f"positive-{index}", "고객 선정 배포 시장 계약", f"E{index + 20}")
            for index in range(13)
        ]
        financial = _record("financial", "현금 부족. 단일고객 의존. 순손실.", "E09")
        general = [
            _record(f"context-{index}", "산업 시장 맥락", f"G{index + 1:02d}", "IEA")
            for index in range(3)
        ]
        corpus = tuple(
            [("Electra Vehicles", record, _tokens(record.content)) for record in positive + [financial]]
            + [("ALL", record, _tokens(record.content)) for record in general]
        )
        for perspective in (Perspective.MARKET, Perspective.STAKEHOLDERS):
            with self.subTest(perspective=perspective), patch(
                "src.orchestration.evidence._load_keyword_corpus", return_value=corpus
            ):
                records = retrieve_keyword_evidence(_task(perspective=perspective))
            self.assertLessEqual(len(records), 15)
            self.assertIn("E09", {record.source_id for record in records})
            self.assertEqual(len([record for record in records if record.source_id.startswith("G")]), 3)

    def test_actual_electra_pdf_market_evidence_includes_adverse_financials_and_origin(self):
        with patch("src.orchestration.evidence._get_db", side_effect=AssertionError("오프라인 검증")), patch(
            "src.orchestration.worker.ChatOpenAI", side_effect=AssertionError("LLM을 호출하면 안 된다")
        ):
            records = retrieve_keyword_evidence(_task())
        by_source_id = {record.source_id: record for record in records}
        self.assertIn("E09", by_source_id)
        financial = by_source_id["E09"]
        self.assertEqual(financial.source_file, "BESS_RAG_Source_Pack.pdf")
        self.assertEqual(financial.page, 19)
        self.assertEqual(financial.source_group, "Electra Vehicles")
        self.assertTrue(financial.url.startswith("https://www.sec.gov/"))
        self.assertIn("1,088,497", financial.content)
        self.assertIn("94%", financial.content)
        self.assertIn("해당 계약은 종료", financial.content)
        self.assertIn("계속기업", financial.content)
        self.assertTrue(any(record.source_id.startswith("G") for record in records))
        self.assertTrue(all(record.source_id.startswith(("E", "G")) for record in records))


class QuoteIdBoundaryTest(unittest.TestCase):
    def test_selected_quote_restores_whitespace_normalized_source_without_mutating_input(self):
        source = _record("EV-valid", "고객\t선정은 확인됐다.\n\n실제   매출은 공개되지 않았다.")
        quotes = quotable_spans([source])[source.evidence_id]
        quote_id = next(key for key, quote in quotes.items() if "매출" in quote)
        analysis = WorkerAnalysis(claims=[Claim(
            text="회사 발표는 선정과 인식 매출을 구분해야 한다.",
            kind="limitation",
            citations=[Citation(evidence_id=source.evidence_id, quote_id=quote_id, quote="위조된 문구")],
        )])
        restored = resolve_quote_ids(analysis, [source])
        quote = restored.claims[0].citations[0].quote
        self.assertEqual(quote, "실제 매출은 공개되지 않았다.")
        self.assertIn(quote, " ".join(source.content.split()))
        self.assertEqual(len(validate_citations(restored, [source]).claims), 1)
        self.assertEqual(analysis.claims[0].citations[0].quote, "위조된 문구")

    def test_unknown_quote_or_evidence_id_discards_whole_claim_even_with_one_valid_citation(self):
        first = _record("EV-first", "고객 선정. 매출 미확인.")
        second = _record("EV-second", "독립 성능시험 미확인.")
        analysis = WorkerAnalysis(claims=[
            Claim(text="유효한 인용", kind="context", citations=[Citation(evidence_id="EV-first", quote_id="Q01")]),
            Claim(text="위조 quote ID", kind="support", citations=[Citation(evidence_id="EV-first", quote_id="Q99", quote="고객 선정.")]),
            Claim(text="입력에 없는 evidence ID", kind="support", citations=[Citation(evidence_id="EV-missing", quote_id="Q01")]),
            Claim(text="다른 문서의 quote ID 사용", kind="support", citations=[Citation(evidence_id="EV-second", quote_id="Q02")]),
            Claim(text="유효 인용과 위조 인용 혼합", kind="support", citations=[
                Citation(evidence_id="EV-first", quote_id="Q01"),
                Citation(evidence_id="EV-first", quote_id="Q99"),
            ]),
        ])
        result = validate_citations(resolve_quote_ids(analysis, [first, second]), [first, second])
        self.assertEqual([claim.text for claim in result.claims], ["유효한 인용"])
        self.assertEqual(len(result.unknowns), 4)
        self.assertTrue(all("인용 검증 실패" in reason for reason in result.unknowns))

    def test_long_source_spans_remain_bounded_real_source_substrings(self):
        source = _record("EV-long", "검증조건과 한계를\n함께 기록한다 " * 60)
        normalized = " ".join(source.content.split())
        quotes = list(quotable_spans([source])[source.evidence_id].values())
        self.assertGreater(len(quotes), 1)
        for quote in quotes:
            self.assertTrue(quote)
            self.assertLessEqual(len(quote), 360)
            self.assertIn(quote, normalized)


if __name__ == "__main__":
    unittest.main()
