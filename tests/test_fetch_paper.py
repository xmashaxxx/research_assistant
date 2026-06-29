"""
tests/test_fetch_paper.py

Live integration tests for the fetch_paper capability.
Hits the real arXiv API — requires network access.

Integration tests share module-scoped fixtures that pre-fetch papers once
per pytest run. If arXiv returns an empty result (e.g. transient 503),
the fixture calls pytest.skip() and the entire dependent test class is
skipped with a clear message rather than failing with an IndexError.

Reference papers used
---------------------
RAG_SURVEY_ID  = "2312.10997"
    Gao et al., "Retrieval-Augmented Generation for Large Language Models:
    A Survey" — primary test anchor.

ORIGINAL_RAG_ID = "2005.11401"
    Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
    NLP Tasks" — used for multi-fetch and single-element tests.

Both are stable, widely-cited, and won't be retracted.
"""
import pytest

from research_assistant.capabilities.fetch_paper import fetch_papers, _strip_version, _normalize_id
from research_assistant.models import PaperRecord

RAG_SURVEY_ID   = "2312.10997"   # Gao et al. 2023
ORIGINAL_RAG_ID = "2005.11401"   # Lewis et al. 2020


# ---------------------------------------------------------------------------
# Module-scoped fixtures — arXiv is called at most twice per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_records():
    """Fetch both test papers in a single call. Skip all dependents on 503."""
    results = fetch_papers([RAG_SURVEY_ID, ORIGINAL_RAG_ID])
    if len(results) < 2:
        pytest.skip(
            f"arXiv returned {len(results)}/2 papers — possible 503, re-run to confirm."
        )
    return results


@pytest.fixture(scope="module")
def rag_survey_record(both_records):
    """Single PaperRecord for the RAG survey — derived from both_records."""
    return both_records[0]


@pytest.fixture(scope="module")
def original_rag_record(both_records):
    """Single PaperRecord for the original RAG paper — derived from both_records."""
    return both_records[1]


@pytest.fixture(scope="module")
def versioned_record():
    """Fetch the RAG survey by versioned ID to test ID normalisation."""
    results = fetch_papers([f"{RAG_SURVEY_ID}v3"])
    if not results:
        pytest.skip(
            "arXiv returned no results for versioned ID — possible 503, re-run to confirm."
        )
    return results[0]


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestNormalizeId:
    """Covers the URL forms that search.py stores in arxiv_id."""

    def test_full_url_with_version(self):
        assert _normalize_id("https://arxiv.org/abs/2506.06962v3") == "2506.06962"

    def test_full_url_no_version(self):
        assert _normalize_id("https://arxiv.org/abs/2506.06962") == "2506.06962"

    def test_clean_id_passthrough(self):
        assert _normalize_id("2312.10997") == "2312.10997"

    def test_clean_id_with_version(self):
        assert _normalize_id("2312.10997v2") == "2312.10997"


class TestStripVersion:
    def test_no_suffix(self):
        assert _strip_version("2312.10997") == "2312.10997"

    def test_with_suffix(self):
        assert _strip_version("2312.10997v2") == "2312.10997"

    def test_with_suffix_v1(self):
        assert _strip_version("2005.11401v1") == "2005.11401"

    def test_legacy_long_form(self):
        assert _strip_version("cs/0601097v1") == "cs/0601097"

    def test_legacy_no_suffix(self):
        assert _strip_version("cs/0601097") == "cs/0601097"


# ---------------------------------------------------------------------------
# Integration tests (network required — use module-scoped fixtures above)
# ---------------------------------------------------------------------------

class TestFetchSinglePaper:
    def test_returns_paper_record_type(self, rag_survey_record):
        assert isinstance(rag_survey_record, PaperRecord)

    def test_correct_arxiv_id(self, rag_survey_record):
        assert rag_survey_record.arxiv_id == RAG_SURVEY_ID

    def test_title_contains_rag(self, rag_survey_record):
        assert "Retrieval-Augmented Generation" in rag_survey_record.title

    def test_has_multiple_authors(self, rag_survey_record):
        assert len(rag_survey_record.authors) > 1

    def test_abstract_is_substantive(self, rag_survey_record):
        assert len(rag_survey_record.abstract) > 200

    def test_pdf_url_is_https(self, rag_survey_record):
        assert rag_survey_record.pdf_url.startswith("http")

    def test_published_date_present(self, rag_survey_record):
        assert rag_survey_record.published != ""

    def test_has_cs_category(self, rag_survey_record):
        assert any(cat.startswith("cs.") for cat in rag_survey_record.categories)

    def test_primary_category_present(self, rag_survey_record):
        assert rag_survey_record.primary_category != ""

    def test_source_is_arxiv(self, rag_survey_record):
        assert rag_survey_record.source == "arxiv"

    def test_version_suffix_stripped(self, versioned_record):
        assert versioned_record.arxiv_id == RAG_SURVEY_ID


class TestFetchMultiplePapers:
    def test_returns_two_records(self, both_records):
        assert len(both_records) == 2

    def test_order_preserved(self, both_records):
        assert both_records[0].arxiv_id == RAG_SURVEY_ID
        assert both_records[1].arxiv_id == ORIGINAL_RAG_ID

    def test_both_records_have_titles(self, both_records):
        for paper in both_records:
            assert len(paper.title) > 0

    def test_both_records_have_abstracts(self, both_records):
        for paper in both_records:
            assert len(paper.abstract) > 50

    def test_both_records_have_authors(self, both_records):
        for paper in both_records:
            assert len(paper.authors) > 0


class TestEdgeCases:
    def test_empty_list_returns_empty(self):
        results = fetch_papers([])
        assert results == []

    def test_single_element_list(self, original_rag_record):
        assert original_rag_record.arxiv_id == ORIGINAL_RAG_ID
