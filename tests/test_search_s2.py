"""
tests/test_search_s2.py

Integration tests for the Semantic Scholar search capability and the
combined search_all function.

All tests hit live APIs — no mocks. Requires network access.
No ANTHROPIC_API_KEY needed.
"""
from __future__ import annotations

import pytest

from research_assistant.capabilities.search_semantic_scholar import (
    search_semantic_scholar,
)
from research_assistant.capabilities.search import (
    search_all,
    _normalize_title,
    _is_arxiv_id,
    _merge_results,
)
from research_assistant.models import PaperRecord

QUERY = "retrieval augmented generation"


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------

class TestIsArxivId:
    def test_modern_format(self):
        assert _is_arxiv_id("2312.10997")

    def test_modern_format_with_version(self):
        assert _is_arxiv_id("2312.10997v2")

    def test_old_format(self):
        assert _is_arxiv_id("cs.AI/0601001")

    def test_s2_hash_rejected(self):
        assert not _is_arxiv_id("1234567890abcdef1234567890abcdef12345678")

    def test_s2_short_hash_rejected(self):
        assert not _is_arxiv_id("abc123def456")


class TestNormalizeTitle:
    def test_lowercases(self):
        assert _normalize_title("RAG Survey") == "rag survey"

    def test_strips_punctuation(self):
        assert _normalize_title("RAG: A Survey.") == "rag a survey"

    def test_collapses_whitespace(self):
        result = _normalize_title("Retrieval  Augmented   Generation")
        assert "  " not in result


class TestMergeResults:
    def _make(self, arxiv_id, title, source="arxiv"):
        return PaperRecord(
            arxiv_id=arxiv_id,
            title=title,
            authors=[],
            abstract="",
            published="",
            pdf_url="",
            arxiv_url="",
            source=source,
        )

    def test_arxiv_results_always_included(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        merged = _merge_results(arxiv, [])
        assert len(merged) == 1
        assert merged[0].source == "arxiv"

    def test_s2_duplicate_by_id_removed(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        s2 = [self._make("2312.10997", "RAG Survey", source="semantic_scholar")]
        merged = _merge_results(arxiv, s2)
        assert len(merged) == 1

    def test_s2_duplicate_by_title_removed(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        # Different arXiv ID but identical title
        s2 = [self._make("2005.11401", "RAG Survey", source="semantic_scholar")]
        merged = _merge_results(arxiv, s2)
        assert len(merged) == 1

    def test_s2_without_arxiv_id_excluded(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        # S2-only paper: paperId as arxiv_id, not a real arXiv ID
        s2 = [self._make("1234567890abcdef1234567890abcdef12345678", "New Paper", source="semantic_scholar")]
        merged = _merge_results(arxiv, s2)
        assert len(merged) == 1  # S2-only paper excluded

    def test_s2_exclusive_arxiv_paper_added(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        # S2 found a different arXiv paper not in arXiv results
        s2 = [self._make("2005.11401", "Original RAG Paper", source="semantic_scholar")]
        merged = _merge_results(arxiv, s2)
        assert len(merged) == 2
        sources = {p.source for p in merged}
        assert "arxiv" in sources
        assert "semantic_scholar" in sources

    def test_arxiv_results_come_first(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        s2 = [self._make("2005.11401", "Original RAG", source="semantic_scholar")]
        merged = _merge_results(arxiv, s2)
        assert merged[0].source == "arxiv"
        assert merged[1].source == "semantic_scholar"

    def test_no_duplicate_ids_in_output(self):
        arxiv = [self._make("2312.10997", "RAG Survey")]
        s2 = [
            self._make("2312.10997", "RAG Survey", source="semantic_scholar"),
            self._make("2005.11401", "Original RAG", source="semantic_scholar"),
        ]
        merged = _merge_results(arxiv, s2)
        ids = [p.arxiv_id for p in merged]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Integration tests — require network
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s2_results():
    results = search_semantic_scholar(QUERY)
    if not results:
        pytest.skip("Semantic Scholar returned no results — possible network issue.")
    return results


@pytest.fixture(scope="module")
def all_results():
    results = search_all(QUERY)
    if not results:
        pytest.skip("search_all returned no results.")
    return results


class TestSearchSemanticScholar:
    def test_returns_at_least_five_results(self, s2_results):
        assert len(s2_results) >= 5, (
            f"Expected >= 5 results, got {len(s2_results)}"
        )

    def test_each_result_is_paper_record(self, s2_results):
        for p in s2_results:
            assert isinstance(p, PaperRecord)

    def test_each_result_has_title(self, s2_results):
        for p in s2_results:
            assert p.title, f"Empty title for paper {p.arxiv_id}"

    def test_each_result_has_abstract(self, s2_results):
        for p in s2_results:
            assert p.abstract, f"Empty abstract for paper {p.arxiv_id}"

    def test_source_is_semantic_scholar(self, s2_results):
        for p in s2_results:
            assert p.source == "semantic_scholar", (
                f"Expected source='semantic_scholar', got {p.source!r} for {p.arxiv_id}"
            )

    def test_print_first_three_results(self, s2_results):
        print(f"\n{'='*60}")
        print("SEMANTIC SCHOLAR — first 3 results")
        print(f"{'='*60}")
        for p in s2_results[:3]:
            print(f"\n  [{p.source}] {p.arxiv_id}")
            print(f"  Title   : {p.title}")
            print(f"  Authors : {', '.join(p.authors[:3])}")
            print(f"  Published: {p.published}")
            print(f"  Citations: {p.citation_count}")
            print(f"  Abstract: {p.abstract[:120]}...")
        print(f"{'='*60}\n")
        assert True


class TestSearchAll:
    def test_returns_results(self, all_results):
        assert len(all_results) > 0

    def test_no_duplicate_arxiv_ids(self, all_results):
        ids = [p.arxiv_id for p in all_results]
        assert len(ids) == len(set(ids)), (
            f"Duplicate arxiv_ids in search_all results: "
            f"{[x for x in ids if ids.count(x) > 1]}"
        )

    def test_arxiv_source_present(self, all_results):
        sources = {p.source for p in all_results}
        assert "arxiv" in sources, f"No arXiv papers in merged results. Sources: {sources}"

    def test_s2_source_present(self, all_results):
        sources = {p.source for p in all_results}
        if "semantic_scholar" not in sources:
            pytest.xfail(
                "S2 returned no papers that weren't already in the arXiv results "
                "for this query — complete overlap is possible but rare."
            )

    def test_all_arxiv_ids_are_valid_format(self, all_results):
        for p in all_results:
            assert _is_arxiv_id(p.arxiv_id), (
                f"Non-arXiv ID leaked into search_all results: "
                f"{p.arxiv_id!r} (source={p.source!r}, title={p.title!r})"
            )

    def test_print_all_results_with_source(self, all_results):
        arxiv_count = sum(1 for p in all_results if p.source == "arxiv")
        s2_count = sum(1 for p in all_results if p.source == "semantic_scholar")
        print(f"\n{'='*60}")
        print(f"SEARCH_ALL — {len(all_results)} results "
              f"({arxiv_count} arXiv, {s2_count} S2-exclusive)")
        print(f"{'='*60}")
        for p in all_results:
            print(f"  [{p.source:20s}] [{p.arxiv_id}] {p.title[:60]}")
        print(f"{'='*60}\n")
        assert True
