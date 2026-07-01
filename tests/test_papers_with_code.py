"""
tests/test_papers_with_code.py

Unit and integration tests for the Papers with Code enrichment helper.

Unit tests mock requests.get — no network required.
Integration tests hit the live PwC API — require network access only,
no API key needed.

Reference papers:
    RAG_SURVEY_ID   = "2312.10997"  Gao et al. 2023 — survey, likely no benchmarks
    ORIGINAL_RAG_ID = "2005.11401"  Lewis et al. 2020 — methods paper, likely has benchmarks
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from research_assistant.capabilities.fetch_paper import _fetch_papers_with_code
from research_assistant.models import PaperRecord

RAG_SURVEY_ID = "2312.10997"
ORIGINAL_RAG_ID = "2005.11401"


# ---------------------------------------------------------------------------
# Helpers for mocking
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------

class TestFetchPapersWithCodeParsing:
    """Tests for response parsing logic, using mocked requests."""

    def test_returns_none_none_when_paper_not_found(self):
        not_found = _mock_response(200, {"count": 0, "results": []})
        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   return_value=not_found):
            result = _fetch_papers_with_code("9999.99999")
        assert result == (None, None)

    def test_returns_none_none_on_http_error(self):
        error_resp = _mock_response(404, {})
        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   return_value=error_resp):
            result = _fetch_papers_with_code("2005.11401")
        assert result == (None, None)

    def test_parses_benchmark_results(self):
        search_resp = _mock_response(200, {
            "results": [{"id": "retrieval-augmented-generation"}]
        })
        results_resp = _mock_response(200, {
            "results": [
                {
                    "task": {"name": "Question Answering"},
                    "dataset": {"name": "Natural Questions"},
                    "metrics": [{"type": {"name": "Exact Match"}, "value": "44.5"}],
                },
                {
                    "task": {"name": "Open Domain QA"},
                    "dataset": {"name": "TriviaQA"},
                    "metrics": [{"type": {"name": "Exact Match"}, "value": "68.0"}],
                },
            ]
        })
        repos_resp = _mock_response(200, {"results": []})

        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=[search_resp, results_resp, repos_resp]):
            benchmarks, code_url = _fetch_papers_with_code("2005.11401")

        assert benchmarks is not None
        assert len(benchmarks) == 2
        assert benchmarks[0]["task"] == "Question Answering"
        assert benchmarks[0]["dataset"] == "Natural Questions"
        assert benchmarks[0]["metric"] == "Exact Match"
        assert benchmarks[0]["score"] == "44.5"
        assert code_url is None

    def test_caps_at_five_benchmark_results(self):
        search_resp = _mock_response(200, {"results": [{"id": "some-paper"}]})
        many_results = [
            {
                "task": {"name": f"Task {i}"},
                "dataset": {"name": f"Dataset {i}"},
                "metrics": [{"type": {"name": "Metric"}, "value": str(i)}],
            }
            for i in range(10)
        ]
        results_resp = _mock_response(200, {"results": many_results})
        repos_resp = _mock_response(200, {"results": []})

        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=[search_resp, results_resp, repos_resp]):
            benchmarks, _ = _fetch_papers_with_code("1234.56789")

        assert benchmarks is not None
        assert len(benchmarks) == 5

    def test_picks_repo_with_most_stars(self):
        search_resp = _mock_response(200, {"results": [{"id": "some-paper"}]})
        results_resp = _mock_response(200, {"results": []})
        repos_resp = _mock_response(200, {
            "results": [
                {"url": "https://github.com/low/stars", "stars": 12},
                {"url": "https://github.com/high/stars", "stars": 3400},
                {"url": "https://github.com/mid/stars", "stars": 200},
            ]
        })

        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=[search_resp, results_resp, repos_resp]):
            _, code_url = _fetch_papers_with_code("1234.56789")

        assert code_url == "https://github.com/high/stars"

    def test_returns_none_none_on_exception(self):
        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=ConnectionError("timeout")):
            result = _fetch_papers_with_code("2005.11401")
        assert result == (None, None)

    def test_benchmark_result_dict_has_required_keys(self):
        search_resp = _mock_response(200, {"results": [{"id": "p"}]})
        results_resp = _mock_response(200, {
            "results": [{
                "task": {"name": "NER"},
                "dataset": {"name": "CoNLL-2003"},
                "metrics": [{"type": {"name": "F1"}, "value": "93.5"}],
            }]
        })
        repos_resp = _mock_response(200, {"results": []})

        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=[search_resp, results_resp, repos_resp]):
            benchmarks, _ = _fetch_papers_with_code("1234.56789")

        assert benchmarks is not None
        for b in benchmarks:
            assert "task" in b
            assert "dataset" in b
            assert "metric" in b
            assert "score" in b

    def test_handles_empty_metrics_gracefully(self):
        search_resp = _mock_response(200, {"results": [{"id": "p"}]})
        results_resp = _mock_response(200, {
            "results": [{
                "task": {"name": "Translation"},
                "dataset": {"name": "WMT14"},
                "metrics": [],
            }]
        })
        repos_resp = _mock_response(200, {"results": []})

        with patch("research_assistant.capabilities.fetch_paper.requests.get",
                   side_effect=[search_resp, results_resp, repos_resp]):
            benchmarks, _ = _fetch_papers_with_code("1234.56789")

        assert benchmarks is not None
        assert benchmarks[0]["metric"] == ""
        assert benchmarks[0]["score"] == ""


# ---------------------------------------------------------------------------
# Integration tests — require network, no API key
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pwc_survey(request):
    """Fetch PwC data for the RAG survey (2312.10997) — likely a survey, no benchmarks."""
    return _fetch_papers_with_code(RAG_SURVEY_ID)


@pytest.fixture(scope="module")
def pwc_original(request):
    """Fetch PwC data for Lewis et al. (2005.11401) — methods paper, likely has benchmarks."""
    return _fetch_papers_with_code(ORIGINAL_RAG_ID)


class TestPwCLiveApi:
    def test_survey_returns_tuple(self, pwc_survey):
        benchmarks, code_url = pwc_survey
        # Result is (None, None) or (list, str|None) — either is valid
        assert isinstance(benchmarks, (list, type(None)))
        assert isinstance(code_url, (str, type(None)))

    def test_original_returns_tuple(self, pwc_original):
        benchmarks, code_url = pwc_original
        assert isinstance(benchmarks, (list, type(None)))
        assert isinstance(code_url, (str, type(None)))

    def test_print_survey_pwc_data(self, pwc_survey):
        benchmarks, code_url = pwc_survey
        print(f"\n{'='*60}")
        print(f"PwC data for RAG survey ({RAG_SURVEY_ID})")
        print(f"{'='*60}")
        print(f"  code_url        : {code_url or 'none'}")
        if benchmarks:
            print(f"  benchmark_results ({len(benchmarks)}):")
            for b in benchmarks:
                print(f"    {b['task']} / {b['dataset']}: {b['metric']} = {b['score']}")
        else:
            print("  benchmark_results: none (expected — survey paper)")
        print(f"{'='*60}\n")
        assert True  # always passes — exists for its output

    def test_print_original_pwc_data(self, pwc_original):
        benchmarks, code_url = pwc_original
        print(f"\n{'='*60}")
        print(f"PwC data for Lewis et al. original RAG ({ORIGINAL_RAG_ID})")
        print(f"{'='*60}")
        print(f"  code_url        : {code_url or 'none'}")
        if benchmarks:
            print(f"  benchmark_results ({len(benchmarks)}):")
            for b in benchmarks:
                print(f"    {b['task']} / {b['dataset']}: {b['metric']} = {b['score']}")
        else:
            print("  benchmark_results: none")
        print(f"{'='*60}\n")
        assert True

    def test_benchmark_results_structure_when_present(self, pwc_original):
        benchmarks, _ = pwc_original
        if benchmarks is None:
            pytest.skip("Lewis et al. not indexed on PwC or has no benchmark results.")
        assert len(benchmarks) <= 5, "Should cap at 5 results"
        for b in benchmarks:
            assert set(b.keys()) >= {"task", "dataset", "metric", "score"}
            assert isinstance(b["task"], str)
            assert isinstance(b["dataset"], str)


@pytest.fixture(scope="module")
def fetched_papers():
    from research_assistant.capabilities.fetch_paper import fetch_papers
    papers = fetch_papers([RAG_SURVEY_ID, ORIGINAL_RAG_ID])
    if len(papers) < 2:
        pytest.skip("arXiv returned fewer than 2 papers — possible 503.")
    return {p.arxiv_id: p for p in papers}


class TestFetchPapersIntegration:
    """Run the full fetch pipeline and verify PwC fields on PaperRecord."""

    def test_paper_records_have_pwc_fields(self, fetched_papers):
        for paper in fetched_papers.values():
            # Fields exist (may be None if not on PwC)
            assert hasattr(paper, "benchmark_results")
            assert hasattr(paper, "code_url")

    def test_benchmark_results_type(self, fetched_papers):
        for paper in fetched_papers.values():
            if paper.benchmark_results is not None:
                assert isinstance(paper.benchmark_results, list)
                for b in paper.benchmark_results:
                    assert isinstance(b, dict)

    def test_code_url_type(self, fetched_papers):
        for paper in fetched_papers.values():
            assert paper.code_url is None or isinstance(paper.code_url, str)

    def test_print_full_fetch_pwc_output(self, fetched_papers):
        print(f"\n{'='*60}")
        print("FULL FETCH — Papers with Code fields")
        print(f"{'='*60}")
        for arxiv_id, paper in fetched_papers.items():
            print(f"\n  [{arxiv_id}] {paper.title[:60]}")
            print(f"    code_url: {paper.code_url or 'none'}")
            if paper.benchmark_results:
                print(f"    benchmarks ({len(paper.benchmark_results)}):")
                for b in paper.benchmark_results:
                    print(f"      {b['task']} / {b['dataset']}: "
                          f"{b['metric']} = {b['score']}")
            else:
                print("    benchmarks: none")
        print(f"\n{'='*60}\n")
        assert True
