"""
tests/test_codegen.py

Unit and integration tests for the experiment code generation capability.

Unit tests mock _get_client — no API key or network required.
Integration test calls the live API — requires ANTHROPIC_API_KEY.

Reference experiment: "Retrieval vs. Generation Bottleneck Ablation"
  (the most-promising idea confirmed in the ideation live output)
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.codegen import (
    _MODEL,
    _extract_metadata,
    generate_experiment_code,
)
from research_assistant.context import ResearchContext


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "What is the current state of retrieval-augmented generation?"

BOTTLENECK_IDEA = {
    "title": "Retrieval vs. Generation Bottleneck Ablation",
    "hypothesis": (
        "In current RAG systems, retrieval quality is a larger bottleneck than generation quality — "
        "improving retrieval yields larger EM gains than an equivalent improvement to generation."
    ),
    "method": (
        "2×2 factorial on Natural Questions and TriviaQA: "
        "(1) retrieval — oracle passages vs. DPR-retrieved passages, "
        "(2) generation — GPT-4 vs. Llama-2-7B as the reader. "
        "Measure Exact Match (EM) and F1. "
        "Baseline: DPR + Llama-2-7B. Primary metric: delta-EM from fixing each factor."
    ),
    "gap_addressed": (
        "Open question of whether retrieval or generation is the primary bottleneck in RAG pipelines, "
        "which determines where future research effort should focus."
    ),
    "difficulty": "medium",
}

_MINIMAL_NOTEBOOK = json.dumps({
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
})

_SAMPLE_SCRIPT = """\
# Retrieval vs. Generation Bottleneck Ablation
# Requirements: torch, transformers, datasets, faiss-cpu, sentence-transformers, evaluate
# Estimated Runtime: 4-8 hours on a single A100 GPU

import argparse
import json
import os


def run_experiment(args):
    print(f"Running on dataset: {args.dataset}")


def main():
    parser = argparse.ArgumentParser(description="RAG bottleneck ablation")
    parser.add_argument("--dataset", default="nq", help="Dataset name")
    parser.add_argument("--model", default="facebook/rag-token-nq", help="Model name")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_experiment(args)
    print("| Condition          | EM    | F1    |")
    print("|--------------------|-------|-------|")
    print("| DPR + Llama        | 41.5  | 53.2  |")
    print("| Oracle + Llama     | 62.1  | 71.8  |")
    print("| DPR + GPT-4        | 47.3  | 59.0  |")
    print("| Oracle + GPT-4     | 66.4  | 75.2  |")


if __name__ == "__main__":
    main()
"""


def _make_mock_client(script_text: str, notebook_json: str) -> MagicMock:
    """Return a mock Anthropic client whose two sequential calls return script then notebook."""
    script_resp = MagicMock()
    script_resp.content = [MagicMock(text=script_text)]

    nb_resp = MagicMock()
    nb_resp.content = [MagicMock(text=notebook_json)]

    client = MagicMock()
    client.messages.create.side_effect = [script_resp, nb_resp]
    return client


# ---------------------------------------------------------------------------
# Unit tests — model constant
# ---------------------------------------------------------------------------

class TestModel:
    def test_uses_sonnet(self):
        assert "sonnet" in _MODEL.lower()

    def test_not_haiku(self):
        assert "haiku" not in _MODEL.lower()


# ---------------------------------------------------------------------------
# Unit tests — metadata extraction
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_parses_requirements(self):
        _, reqs = _extract_metadata(_SAMPLE_SCRIPT)
        assert "torch" in reqs
        assert "transformers" in reqs
        assert "datasets" in reqs

    def test_parses_runtime(self):
        runtime, _ = _extract_metadata(_SAMPLE_SCRIPT)
        assert "hours" in runtime.lower()
        assert runtime != "unknown"

    def test_missing_lines_return_defaults(self):
        runtime, reqs = _extract_metadata("# just a plain comment\ndef main(): pass")
        assert runtime == "unknown"
        assert reqs == []

    def test_case_insensitive_keys(self):
        script = "# REQUIREMENTS: numpy, scipy\n# ESTIMATED RUNTIME: 1 hour\n"
        runtime, reqs = _extract_metadata(script)
        assert "numpy" in reqs
        assert runtime == "1 hour"

    def test_only_scans_first_30_lines(self):
        padding = "\n".join(f"# line {i}" for i in range(35))
        script = padding + "\n# Requirements: late_pkg\n"
        _, reqs = _extract_metadata(script)
        assert "late_pkg" not in reqs  # appears after line 30, should not be found


# ---------------------------------------------------------------------------
# Unit tests — generate_experiment_code (mocked)
# ---------------------------------------------------------------------------

class TestGenerateExperimentCodeMocked:
    def test_output_has_all_required_keys(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        required = {"python_script", "notebook", "experiment_title", "estimated_runtime", "requirements"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_both_api_calls_happen(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        assert mock_client.messages.create.call_count == 2, (
            "Expected exactly 2 API calls: one for the script, one for the notebook"
        )

    def test_correct_model_for_both_calls(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        for i, call in enumerate(mock_client.messages.create.call_args_list):
            assert call[1]["model"] == _MODEL, f"Call {i} used wrong model: {call[1]['model']}"

    def test_experiment_title_matches_input(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        assert result["experiment_title"] == BOTTLENECK_IDEA["title"]

    def test_notebook_is_valid_json(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        parsed = json.loads(result["notebook"])
        assert isinstance(parsed, dict)

    def test_strips_markdown_fences_from_script(self):
        fenced = f"```python\n{_SAMPLE_SCRIPT}\n```"
        mock_client = _make_mock_client(fenced, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        assert not result["python_script"].startswith("```")

    def test_strips_markdown_fences_from_notebook(self):
        fenced_nb = f"```json\n{_MINIMAL_NOTEBOOK}\n```"
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, fenced_nb)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        json.loads(result["notebook"])  # must not raise

    def test_invalid_notebook_json_falls_back_to_minimal_notebook(self):
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, "{{ not valid json")
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            result = generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)
        # Must not raise — fallback notebook should be valid JSON
        parsed = json.loads(result["notebook"])
        assert "cells" in parsed
        # Fallback wraps the whole script in a code cell
        code_cells = [c for c in parsed["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 1
        full_source = "".join(code_cells[0]["source"])
        assert "def main(" in full_source

    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)


# ---------------------------------------------------------------------------
# Unit tests — codegen_capability (context wrapper)
# ---------------------------------------------------------------------------

class TestCodegenCapability:
    def test_skips_when_no_experiment_ideas(self, capsys):
        from research_assistant.capabilities.codegen import codegen_capability
        ctx = ResearchContext(query="test")
        codegen_capability(ctx)
        out = capsys.readouterr().out
        assert "skipping" in out.lower()
        assert ctx.generated_code is None

    def test_uses_most_promising_idea(self):
        from research_assistant.capabilities.codegen import codegen_capability
        idea_a = {**BOTTLENECK_IDEA, "title": "Idea A"}
        idea_b = {**BOTTLENECK_IDEA, "title": "Idea B"}
        ctx = ResearchContext(
            query=SAMPLE_QUERY,
            experiment_ideas={
                "experiment_ideas": [idea_a, idea_b],
                "most_promising": 1,
                "reasoning": "B is better.",
            },
        )
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            codegen_capability(ctx)
        assert ctx.generated_code is not None
        assert ctx.generated_code["experiment_title"] == "Idea B"

    def test_writes_generated_code_to_context(self):
        from research_assistant.capabilities.codegen import codegen_capability
        ctx = ResearchContext(
            query=SAMPLE_QUERY,
            experiment_ideas={
                "experiment_ideas": [BOTTLENECK_IDEA],
                "most_promising": 0,
                "reasoning": "Only idea.",
            },
        )
        mock_client = _make_mock_client(_SAMPLE_SCRIPT, _MINIMAL_NOTEBOOK)
        with patch("research_assistant.capabilities.codegen._get_client", return_value=mock_client):
            codegen_capability(ctx)
        assert ctx.generated_code is not None
        assert "python_script" in ctx.generated_code
        assert "notebook" in ctx.generated_code


# ---------------------------------------------------------------------------
# Integration test — requires ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_codegen_result():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live codegen test.")
    return generate_experiment_code(BOTTLENECK_IDEA, SAMPLE_QUERY)


class TestCodegenLive:
    def test_python_script_non_empty(self, live_codegen_result):
        assert len(live_codegen_result["python_script"]) > 200

    def test_script_contains_main_function(self, live_codegen_result):
        assert "def main(" in live_codegen_result["python_script"]

    def test_script_contains_entrypoint(self, live_codegen_result):
        script = live_codegen_result["python_script"]
        assert '__name__ == "__main__"' in script or "__name__ == '__main__'" in script

    def test_notebook_is_valid_json(self, live_codegen_result):
        parsed = json.loads(live_codegen_result["notebook"])
        assert "cells" in parsed

    def test_requirements_non_empty(self, live_codegen_result):
        reqs = live_codegen_result["requirements"]
        assert isinstance(reqs, list) and len(reqs) > 0

    def test_estimated_runtime_non_empty(self, live_codegen_result):
        rt = live_codegen_result["estimated_runtime"]
        assert isinstance(rt, str) and rt.strip() and rt != "unknown"

    def test_print_full_output(self, live_codegen_result):
        """Print metadata and first 50 lines of the generated script for review."""
        script = live_codegen_result["python_script"]
        lines = script.splitlines()
        print(f"\n{'='*70}")
        print("CODEGEN — Live output")
        print(f"{'='*70}")
        print(f"Experiment   : {live_codegen_result['experiment_title']}")
        print(f"Runtime est. : {live_codegen_result['estimated_runtime']}")
        print(f"Requirements : {', '.join(live_codegen_result['requirements'])}")
        print(f"Script length: {len(script):,} chars / {len(lines)} lines")
        print(f"Notebook JSON: {len(live_codegen_result['notebook']):,} chars")
        print(f"\n--- First 50 lines of generated Python script ---\n")
        for line in lines[:50]:
            print(line)
        print(f"\n{'='*70}\n")
        assert True  # always passes — exists for its printed output
