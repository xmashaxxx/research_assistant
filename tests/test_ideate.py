"""
tests/test_ideate.py

Unit and integration tests for the experiment ideation capability.

Unit tests mock the Anthropic client — no API key required.
Integration test calls the live API — requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.ideate import (
    _IDEATE_TOOL,
    _MODEL,
    _build_ideation_prompt,
    generate_experiment_ideas,
)
from research_assistant.context import ResearchContext


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "What is the current state of retrieval-augmented generation?"

SAMPLE_SYNTHESIS = (
    "Retrieval-augmented generation (RAG) has emerged as a dominant paradigm "
    "for grounding language model outputs in external knowledge. Open questions "
    "remain around optimal chunk sizes, dense vs. sparse retrieval, and whether "
    "retrieval quality or generation quality is the current bottleneck."
)

SAMPLE_COMPARISONS = {
    "agreements": ["Dense retrieval outperforms BM25 on most benchmarks."],
    "disagreements": [
        {
            "point": "Optimal chunk size for RAG pipelines",
            "positions": {
                "2312.10997": "Smaller chunks (128 tokens) improve precision.",
                "2005.11401": "Larger passages (512 tokens) provide more context.",
            },
        }
    ],
    "unique_contributions": {
        "2312.10997": "Comprehensive taxonomy of RAG variants.",
        "2005.11401": "Original seq2seq RAG formulation on NQ and TriviaQA.",
    },
}

_VALID_IDEA = {
    "title": "Chunk-size ablation on NQ",
    "hypothesis": "Smaller chunks (128 tokens) reduce hallucination on Natural Questions.",
    "method": "Compare 128/256/512-token chunks with DPR retriever on NQ EM metric.",
    "gap_addressed": "Disagreement over optimal RAG chunk size.",
    "difficulty": "low",
}


def _mock_tool_response(ideas: list[dict], most_promising: int = 0, reasoning: str = "Best ratio.") -> MagicMock:
    """Build a fake Anthropic messages response with a propose_experiments tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "propose_experiments"
    block.input = {
        "experiment_ideas": ideas,
        "most_promising": most_promising,
        "reasoning": reasoning,
    }
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"
    return response


# ---------------------------------------------------------------------------
# Unit tests — tool schema
# ---------------------------------------------------------------------------

class TestIdeateToolSchema:
    def test_tool_name(self):
        assert _IDEATE_TOOL["name"] == "propose_experiments"

    def test_schema_has_required_top_level_keys(self):
        props = _IDEATE_TOOL["input_schema"]["properties"]
        assert "experiment_ideas" in props
        assert "most_promising" in props
        assert "reasoning" in props

    def test_experiment_ideas_is_array(self):
        prop = _IDEATE_TOOL["input_schema"]["properties"]["experiment_ideas"]
        assert prop["type"] == "array"

    def test_min_max_items(self):
        prop = _IDEATE_TOOL["input_schema"]["properties"]["experiment_ideas"]
        assert prop["minItems"] == 3
        assert prop["maxItems"] == 5

    def test_item_required_fields(self):
        items = _IDEATE_TOOL["input_schema"]["properties"]["experiment_ideas"]["items"]
        required = set(items["required"])
        assert required == {"title", "hypothesis", "method", "gap_addressed", "difficulty"}

    def test_difficulty_enum(self):
        items = _IDEATE_TOOL["input_schema"]["properties"]["experiment_ideas"]["items"]
        diff_prop = items["properties"]["difficulty"]
        assert set(diff_prop["enum"]) == {"low", "medium", "high"}

    def test_most_promising_is_integer(self):
        prop = _IDEATE_TOOL["input_schema"]["properties"]["most_promising"]
        assert prop["type"] == "integer"

    def test_top_level_required(self):
        required = set(_IDEATE_TOOL["input_schema"]["required"])
        assert required == {"experiment_ideas", "most_promising", "reasoning"}


# ---------------------------------------------------------------------------
# Unit tests — model constant
# ---------------------------------------------------------------------------

class TestModel:
    def test_uses_sonnet(self):
        assert "sonnet" in _MODEL.lower()


# ---------------------------------------------------------------------------
# Unit tests — prompt construction
# ---------------------------------------------------------------------------

class TestBuildIdeationPrompt:
    def test_query_appears_in_prompt(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert SAMPLE_QUERY in prompt

    def test_synthesis_appears_in_prompt(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert "retrieval-augmented generation" in prompt.lower()

    def test_disagreements_appear_in_prompt(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert "Optimal chunk size" in prompt

    def test_unique_contributions_appear_in_prompt(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert "taxonomy" in prompt.lower()

    def test_paper_positions_appear_in_prompt(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert "2312.10997" in prompt
        assert "2005.11401" in prompt

    def test_empty_comparisons_does_not_crash(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, {})
        assert SAMPLE_QUERY in prompt

    def test_prompt_contains_task_instructions(self):
        prompt = _build_ideation_prompt(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
        assert "propose_experiments" in prompt or "experiment" in prompt.lower()


# ---------------------------------------------------------------------------
# Unit tests — generate_experiment_ideas (mocked)
# ---------------------------------------------------------------------------

def _make_mock_client(response: MagicMock) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns `response`."""
    client = MagicMock()
    client.messages.create.return_value = response
    return client


class TestGenerateExperimentIdeasMocked:
    def test_returns_dict_with_expected_keys(self):
        ideas = [_VALID_IDEA, {**_VALID_IDEA, "title": "B", "difficulty": "medium"},
                 {**_VALID_IDEA, "title": "C", "difficulty": "high"}]
        mock_client = _make_mock_client(_mock_tool_response(ideas, most_promising=0, reasoning="Low effort."))

        with patch("research_assistant.capabilities.ideate._get_client", return_value=mock_client):
            result = generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)

        assert "experiment_ideas" in result
        assert "most_promising" in result
        assert "reasoning" in result

    def test_passes_correct_model(self):
        ideas = [_VALID_IDEA, {**_VALID_IDEA, "title": "B", "difficulty": "medium"},
                 {**_VALID_IDEA, "title": "C", "difficulty": "high"}]
        mock_client = _make_mock_client(_mock_tool_response(ideas))

        with patch("research_assistant.capabilities.ideate._get_client", return_value=mock_client):
            generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
            call_kwargs = mock_client.messages.create.call_args[1]

        assert call_kwargs["model"] == _MODEL

    def test_uses_tool_choice_forced(self):
        ideas = [_VALID_IDEA, {**_VALID_IDEA, "title": "B", "difficulty": "medium"},
                 {**_VALID_IDEA, "title": "C", "difficulty": "high"}]
        mock_client = _make_mock_client(_mock_tool_response(ideas))

        with patch("research_assistant.capabilities.ideate._get_client", return_value=mock_client):
            generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)
            call_kwargs = mock_client.messages.create.call_args[1]

        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "propose_experiments"}

    def test_raises_if_no_tool_use_block(self):
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_client = _make_mock_client(mock_response)

        with patch("research_assistant.capabilities.ideate._get_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="propose_experiments"):
                generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)

    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)


# ---------------------------------------------------------------------------
# Unit tests — ideate_capability (context wrapper)
# ---------------------------------------------------------------------------

class TestIdeateCapability:
    def test_skips_when_no_synthesis(self, capsys):
        from research_assistant.capabilities.ideate import ideate_capability
        ctx = ResearchContext(query="test")
        # synthesis is None by default
        ideate_capability(ctx)
        out = capsys.readouterr().out
        assert "skipping" in out.lower()
        assert ctx.experiment_ideas is None

    def test_writes_experiment_ideas_to_context(self):
        from research_assistant.capabilities.ideate import ideate_capability
        ctx = ResearchContext(
            query=SAMPLE_QUERY,
            synthesis=SAMPLE_SYNTHESIS,
            comparisons=SAMPLE_COMPARISONS,
        )
        ideas = [_VALID_IDEA, {**_VALID_IDEA, "title": "B", "difficulty": "medium"},
                 {**_VALID_IDEA, "title": "C", "difficulty": "high"}]
        mock_client = _make_mock_client(_mock_tool_response(ideas, most_promising=1, reasoning="Medium is best."))

        with patch("research_assistant.capabilities.ideate._get_client", return_value=mock_client):
            ideate_capability(ctx)

        assert ctx.experiment_ideas is not None
        assert len(ctx.experiment_ideas["experiment_ideas"]) == 3
        assert ctx.experiment_ideas["most_promising"] == 1


# ---------------------------------------------------------------------------
# Integration test — requires ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_ideation_result():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live ideation test.")
    return generate_experiment_ideas(SAMPLE_QUERY, SAMPLE_SYNTHESIS, SAMPLE_COMPARISONS)


class TestIdeationLive:
    def test_returns_dict(self, live_ideation_result):
        assert isinstance(live_ideation_result, dict)

    def test_has_at_least_three_ideas(self, live_ideation_result):
        ideas = live_ideation_result.get("experiment_ideas") or []
        assert len(ideas) >= 3, f"Expected ≥3 ideas, got {len(ideas)}"

    def test_has_at_most_five_ideas(self, live_ideation_result):
        ideas = live_ideation_result.get("experiment_ideas") or []
        assert len(ideas) <= 5

    def test_each_idea_has_required_keys(self, live_ideation_result):
        required = {"title", "hypothesis", "method", "gap_addressed", "difficulty"}
        for idea in live_ideation_result["experiment_ideas"]:
            missing = required - set(idea.keys())
            assert not missing, f"Idea missing keys: {missing}"

    def test_each_idea_difficulty_is_valid(self, live_ideation_result):
        valid = {"low", "medium", "high"}
        for idea in live_ideation_result["experiment_ideas"]:
            assert idea["difficulty"] in valid, f"Invalid difficulty: {idea['difficulty']}"

    def test_most_promising_is_valid_index(self, live_ideation_result):
        ideas = live_ideation_result.get("experiment_ideas") or []
        idx = live_ideation_result.get("most_promising")
        assert isinstance(idx, int)
        assert 0 <= idx < len(ideas), f"most_promising={idx} out of range for {len(ideas)} ideas"

    def test_reasoning_is_non_empty_string(self, live_ideation_result):
        reasoning = live_ideation_result.get("reasoning", "")
        assert isinstance(reasoning, str) and reasoning.strip()

    def test_print_full_ideation_output(self, live_ideation_result):
        """Print the full experiment ideas so the user can read them."""
        ideas = live_ideation_result.get("experiment_ideas") or []
        most_promising = live_ideation_result.get("most_promising", 0)
        reasoning = live_ideation_result.get("reasoning", "")
        print(f"\n{'='*70}")
        print("EXPERIMENT IDEAS — Live ideation output")
        print(f"{'='*70}")
        print(f"Most promising: #{most_promising} | Reasoning: {reasoning}\n")
        for i, idea in enumerate(ideas):
            star = " *** MOST PROMISING ***" if i == most_promising else ""
            print(f"[{i}] {idea['title']}{star}")
            print(f"    Difficulty : {idea['difficulty']}")
            print(f"    Hypothesis : {idea['hypothesis']}")
            print(f"    Method     : {idea['method']}")
            print(f"    Gap        : {idea['gap_addressed']}")
            print()
        print(f"{'='*70}\n")
        assert True  # always passes — exists for its output
