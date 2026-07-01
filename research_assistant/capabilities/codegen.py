"""Experiment code generation capability — produces a runnable .py script and .ipynb notebook."""

from __future__ import annotations

import json
import os
import re

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.registry import register

_MODEL = "claude-sonnet-4-6"

_SYSTEM_ENGINEER = (
    "You are an expert ML research engineer who writes clean, reproducible experiment code. "
    "Your implementations are complete and runnable — no pseudocode, no placeholder comments "
    "like 'fill this in'. Every function body is fully implemented. You prefer clarity over "
    "cleverness and always use the minimum viable set of dependencies."
)

_SYSTEM_NOTEBOOK = (
    "You are an expert ML research engineer who converts Python scripts into well-structured "
    "Jupyter notebooks. You produce valid .ipynb JSON that opens directly in Jupyter. "
    "You add clear markdown cells that explain what each section does. "
    "Return ONLY the raw JSON object — no prose, no markdown code fences, nothing outside the JSON."
)


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the codegen capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_script_prompt(experiment_idea: dict, query: str) -> str:
    title = experiment_idea.get("title", "")
    hypothesis = experiment_idea.get("hypothesis", "")
    method = experiment_idea.get("method", "")
    gap_addressed = experiment_idea.get("gap_addressed", "")
    difficulty = experiment_idea.get("difficulty", "medium")

    return f"""Write a complete, runnable Python experiment script for the following ML research experiment.

Research context: {query}

Experiment:
  Title:         {title}
  Hypothesis:    {hypothesis}
  Method:        {method}
  Gap addressed: {gap_addressed}
  Difficulty:    {difficulty}

Requirements for the script:
1. Start with a header comment block (lines beginning with #) that includes exactly these two lines
   so they can be parsed by downstream tooling:
       # Requirements: <comma-separated pip packages, e.g.: torch, transformers, datasets, faiss-cpu>
       # Estimated Runtime: <realistic wall-clock estimate, e.g.: 4-8 hours on a single A100 GPU>
2. All imports come immediately after the header block.
3. A main() function with argparse covering the key hyperparameters: dataset name, model name,
   chunk size, batch size, random seed, and output directory.
4. Only use these libraries (no others): torch, transformers, datasets, faiss-cpu,
   sentence-transformers, evaluate, numpy, pandas, tqdm — plus Python stdlib
   (argparse, json, os, pathlib, logging, time).
5. Inline comments on every non-obvious step.
6. A results section at the end of main() that prints a formatted summary table.
7. End with:
       if __name__ == "__main__":
           main()
8. Every function must be fully implemented — no pseudocode, no "# TODO" stubs.

Return ONLY the Python script. No prose before or after it, no markdown code fences."""


def _build_notebook_prompt(python_script: str, title: str) -> str:
    return f"""Convert the Python experiment script below into a Jupyter notebook (.ipynb format).

Rules:
- First cell: a markdown cell with "{title}" as an H1 heading and a 1–2 sentence description.
- Split the script into logical sections, each preceded by a markdown cell describing what it does.
  Suggested sections: Setup & Imports | Configuration & Arguments | Data Loading |
  Model & Retriever Setup | Experiment Loop | Results & Analysis
- Preserve all code exactly — do not simplify or omit any logic.
- Use nbformat 4 / nbformat_minor 5.
- Each code cell: cell_type "code", source as a JSON array of strings (one per line, each ending
  in \\n except the last line of the cell), metadata {{}}, outputs [], execution_count null.
- Each markdown cell: cell_type "markdown", source as a JSON array of strings, metadata {{}}.
- Return ONLY the raw JSON object. No prose, no code fences, nothing outside the JSON.

Python script:
{python_script}"""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if Claude wrapped its output in them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _fallback_notebook(python_script: str, title: str) -> str:
    """Build a minimal valid .ipynb when Claude's notebook conversion produces invalid JSON.

    Wraps the entire Python script in a single code cell so the download
    still works even if the pretty notebook conversion failed.
    """
    lines = python_script.splitlines(keepends=True)
    # Notebook source arrays: every line ends with \n except the last
    source = [line if line.endswith("\n") else line for line in lines]
    if source and source[-1].endswith("\n"):
        source[-1] = source[-1][:-1]

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "Auto-generated experiment script (notebook conversion failed)."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)


def _extract_metadata(script: str) -> tuple[str, list[str]]:
    """Parse # Requirements and # Estimated Runtime lines from the script header.

    Scans the first 30 lines (comment block at top of file).
    Returns (estimated_runtime, requirements_list).
    """
    estimated_runtime = "unknown"
    requirements: list[str] = []

    for line in script.splitlines()[:30]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        m = re.match(r"#\s*Requirements?\s*:\s*(.+)", stripped, re.IGNORECASE)
        if m:
            requirements = [r.strip() for r in m.group(1).split(",") if r.strip()]
            continue
        m = re.match(r"#\s*Estimated\s+Runtime\s*:\s*(.+)", stripped, re.IGNORECASE)
        if m:
            estimated_runtime = m.group(1).strip()

    return estimated_runtime, requirements


def generate_experiment_code(experiment_idea: dict, query: str) -> dict:
    """Generate a complete Python script and Jupyter notebook for one experiment idea.

    Makes two sequential Claude Sonnet calls:
      1. Generate the .py script (plain text, full generation capability).
      2. Convert the script to a .ipynb notebook (restructured with markdown cells).

    Args:
        experiment_idea: Dict with keys title, hypothesis, method, gap_addressed, difficulty.
        query:           Original research question — gives Claude context for the implementation.

    Returns a dict with keys:
        python_script     — complete .py file content as a string
        notebook          — complete .ipynb JSON as a string
        experiment_title  — title from the input idea
        estimated_runtime — Claude's estimate of wall-clock run time
        requirements      — list of pip package names needed
    """
    client = _get_client()
    title = experiment_idea.get("title", "experiment")

    # --- Call 1: generate Python script ---
    print(f"[codegen] Generating Python script for: {title!r}")
    script_resp = client.messages.create(
        model=_MODEL,
        max_tokens=8192,
        system=_SYSTEM_ENGINEER,
        messages=[{"role": "user", "content": _build_script_prompt(experiment_idea, query)}],
    )
    python_script = _strip_fences(script_resp.content[0].text)
    print(f"[codegen] Script: {len(python_script):,} chars")

    estimated_runtime, requirements = _extract_metadata(python_script)

    # --- Call 2: convert to Jupyter notebook ---
    # max_tokens=16384: notebook JSON is larger than the script (JSON encoding + markdown cells),
    # so 8192 tokens (~27 K chars) truncates for scripts over ~15 K chars.
    print("[codegen] Converting to Jupyter notebook…")
    nb_resp = client.messages.create(
        model=_MODEL,
        max_tokens=16384,
        system=_SYSTEM_NOTEBOOK,
        messages=[{"role": "user", "content": _build_notebook_prompt(python_script, title)}],
    )
    notebook_raw = _strip_fences(nb_resp.content[0].text)

    try:
        json.loads(notebook_raw)
        print(f"[codegen] Notebook: {len(notebook_raw):,} chars (valid JSON)")
    except json.JSONDecodeError as exc:
        print(
            f"[codegen] WARNING: notebook conversion returned invalid JSON "
            f"({exc}); using fallback single-cell notebook."
        )
        notebook_raw = _fallback_notebook(python_script, title)

    return {
        "python_script": python_script,
        "notebook": notebook_raw,
        "experiment_title": title,
        "estimated_runtime": estimated_runtime,
        "requirements": requirements,
    }


@register("codegen")
def codegen_capability(context: ResearchContext) -> None:
    """User-triggered capability: generate code for the most promising experiment idea.

    NOT part of the automatic pipeline — must be called explicitly.
    Reads: context.experiment_ideas, context.query
    Writes: context.generated_code
    """
    ideas_data = context.experiment_ideas or {}
    ideas_list = ideas_data.get("experiment_ideas") or []
    if not ideas_list:
        print("[codegen] No experiment ideas in context — skipping.")
        return

    most_promising = ideas_data.get("most_promising", 0)
    idea = ideas_list[most_promising]

    print(f"[codegen] Generating code for: {idea.get('title', '')!r}")
    result = generate_experiment_code(experiment_idea=idea, query=context.query)
    context.generated_code = result
    print(
        f"[codegen] Done — {len(result['python_script']):,} chars Python, "
        f"{len(result['notebook']):,} chars notebook JSON."
    )
