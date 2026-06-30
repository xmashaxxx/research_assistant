"""
eval/run_evaluation.py

Runs the full research pipeline against the canonical RAG evaluation query
and saves all intermediate and final outputs to eval/results/.

Usage:
    python eval/run_evaluation.py

Requires ANTHROPIC_API_KEY in the environment.
Results are saved to eval/results/rag_state_of_field_run.json (gitignored).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- ensure repo root is on sys.path when run directly ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import research_assistant  # noqa: F401 — triggers capability self-registration
from research_assistant.capabilities.fetch_paper import fetch_papers
from research_assistant.capabilities.extract import extract_paper
from research_assistant.capabilities.compare import compare_extractions
from research_assistant.capabilities.synthesize import synthesize
from research_assistant.capabilities.search import search_papers
from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY = "What is the current state of retrieval-augmented generation for large language models?"
GAO_SURVEY_ID = "2312.10997"   # The paper we are being evaluated against
RESULTS_DIR = REPO_ROOT / "eval" / "results"
OUTPUT_FILE = RESULTS_DIR / "rag_state_of_field_run.json"

DIVIDER = "=" * 70


def _banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _record_to_dict(paper: PaperRecord) -> dict:
    d = dataclasses.asdict(paper)
    # Truncate full_text in JSON — it's large and already in raw_text via summaries
    if d.get("full_text"):
        d["full_text"] = d["full_text"][:500] + "  [truncated in JSON]"
    return d


def run() -> None:
    # --- pre-flight ---
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    _banner(f"RESEARCH ASSISTANT EVALUATION RUN")
    print(f"Query  : {QUERY}")
    print(f"Started: {started_at}")

    # -----------------------------------------------------------------------
    # Stage 1: Search
    # -----------------------------------------------------------------------

    _banner("STAGE 1 — SEARCH")
    ctx = ResearchContext(query=QUERY)
    search_papers(ctx)

    papers = ctx.found_papers
    print(f"\nFound {len(papers)} papers:\n")

    gao_in_results = False
    for i, p in enumerate(papers, 1):
        flag = ""
        if p.arxiv_id == GAO_SURVEY_ID or GAO_SURVEY_ID in p.arxiv_id:
            flag = "  ⚠️  THIS IS THE GAO ET AL. SURVEY (evaluation target)"
            gao_in_results = True
        print(f"  {i:2}. [{p.arxiv_id}] {p.title}{flag}")
        print(f"       Authors : {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"       Published: {p.published[:10]}")
        print()

    if gao_in_results:
        print("⚠️  EVALUATION NOTE: arXiv 2312.10997 (Gao et al. 'Retrieval-Augmented")
        print("   Generation for Large Language Models: A Survey') appeared in the")
        print("   search results. This is the paper we are evaluating against.")
        print("   The agent is potentially summarising the evaluation target itself.")
        print("   For a clean evaluation, consider excluding this paper from the")
        print("   fetched set and re-running, then comparing agent output to Gao et al.")
    else:
        print(f"✓  arXiv {GAO_SURVEY_ID} (Gao et al. survey) did NOT appear in search")
        print(f"   results — evaluation corpus is independent of the reference paper.")

    # -----------------------------------------------------------------------
    # Stage 2: Fetch
    # -----------------------------------------------------------------------

    _banner("STAGE 2 — FETCH")
    ids = [p.arxiv_id for p in papers]
    fetched = fetch_papers(ids)
    # Rebuild found_papers from fetch output (fetch returns fresh PaperRecords)
    ctx.found_papers = fetched
    for p in fetched:
        ctx.summaries[p.arxiv_id] = {
            "raw_text": p.full_text or p.abstract,
            "metadata": {
                "citation_count": p.citation_count,
                "references": p.references,
            },
        }
    print(f"\nFetched {len(fetched)} papers.")
    for p in fetched:
        src = f"HTML ({len(p.full_text):,} chars)" if p.full_text else "abstract only"
        cites = f", {p.citation_count:,} citations" if p.citation_count is not None else ""
        print(f"  [{p.arxiv_id}] {src}{cites}")

    # -----------------------------------------------------------------------
    # Stage 3: Extract
    # -----------------------------------------------------------------------

    _banner("STAGE 3 — EXTRACT")
    print()
    for p in fetched:
        print(f"  Extracting [{p.arxiv_id}] {p.title[:60]}...")
        extraction = extract_paper(p)
        ctx.summaries[p.arxiv_id]["extraction"] = extraction
        print(f"    research_question: {extraction.get('research_question', '')[:80]}")
        print(f"    method           : {extraction.get('method', '')[:80]}")
        print(f"    key_results      : {extraction.get('key_results', '')[:80]}")
        print()

    # -----------------------------------------------------------------------
    # Stage 4: Compare
    # -----------------------------------------------------------------------

    _banner("STAGE 4 — COMPARE")
    extractions = {
        aid: s["extraction"]
        for aid, s in ctx.summaries.items()
        if "extraction" in s
    }
    titles = {p.arxiv_id: p.title for p in fetched}
    comparisons = compare_extractions(extractions, titles)
    ctx.comparisons = comparisons

    print(f"\nAgreements ({len(comparisons.get('agreements', []))}):")
    for ag in comparisons.get("agreements", []):
        print(f"  - {ag}")

    print(f"\nDisagreements ({len(comparisons.get('disagreements', []))}):")
    for d in comparisons.get("disagreements", []):
        print(f"  Topic: {d.get('point', '')}")
        for pid, stance in (d.get("positions") or {}).items():
            print(f"    [{pid}] {stance[:100]}")

    print(f"\nUnique contributions ({len(comparisons.get('unique_contributions', {}))}):")
    for pid, contrib in (comparisons.get("unique_contributions") or {}).items():
        print(f"  [{pid}] {contrib[:100]}")

    # -----------------------------------------------------------------------
    # Stage 5: Synthesize
    # -----------------------------------------------------------------------

    _banner("STAGE 5 — SYNTHESIZE")
    synthesis_text = synthesize(QUERY, ctx.summaries, ctx.comparisons)
    ctx.synthesis = synthesis_text

    print(f"\n{synthesis_text}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    _banner("SAVING RESULTS")

    output = {
        "run_metadata": {
            "query": QUERY,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "gao_survey_in_search_results": gao_in_results,
            "paper_count": len(fetched),
        },
        "found_papers": [_record_to_dict(p) for p in fetched],
        "extractions": {
            aid: s.get("extraction")
            for aid, s in ctx.summaries.items()
        },
        "comparisons": ctx.comparisons,
        "synthesis": ctx.synthesis,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"File size       : {OUTPUT_FILE.stat().st_size:,} bytes")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    _banner("RUN COMPLETE")
    print(f"  Papers found     : {len(fetched)}")
    print(f"  Papers extracted : {len(extractions)}")
    print(f"  Agreements       : {len(comparisons.get('agreements', []))}")
    print(f"  Disagreements    : {len(comparisons.get('disagreements', []))}")
    print(f"  Synthesis length : {len(synthesis_text):,} chars")
    print(f"  Gao survey found : {'YES — see evaluation note above' if gao_in_results else 'No'}")
    print()


if __name__ == "__main__":
    run()
