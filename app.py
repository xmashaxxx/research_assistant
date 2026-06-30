"""Streamlit presentation layer for the research assistant pipeline."""

import os

import streamlit as st

import research_assistant  # noqa: F401 — triggers capability self-registration
from research_assistant.context import ResearchContext
from research_assistant.registry import registry

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Research Assistant")
    st.markdown(
        "Enter a research question. The assistant searches arXiv for relevant "
        "papers, fetches their content, extracts structured findings using Claude, "
        "compares them across papers, and writes a synthesised literature review."
    )
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Search — arXiv keyword search  \n"
        "2. Fetch — full text + Semantic Scholar  \n"
        "3. Extract — Claude Haiku (tool-use)  \n"
        "4. Compare — Claude Haiku (tool-use)  \n"
        "5. Synthesize — Claude Sonnet  \n"
    )
    st.markdown("---")
    st.caption(
        "Demonstration of an agentic research pipeline built with Claude. "
        "Searches publicly available arXiv papers only."
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Research Assistant")
st.markdown("Ask a research question and receive a synthesised answer drawn from arXiv literature.")

query = st.text_input(
    label="Research query",
    placeholder="What is the current state of retrieval-augmented generation for large language models?",
    label_visibility="collapsed",
)

run = st.button("Run Research", type="primary", disabled=not bool(query.strip()))

if not run:
    st.stop()

# ---------------------------------------------------------------------------
# Pre-flight: API key check
# ---------------------------------------------------------------------------

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "**ANTHROPIC_API_KEY is not set.**\n\n"
        "The extract, compare, and synthesize stages require the Anthropic API. "
        "Set the environment variable and restart:\n\n"
        "```\nexport ANTHROPIC_API_KEY=sk-ant-...\nstreamlit run app.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution with live status updates
# ---------------------------------------------------------------------------

context = ResearchContext(query=query.strip())
failed_stage: str | None = None

with st.status("Starting pipeline…", expanded=True) as status:
    try:
        status.update(label="Searching arXiv…")
        registry["search"](context)

        if not context.found_papers:
            status.update(label="No papers found.", state="error")
            st.warning("arXiv returned no results for this query. Try rephrasing.")
            st.stop()

        st.write(f"Found **{len(context.found_papers)}** papers.")

        status.update(label=f"Fetching {len(context.found_papers)} papers…")
        registry["fetch"](context)
        st.write(f"Fetched text for **{len(context.summaries)}** papers.")

        status.update(label="Extracting key information…")
        registry["extract"](context)
        st.write(f"Extracted structured fields from **{len(context.summaries)}** papers.")

        status.update(label="Comparing findings across papers…")
        registry["compare"](context)
        n_agreements = len(context.comparisons.get("agreements") or [])
        n_disagreements = len(context.comparisons.get("disagreements") or [])
        st.write(f"Identified **{n_agreements}** agreements and **{n_disagreements}** disagreements.")

        status.update(label="Writing synthesis…")
        registry["synthesize"](context)
        st.write(f"Synthesis complete ({len(context.synthesis or ''):,} chars).")

        status.update(label="Done.", state="complete")

    except Exception as exc:
        failed_stage = str(exc)
        status.update(label="Pipeline error.", state="error")

if failed_stage:
    st.error(f"**Pipeline error:** {failed_stage}")
    st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Synthesis")

if context.synthesis:
    st.markdown(context.synthesis)
else:
    st.warning("No synthesis was produced.")

st.divider()

# Papers reviewed
with st.expander(f"Papers reviewed ({len(context.found_papers)})", expanded=False):
    for paper in context.found_papers:
        col_title, col_meta = st.columns([3, 1])
        with col_title:
            st.markdown(f"**[{paper.title}]({paper.arxiv_url})**")
            author_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                author_str += f" et al. ({len(paper.authors)})"
            st.caption(author_str)
        with col_meta:
            st.caption(paper.published[:10])
            st.caption(paper.arxiv_id)
            if paper.citation_count is not None:
                st.caption(f"{paper.citation_count:,} citations")
        st.markdown("---")

# Cross-paper comparison
if context.comparisons:
    with st.expander("Cross-paper comparison", expanded=False):
        agreements = context.comparisons.get("agreements") or []
        if agreements:
            st.markdown("**Where the papers agree**")
            for ag in agreements:
                st.markdown(f"- {ag}")

        disagreements = context.comparisons.get("disagreements") or []
        if disagreements:
            st.markdown("**Where the papers differ**")
            for d in disagreements:
                st.markdown(f"*{d.get('point', '')}*")
                for paper_id, stance in (d.get("positions") or {}).items():
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{paper_id}` — {stance}")

        unique = context.comparisons.get("unique_contributions") or {}
        if unique:
            st.markdown("**Unique contributions**")
            for paper_id, contrib in unique.items():
                st.markdown(f"- `{paper_id}` — {contrib}")
