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
        "Two modes: ask a general research question and get a synthesised "
        "literature review, or describe your own project and get a curated list "
        "of related papers with specific relevance notes."
    )
    st.markdown("---")
    st.markdown("**Pipeline (both modes)**")
    st.markdown(
        "1. Search — arXiv keyword search  \n"
        "2. Fetch — full text + Semantic Scholar  \n"
        "3. Extract — Claude Haiku (tool-use)  \n"
        "4. Compare — Claude Haiku (tool-use)  \n"
        "5a. *Question mode* — Synthesize (Claude Sonnet)  \n"
        "5b. *Project mode* — Relate to project (Claude Haiku)  \n"
    )
    st.markdown("---")
    st.caption(
        "Demonstration of an agentic research pipeline built with Claude. "
        "Searches publicly available arXiv papers only."
    )

# ---------------------------------------------------------------------------
# Main UI — mode selection
# ---------------------------------------------------------------------------

st.title("Research Assistant")

mode = st.radio(
    "mode",
    ["Ask a Research Question", "Find Related Work for My Project"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("")  # breathing room below the radio

# ---------------------------------------------------------------------------
# Mode-specific input
# ---------------------------------------------------------------------------

query: str = ""
project_description: str = ""

if mode == "Ask a Research Question":
    st.markdown("Ask a research question and receive a synthesised answer drawn from arXiv literature.")
    query = st.text_input(
        label="Research query",
        placeholder="What is the current state of retrieval-augmented generation for large language models?",
        label_visibility="collapsed",
    )
    ready = bool(query.strip())

else:
    st.markdown(
        "Describe your project — a sentence or two, or paste your full abstract. "
        "The assistant finds arXiv papers relevant to your specific work and explains why each one matters."
    )
    project_description = st.text_area(
        label="Project description",
        placeholder=(
            "I'm building a retrieval-augmented generation pipeline using a quantized Llama model "
            "running in Google Colab, for a coursework project. The system retrieves relevant passages "
            "from a small document corpus using FAISS, then passes them as context to the language "
            "model to answer user questions."
        ),
        height=140,
        label_visibility="collapsed",
    )
    ready = bool(project_description.strip())

run = st.button("Run Research", type="primary", disabled=not ready)

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
# Pipeline execution — shared stages (search → fetch → extract → compare)
# ---------------------------------------------------------------------------

if mode == "Ask a Research Question":
    context = ResearchContext(query=query.strip())
else:
    context = ResearchContext(project_description=project_description.strip())

failed_stage: str | None = None

with st.status("Starting pipeline…", expanded=True) as status:
    try:
        if mode == "Find Related Work for My Project":
            status.update(label="Distilling search terms from project description…")
        else:
            status.update(label="Searching arXiv…")
        registry["search"](context)

        if not context.found_papers:
            status.update(label="No papers found.", state="error")
            st.warning("arXiv returned no results. Try rephrasing your input.")
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

        # --- mode-specific final stage ---
        if mode == "Ask a Research Question":
            status.update(label="Writing synthesis…")
            registry["synthesize"](context)
            st.write(f"Synthesis complete ({len(context.synthesis or ''):,} chars).")
        else:
            status.update(label="Mapping papers to your project…")
            registry["relate"](context)
            n_related = len((context.related_work_result or {}).get("related_papers") or [])
            st.write(f"Found **{n_related}** relevant papers.")

        status.update(label="Done.", state="complete")

    except Exception as exc:
        failed_stage = str(exc)
        status.update(label="Pipeline error.", state="error")

if failed_stage:
    st.error(f"**Pipeline error:** {failed_stage}")
    st.stop()

# ---------------------------------------------------------------------------
# Results — Research Question mode
# ---------------------------------------------------------------------------

if mode == "Ask a Research Question":
    st.divider()
    st.subheader("Synthesis")

    if context.synthesis:
        st.markdown(context.synthesis)
    else:
        st.warning("No synthesis was produced.")

    st.divider()

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

# ---------------------------------------------------------------------------
# Results — Find Related Work mode
# ---------------------------------------------------------------------------

else:
    result = context.related_work_result or {}
    related_papers = result.get("related_papers") or []
    summary = result.get("summary") or ""

    st.divider()
    st.subheader("Related Work")

    if summary:
        st.markdown(summary)
    else:
        st.warning("No summary was produced.")

    if related_papers:
        st.markdown("")
        for paper in related_papers:
            arxiv_id = paper.get("arxiv_id", "")
            title = paper.get("title", arxiv_id)
            note = paper.get("relevance_note", "")
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

            st.markdown(f"**[{title}]({arxiv_url})**")
            st.markdown(f"`{arxiv_id}` &nbsp;·&nbsp; {note}")
            st.markdown("---")
    else:
        st.warning("No related papers were identified.")

    st.divider()

    with st.expander(f"All papers searched ({len(context.found_papers)})", expanded=False):
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
