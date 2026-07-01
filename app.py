"""Streamlit presentation layer for the research assistant pipeline."""

import io
import os
import re

import streamlit as st

import research_assistant  # noqa: F401 — triggers capability self-registration
from research_assistant.capabilities.codegen import generate_experiment_code
from research_assistant.context import ResearchContext
from research_assistant.guardrails import (
    GuardrailError,
    check_synthesis_grounding,
    validate_query,
    validate_search_results,
    validate_stage_output,
)
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
        "1. Search — arXiv + Semantic Scholar  \n"
        "2. Fetch — full text + Papers with Code  \n"
        "3. Extract — Claude Haiku (tool-use)  \n"
        "4. Compare — Claude Haiku (tool-use)  \n"
        "5a. *Question mode* — Synthesize (Claude Sonnet)  \n"
        "5b. *Question mode* — Ideate (Claude Sonnet)  \n"
        "5c. *Project mode* — Relate to project (Claude Haiku)  \n"
        "6. *Optional* — Generate code (Claude Sonnet, user-triggered)  \n"
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
        "Describe your project — paste text below, or upload a PDF (abstract, proposal, or draft). "
        "If you provide both, the PDF takes precedence."
    )

    _PDF_CHAR_LIMIT = 15_000

    uploaded_pdf = st.file_uploader("Upload your project description (PDF)", type=["pdf"])
    pdf_text: str = ""

    if uploaded_pdf is not None:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(uploaded_pdf.read()))
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            raw = "\n".join(pages_text).strip()
            if not raw:
                st.error(
                    "Couldn't extract text from this PDF — it may be a scanned image. "
                    "Try pasting the text instead."
                )
            else:
                truncated = len(raw) > _PDF_CHAR_LIMIT
                pdf_text = raw[:_PDF_CHAR_LIMIT]
                msg = f"Extracted {len(pdf_text):,} characters from **{uploaded_pdf.name}**"
                if truncated:
                    msg += f" (truncated from {len(raw):,} — first {_PDF_CHAR_LIMIT:,} chars used)"
                st.success(msg)
        except Exception as exc:
            st.error(
                f"Couldn't read this PDF ({exc}). "
                "It may be corrupted or encrypted. Try pasting the text instead."
            )

    typed_text = st.text_area(
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

    # PDF takes precedence when both are provided
    project_description = pdf_text if pdf_text else typed_text
    ready = bool(project_description.strip())

run = st.button("Run Research", type="primary", disabled=not ready)

# ---------------------------------------------------------------------------
# Gate: pass through if pipeline just ran OR results are cached from last run
# ---------------------------------------------------------------------------

if not run and "pipeline_results" not in st.session_state:
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution — only when "Run Research" is clicked
# ---------------------------------------------------------------------------

if run:
    # Clear stale results so the code generation section doesn't show old output
    for _k in ("pipeline_results", "generated_code"):
        st.session_state.pop(_k, None)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "**ANTHROPIC_API_KEY is not set.**\n\n"
            "The extract, compare, and synthesize stages require the Anthropic API. "
            "Set the environment variable and restart:\n\n"
            "```\nexport ANTHROPIC_API_KEY=sk-ant-...\nstreamlit run app.py\n```"
        )
        st.stop()

    if mode == "Ask a Research Question":
        context = ResearchContext(query=query.strip())
    else:
        context = ResearchContext(project_description=project_description.strip())

    # --- Query / input validation ---
    _input_text = query.strip() if mode == "Ask a Research Question" else project_description.strip()
    _qv = validate_query(_input_text)
    if _qv["severity"] == "block":
        st.error(f"**Input blocked:** {_qv['issue']}")
        st.stop()
    if _qv["severity"] == "warn":
        st.warning(f"**Query notice:** {_qv['issue']}")

    _failed_stage: str | None = None

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

            _sr = validate_search_results(
                _input_text, context.found_papers
            )
            if _sr["severity"] != "ok":
                st.warning(_sr["issue"])

            st.write(f"Found **{len(context.found_papers)}** papers.")

            status.update(label=f"Fetching {len(context.found_papers)} papers…")
            registry["fetch"](context)
            st.write(f"Fetched text for **{len(context.summaries)}** papers.")

            status.update(label="Extracting key information…")
            registry["extract"](context)
            st.write(f"Extracted structured fields from **{len(context.summaries)}** papers.")
            _ev = validate_stage_output("extract", context.summaries, context)
            for _w in _ev["warnings"]:
                st.warning(_w)

            status.update(label="Comparing findings across papers…")
            registry["compare"](context)
            n_agreements = len(context.comparisons.get("agreements") or [])
            n_disagreements = len(context.comparisons.get("disagreements") or [])
            st.write(f"Identified **{n_agreements}** agreements and **{n_disagreements}** disagreements.")
            _cv = validate_stage_output("compare", context.comparisons, context)
            for _w in _cv["warnings"]:
                st.warning(_w)

            if mode == "Ask a Research Question":
                status.update(label="Writing synthesis…")
                registry["synthesize"](context)
                st.write(f"Synthesis complete ({len(context.synthesis or ''):,} chars).")
                _sv = validate_stage_output("synthesize", {"synthesis": context.synthesis}, context)
                for _w in _sv["warnings"]:
                    st.warning(_w)

                status.update(label="Checking synthesis grounding…")
                context.grounding_check = check_synthesis_grounding(
                    context.synthesis or "", context.summaries
                )
                _gc = context.grounding_check
                if _gc.get("warning"):
                    st.warning(f"**Grounding notice:** {_gc['warning']}")
                else:
                    st.write(
                        f"Grounding check: **{_gc.get('confidence', 'unknown')}** confidence — "
                        f"{len(_gc.get('grounded_claims') or [])} claims verified."
                    )

                status.update(label="Generating experiment ideas…")
                registry["ideate"](context)
                n_ideas = len((context.experiment_ideas or {}).get("experiment_ideas") or [])
                st.write(f"Generated **{n_ideas}** experiment ideas.")
                _iv = validate_stage_output("ideate", context.experiment_ideas or {}, context)
                for _w in _iv["warnings"]:
                    st.warning(_w)
            else:
                status.update(label="Mapping papers to your project…")
                registry["relate"](context)
                n_related = len((context.related_work_result or {}).get("related_papers") or [])
                st.write(f"Found **{n_related}** relevant papers.")

            status.update(label="Done.", state="complete")

        except GuardrailError as exc:
            _failed_stage = str(exc)
            status.update(label="Blocked by guardrail.", state="error")
        except Exception as exc:
            _failed_stage = str(exc)
            status.update(label="Pipeline error.", state="error")

    if _failed_stage:
        st.error(f"**Pipeline error:** {_failed_stage}")
        st.stop()

    # Cache pipeline outputs so results persist across widget interactions
    st.session_state["pipeline_results"] = {
        "mode": mode,
        "query": context.query,
        "synthesis": context.synthesis,
        "comparisons": context.comparisons,
        "experiment_ideas": context.experiment_ideas,
        "found_papers": context.found_papers,
        "related_work_result": context.related_work_result,
        "grounding_check": context.grounding_check,
    }

# ---------------------------------------------------------------------------
# Restore from session_state (works for both fresh run and widget re-renders)
# ---------------------------------------------------------------------------

pr = st.session_state["pipeline_results"]
pr_mode = pr["mode"]

# ---------------------------------------------------------------------------
# Results — Research Question mode
# ---------------------------------------------------------------------------

if pr_mode == "Ask a Research Question":
    st.divider()
    st.subheader("Synthesis")

    synthesis = pr.get("synthesis")
    if synthesis:
        st.markdown(synthesis)
    else:
        st.warning("No synthesis was produced.")

    # --- Grounding check ---
    gc = pr.get("grounding_check")
    if gc:
        grounded = gc.get("grounded_claims") or []
        ungrounded = gc.get("ungrounded_claims") or []
        confidence = gc.get("confidence", "unknown")
        gc_warning = gc.get("warning")
        with st.expander(
            f"Grounding check — {confidence} confidence"
            + (f" ({len(ungrounded)} unverified claim(s))" if ungrounded else ""),
            expanded=bool(ungrounded),
        ):
            if gc_warning:
                st.warning(gc_warning)
            if grounded:
                st.markdown("**Verified claims** (traceable to source extractions)")
                for c in grounded:
                    st.markdown(f"- {c}")
            if ungrounded:
                st.markdown("**Unverified claims** (could not be traced to source extractions)")
                for c in ungrounded:
                    st.markdown(f"- {c}")
            st.caption(
                "This is a probabilistic check — a claim may be correct even if absent from "
                "the extracted snippets."
            )

    # --- Experiment Ideas ---
    ideas_data = pr.get("experiment_ideas") or {}
    ideas_list = ideas_data.get("experiment_ideas") or []
    if ideas_list:
        st.divider()
        st.subheader("Experiment Ideas")
        most_promising = ideas_data.get("most_promising", 0)
        reasoning = ideas_data.get("reasoning", "")
        if reasoning:
            st.markdown(f"**Most promising:** {reasoning}")
        st.markdown("")
        _DIFF_COLOR = {"low": "green", "medium": "orange", "high": "red"}
        for i, idea in enumerate(ideas_list):
            diff = idea.get("difficulty", "medium")
            color = _DIFF_COLOR.get(diff, "grey")
            label = f":{color}[{diff}]"
            star = " ⭐" if i == most_promising else ""
            with st.expander(
                f"**{idea.get('title', f'Idea {i+1}')}**{star} &nbsp; {label}",
                expanded=(i == most_promising),
            ):
                st.markdown(f"**Hypothesis:** {idea.get('hypothesis', '')}")
                st.markdown(f"**Method:** {idea.get('method', '')}")
                st.markdown(f"**Gap addressed:** {idea.get('gap_addressed', '')}")
                st.caption(f"Difficulty: {diff}")

        # --- Code generation UI ---
        st.markdown("---")
        titles = [idea["title"] for idea in ideas_list]
        selected_title = st.selectbox(
            "Generate code for this experiment:",
            ["— select —"] + titles,
            key="codegen_select",
        )
        gen_ready = selected_title != "— select —"
        if st.button("Generate Code", disabled=not gen_ready, key="codegen_btn"):
            idea_idx = titles.index(selected_title)
            idea = ideas_list[idea_idx]
            with st.spinner(f"Generating code for '{selected_title}'…"):
                code_result = generate_experiment_code(idea, pr["query"])
                st.session_state["generated_code"] = code_result

        if "generated_code" in st.session_state:
            code_result = st.session_state["generated_code"]
            st.divider()
            code_title = code_result.get("experiment_title", "Experiment")
            st.subheader(f"Generated Code — {code_title}")

            runtime = code_result.get("estimated_runtime", "")
            reqs = code_result.get("requirements") or []
            if runtime or reqs:
                meta_parts = []
                if runtime:
                    meta_parts.append(f"**Estimated runtime:** {runtime}")
                if reqs:
                    meta_parts.append(f"**Requirements:** `pip install {' '.join(reqs)}`")
                st.markdown("  \n".join(meta_parts))

            script = code_result["python_script"]
            notebook = code_result["notebook"]
            safe_name = re.sub(r"[^a-z0-9]+", "_", code_title.lower()).strip("_")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇ Download .py",
                    data=script,
                    file_name=f"{safe_name}.py",
                    mime="text/plain",
                    key="dl_py",
                )
            with col2:
                st.download_button(
                    "⬇ Download .ipynb",
                    data=notebook,
                    file_name=f"{safe_name}.ipynb",
                    mime="application/json",
                    key="dl_nb",
                )

            st.code(script, language="python")

    st.divider()

    found_papers = pr.get("found_papers") or []
    with st.expander(f"Papers reviewed ({len(found_papers)})", expanded=False):
        for paper in found_papers:
            col_title, col_meta = st.columns([3, 1])
            with col_title:
                st.markdown(f"**[{paper.title}]({paper.arxiv_url})**")
                author_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    author_str += f" et al. ({len(paper.authors)})"
                st.caption(author_str)
                if paper.code_url:
                    st.markdown(f"[View Code]({paper.code_url})")
                if paper.benchmark_results:
                    for b in paper.benchmark_results:
                        st.caption(
                            f"{b.get('task','')} / {b.get('dataset','')}: "
                            f"{b.get('metric','')} = {b.get('score','')}"
                        )
            with col_meta:
                st.caption(paper.published[:10])
                st.caption(paper.arxiv_id)
                if paper.citation_count is not None:
                    st.caption(f"{paper.citation_count:,} citations")
            st.markdown("---")

    comparisons = pr.get("comparisons") or {}
    if comparisons:
        with st.expander("Cross-paper comparison", expanded=False):
            agreements = comparisons.get("agreements") or []
            if agreements:
                st.markdown("**Where the papers agree**")
                for ag in agreements:
                    st.markdown(f"- {ag}")

            disagreements = comparisons.get("disagreements") or []
            if disagreements:
                st.markdown("**Where the papers differ**")
                for d in disagreements:
                    st.markdown(f"*{d.get('point', '')}*")
                    for paper_id, stance in (d.get("positions") or {}).items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{paper_id}` — {stance}")

            unique = comparisons.get("unique_contributions") or {}
            if unique:
                st.markdown("**Unique contributions**")
                for paper_id, contrib in unique.items():
                    st.markdown(f"- `{paper_id}` — {contrib}")

# ---------------------------------------------------------------------------
# Results — Find Related Work mode
# ---------------------------------------------------------------------------

else:
    result = pr.get("related_work_result") or {}
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

    found_papers = pr.get("found_papers") or []
    with st.expander(f"All papers searched ({len(found_papers)})", expanded=False):
        for paper in found_papers:
            col_title, col_meta = st.columns([3, 1])
            with col_title:
                st.markdown(f"**[{paper.title}]({paper.arxiv_url})**")
                author_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    author_str += f" et al. ({len(paper.authors)})"
                st.caption(author_str)
                if paper.code_url:
                    st.markdown(f"[View Code]({paper.code_url})")
                if paper.benchmark_results:
                    for b in paper.benchmark_results:
                        st.caption(
                            f"{b.get('task','')} / {b.get('dataset','')}: "
                            f"{b.get('metric','')} = {b.get('score','')}"
                        )
            with col_meta:
                st.caption(paper.published[:10])
                st.caption(paper.arxiv_id)
                if paper.citation_count is not None:
                    st.caption(f"{paper.citation_count:,} citations")
            st.markdown("---")

    comparisons = pr.get("comparisons") or {}
    if comparisons:
        with st.expander("Cross-paper comparison", expanded=False):
            agreements = comparisons.get("agreements") or []
            if agreements:
                st.markdown("**Where the papers agree**")
                for ag in agreements:
                    st.markdown(f"- {ag}")

            disagreements = comparisons.get("disagreements") or []
            if disagreements:
                st.markdown("**Where the papers differ**")
                for d in disagreements:
                    st.markdown(f"*{d.get('point', '')}*")
                    for paper_id, stance in (d.get("positions") or {}).items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{paper_id}` — {stance}")

            unique = comparisons.get("unique_contributions") or {}
            if unique:
                st.markdown("**Unique contributions**")
                for paper_id, contrib in unique.items():
                    st.markdown(f"- `{paper_id}` — {contrib}")
