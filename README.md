# research_assistant

An agentic research assistant that takes a natural-language research question, searches arXiv and Semantic Scholar for relevant papers, reads and extracts structured findings from each, compares them across papers, synthesizes a literature review that directly answers the question, proposes concrete follow-on experiment ideas, and can generate a complete runnable Python implementation of any selected experiment on demand. A second mode accepts a project description — typed or uploaded as a PDF — and returns an annotated list of related papers with a positioning summary. The technical thesis is that a multi-step pipeline — search, extract, compare, synthesize, ideate, codegen as separable capabilities — produces a more accurate and traceable result than single-pass summarization: each stage operates on structured output from the one before it, so errors are localized and every output is grounded in verified intermediate representations rather than raw text.
Watch the demo ----> https://youtu.be/ypGdeR_sSE8
---

## Modes

**Ask a Research Question** — the original mode. Provide a natural-language question; the pipeline searches the literature, synthesizes an answer grounded in extracted findings, proposes 3–5 concrete follow-on experiments, and optionally generates a complete runnable implementation for any of them.

**Find Related Work for My Project** — describe your project in the text box, or upload a PDF (abstract, proposal, or draft paper). The pipeline distills search keywords from your description, retrieves the nearest literature, and returns an annotated list of related papers with a one-paragraph positioning summary explaining how each paper relates to your specific project.

---

## Pipeline

**Stage 1 — Search** (`research_assistant/capabilities/search.py`)

Queries arXiv and Semantic Scholar in parallel, merges the results, and deduplicates by arXiv ID and normalized title. Returns up to ten papers ranked by relevance. Each paper is parsed into a `PaperRecord` carrying the arXiv ID, title, authors, abstract, published date, categories, and URLs. Papers from Semantic Scholar are only retained when they carry a verifiable arXiv ID — the fetch stage requires one to retrieve full text. In "Find Related Work" mode, search keywords are first distilled from the project description by a brief Claude Haiku call before querying both sources.

**Stage 2 — Fetch** (`research_assistant/capabilities/fetch_paper.py`)

For each paper, fetches full text from the arXiv HTML endpoint and enriches metadata from three external sources: Semantic Scholar (citation count, reference list) and Papers with Code (benchmark results, top code repository by star count). Full text is capped at 50,000 characters; where the HTML endpoint is unavailable, the abstract falls back as the text source. Fetch retries transient arXiv errors with exponential backoff (5s, 15s). Papers with Code enrichment is best-effort — any network or rate-limit failure returns `(None, None)` and the pipeline continues unaffected.

**Stage 3 — Extract** (`research_assistant/capabilities/extract.py`)

Calls Claude Haiku (`claude-haiku-4-5-20251001`) with tool-use forced via `tool_choice` to extract six structured fields from each paper: research question, method, dataset, key results, limitations, and related work. Forcing a named tool means the output is always a typed dict — no parsing, no markdown stripping. The extraction schema is defined in `research_assistant/schemas/general_cs_paper.json` and is designed to be swapped for domain-specific schemas (clinical PICO, social science, economics) without touching any other code.

**Stage 4 — Compare** (`research_assistant/capabilities/compare.py`)

Calls Claude Haiku again, this time with all per-paper extractions as input, and produces three structured outputs: agreements (claims that appear across multiple papers), disagreements (positions that conflict, with per-paper stances), and unique contributions (findings specific to a single paper). This stage runs on structured extractions, not raw text — it is comparing semantics, not prose.

**Stage 5 — Synthesize** (`research_assistant/capabilities/synthesize.py`) *(Ask a Research Question mode only)*

Calls Claude Sonnet (`claude-sonnet-4-6`) with the per-paper extractions and the comparison summary, and writes a flowing prose literature review that answers the original question. Sonnet is used here rather than Haiku because this is the user-facing output and writing quality is consequential. No tool-use — the instruction is for natural prose, 3–5 paragraphs, with papers cited by title.

**Stage 5 (alt) — Relate** (`research_assistant/capabilities/relate_to_project.py`) *(Find Related Work mode only)*

Calls Claude Haiku with forced tool-use to map the retrieved papers against the user's project description. Produces an annotated list of related papers — each with a specific relevance note explaining the connection to the project — and a one-paragraph positioning summary. Duplicate arXiv IDs in the model's output are deduplicated before returning.

**Stage 6 — Ideate** (`research_assistant/capabilities/ideate.py`) *(Ask a Research Question mode only)*

Calls Claude Sonnet with the synthesis and the structured gap/disagreement data from the compare stage, and proposes 3–5 concrete follow-on experiment ideas using forced tool-use. Each idea specifies a title, hypothesis, full method (named dataset, model, baseline, and evaluation metric), the specific gap it addresses, and a difficulty rating (low / medium / high). The model also selects the single most-promising idea with a one-sentence justification. Runs automatically after synthesis.

**Stage 7 — Codegen** (`research_assistant/capabilities/codegen.py`) *(user-triggered)*

Takes a selected experiment idea and generates a complete implementation via two sequential Claude Sonnet calls — plain text, no tool-use, full generation capability. The first call produces a runnable Python script with `argparse` for all key hyperparameters, only standard ML libraries (torch, transformers, datasets, faiss-cpu, sentence-transformers, evaluate), a formatted results table at the end of `main()`, and `if __name__ == "__main__": main()`. Header comment lines (`# Requirements:`, `# Estimated Runtime:`) are parsed and surfaced in the UI. The second call restructures the script into a Jupyter notebook with markdown section cells (Setup, Config, Data Loading, Model Setup, Experiment Loop, Results). If the notebook JSON is truncated or malformed, a minimal single-cell fallback notebook is returned so the download always works. Both files are available as direct downloads from the UI.

---

## Data sources

| Source | Used for | Notes |
|--------|----------|-------|
| arXiv Atom API | Paper search, metadata, full text (HTML endpoint) | Primary source; always active |
| Semantic Scholar Graph API | Second search source; citation count, reference list | Free tier; 429s are retried once then skipped gracefully |
| Papers with Code API | Benchmark results, top code repository per paper | Best-effort; silently skipped on any network or rate-limit error |

---

## Example output

Query: *"What is the current state of retrieval-augmented generation for large language models?"*

The following is the first two paragraphs of the synthesis produced on 2026-06-30, from 10 papers found independently by the agent (full synthesis and run metadata in `eval/results/`, methodology in [`eval/EVALUATION.md`](eval/EVALUATION.md)):

> Retrieval-Augmented Generation (RAG) has rapidly matured from a niche technique into a broadly adopted architectural paradigm for grounding large language model outputs in external knowledge. The breadth of recent work reflects this maturity: RAG has been applied to domains as varied as clinical prediction (EHR-RAGp, [2605.12335]), legal document evaluation ([2509.12382]), code generation (EvoR, [2402.12317]), multi-hop question answering (FAIR-RAG, [2510.22344]), Quranic studies ([2503.16581]), automated literature review ([2411.18583]), and even image generation ([2506.06962]). Across the literature, there is strong consensus that retrieved context quality is the primary determinant of downstream response quality, and that faithfulness, relevance, and accuracy constitute the canonical triad of evaluation metrics. The field broadly agrees that dynamic and iterative retrieval mechanisms outperform static, single-pass approaches — a finding replicated across question answering, code generation, and image synthesis settings alike.
>
> Where the literature diverges meaningfully is on the question of *how* retrieval should be structured. EvoR ([2402.12317]) argues that a heterogeneous "knowledge soup" combining library documentation, web search results, execution traces, and LLM-generated snippets — updated synchronously with iteratively refined queries — yields 2–4× accuracy improvements over single-source baselines in code generation. FAIR-RAG ([2510.22344]) takes yet another perspective, attributing performance gains primarily to query-side complexity analysis: its Structured Evidence Assessment (SEA) module deconstructs multi-hop queries into required sub-findings, achieving an 8.3-point F1 improvement over iterative baselines on HotpotQA. The multi-agent RAG architecture ([2412.05838]) proposes that retrieval should be decomposed by data modality — deploying specialized agents for relational databases, document stores, and graph databases respectively. These positions suggest that optimal retrieval granularity and source diversity are task-dependent design choices rather than universal recommendations.

The 10 papers in this run, discovered by arXiv keyword search with no manual curation:

| arXiv ID | Title |
|---|---|
| 2503.16581 | Investigating Retrieval-Augmented Generation in Quranic Studies |
| 2411.18583 | Automated Literature Review Using NLP Techniques and LLM-Based RAG |
| 2506.06962 | AR-RAG: Autoregressive Retrieval Augmentation for Image Generation |
| 2402.12317 | EvoR: Evolving Retrieval for Code Generation |
| 2510.22344 | FAIR-RAG: Faithful Adaptive Iterative Refinement for RAG |
| 2502.00306 | Riddle Me This! Stealthy Membership Inference for RAG |
| 2605.12335 | EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model |
| 2412.05838 | A Collaborative Multi-Agent Approach to RAG with Structured Knowledge |
| 2601.05264 | Engineering the RAG Stack: A Comprehensive Review |
| 2509.12382 | LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation |

---

## Evaluation

The agent was given the query that Gao et al.'s 2023 survey (*Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv 2312.10997) was written to answer, and searched independently. The Gao survey did not appear in the agent's results, confirming the evaluation corpus is genuinely independent. The agent's synthesis themes — retrieval mechanism design, knowledge source diversity, model size trade-offs, security, and evaluation rigor — were mapped against Gao et al.'s Naive/Advanced/Modular taxonomy and Retrieval/Generation/Augmentation framing. Three specific numerical claims in the synthesis were verified manually against primary sources: all three matched exactly with no distortion.

Full methodology, coverage assessment, and factual accuracy spot-check: [`eval/EVALUATION.md`](eval/EVALUATION.md).

---

## Architecture

Capabilities are registered in a global dict (`research_assistant/registry.py`) using a `@register("name")` decorator. The orchestrator resolves capabilities by name at runtime and never imports them directly — adding a new capability requires only creating a new module with `@register` and importing it. `ResearchContext` (`research_assistant/context.py`) is the only shared state between stages: a typed dataclass that flows through the pipeline, each stage reading fields written upstream and writing fields consumed downstream.

The extraction schema (`research_assistant/schemas/general_cs_paper.json`) is decoupled from the extraction logic. Swapping the schema file changes what fields are extracted without touching any capability code.

Full design rationale, data-flow diagram, and extension points: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**v2 slots** (designed for, not yet built):

- Papers with Code integration — benchmark results, linked code repositories
- Domain-specific extraction schemas — clinical PICO, social science, economics
- Code generation from synthesized method descriptions
- Persistent paper and extraction cache
- Incremental / streaming execution

---

## v1 → v2

v1 shipped search through synthesize — a five-stage pipeline with a single "Ask a Research Question" mode and a basic Streamlit UI.

v2 additions, in the order they were built:

- **Find Related Work mode** — new pipeline branch (search → fetch → extract → compare → relate) that takes a project description and returns an annotated bibliography with a positioning summary. The relate stage uses Claude Haiku with forced tool-use.
- **PDF upload** — project descriptions can be uploaded as a PDF rather than typed; text is extracted with `pypdf` and capped at 15,000 characters.
- **Multi-source search** — Semantic Scholar added as a second search source alongside arXiv; results are merged and deduplicated by arXiv ID and normalized title.
- **Papers with Code enrichment** — benchmark results and top code repository URL fetched per paper during the fetch stage and surfaced in the UI and extraction prompt.
- **Experiment ideation (Stage 6)** — Claude Sonnet proposes 3–5 concrete, grounded experiment ideas from the synthesis and comparison output, each with named datasets, models, metrics, and a difficulty rating.
- **Code generation (Stage 7)** — user-triggered; Claude Sonnet generates a complete runnable Python script and Jupyter notebook for any selected experiment idea, with download buttons for both in the UI.

---

## Guardrails

The pipeline includes a validation layer (`research_assistant/guardrails.py`) that runs automatically at each stage, catching bad inputs, poor-quality outputs, and unsupported factual claims before they reach the user.

**Query & Search Validation**

Before the pipeline starts, the query is checked locally for obvious problems (empty, fewer than 10 characters) and then by Claude Haiku for validity — harmful requests, arithmetic, gibberish, and non-research content are blocked outright, while vague queries and non-English input produce warnings rather than stopping the pipeline. After the search stage, results are checked for volume and relevance: if fewer than 3 papers are returned, or fewer than 30% of retrieved titles match query keywords, the user is warned before the pipeline continues.

**Stage Output Validation**

After each of the four core processing stages, the output is checked against minimum quality thresholds: extraction coverage (at least 80% of papers must return a non-null research question and key results), comparison completeness (at least one agreement and one disagreement must be identified), synthesis quality (more than 500 characters, predominantly ASCII text, at least one paper citation by arXiv ID, "et al.", or title), and experiment idea completeness (each idea must have a non-empty hypothesis and method, and `most_promising` must be a valid index). Fatal failures stop the pipeline immediately; non-fatal issues surface as inline warnings inside the progress status block in the UI.

**Hallucination / Grounding Detection**

After synthesis, Claude Haiku checks whether specific factual claims in the synthesis — numbers, percentages, benchmark scores, named techniques — can be traced back to the source paper extractions. Grounded and ungrounded claims are reported separately alongside a confidence rating (high / medium / low). If ungrounded claims are found, a warning expander appears directly below the synthesis in the UI, listing the specific claims that couldn't be verified against the source material; the synthesis is still shown in full, but the user can see exactly what to check. This was validated against the same three numerical claims fact-checked manually in [`eval/EVALUATION.md`](eval/EVALUATION.md) — all three were correctly identified as grounded.

---

## Running it

```bash
git clone https://github.com/xmashaxxx/research_assistant.git
cd research_assistant
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...   # required for extract, compare, synthesize
streamlit run app.py                  # interactive UI at http://localhost:8501
```

Or run the evaluation script directly:

```bash
python eval/run_evaluation.py
```

Results are saved to `eval/results/rag_state_of_field_run.json` (gitignored — generated output, not committed).

**Cost**: a full pipeline run (10 papers, all five stages) costs roughly $0.10–$0.30 depending on paper length, split between Claude Haiku for extraction and comparison and Claude Sonnet for synthesis.

---

## Tests

90 tests across five test modules, one per capability. All tests that call external APIs (arXiv, Semantic Scholar, Anthropic) are live integration tests against real endpoints — no mocks. Module-scoped fixtures ensure each test module makes at most two arXiv API calls regardless of the number of test functions, keeping the full suite runtime under two minutes.

```bash
python -m pytest tests/ -v
```

Tests that require `ANTHROPIC_API_KEY` are skipped automatically when the key is absent, so the fetch and search tests run without API credentials.

---

## Built with Claude Code

This project was built collaboratively with [Claude Code](https://claude.com/claude-code). Claude is listed as co-author on individual commits throughout the history — that attribution reflects how the code was actually written, not a generic disclosure.

---

## License

MIT. See [LICENSE](LICENSE).
