# research_assistant

An agentic research assistant that takes a natural-language research question, searches arXiv for relevant papers, reads and extracts structured findings from each, compares them across papers, and synthesizes a literature review that directly answers the question with citations. The technical thesis is that a multi-step pipeline — search, extract, compare, synthesize as separable capabilities — produces a more accurate and traceable synthesis than single-pass summarization: each stage operates on structured output from the one before it, so errors are localized and the final synthesis is grounded in verified intermediate representations rather than raw text.
Watch the demo ----> https://youtu.be/ypGdeR_sSE8
---

## Pipeline

**Stage 1 — Search** (`research_assistant/capabilities/search.py`)

Queries the arXiv Atom API with the user's research question and returns up to ten papers ranked by relevance. Each paper is parsed into a `PaperRecord` carrying the arXiv ID, title, authors, abstract, published date, categories, and URLs. The search stage deliberately returns lightweight records — full text is deferred to the fetch stage so the search can be fast and retried cheaply.

**Stage 2 — Fetch** (`research_assistant/capabilities/fetch_paper.py`)

For each paper ID from the search stage, fetches the full paper text from the arXiv HTML endpoint and enriches metadata via the Semantic Scholar Graph API (citation count, references). Full text is capped at 50,000 characters. Where the HTML endpoint is unavailable, the abstract falls back as the text source. Fetch retries transient errors with exponential backoff (5s, 15s) before giving up.

**Stage 3 — Extract** (`research_assistant/capabilities/extract.py`)

Calls Claude Haiku (`claude-haiku-4-5-20251001`) with tool-use forced via `tool_choice` to extract six structured fields from each paper: research question, method, dataset, key results, limitations, and related work. Forcing a named tool means the output is always a typed dict — no parsing, no markdown stripping. The extraction schema is defined in `research_assistant/schemas/general_cs_paper.json` and is designed to be swapped for domain-specific schemas (clinical PICO, social science, economics) without touching any other code.

**Stage 4 — Compare** (`research_assistant/capabilities/compare.py`)

Calls Claude Haiku again, this time with all per-paper extractions as input, and produces three structured outputs: agreements (claims that appear across multiple papers), disagreements (positions that conflict, with per-paper stances), and unique contributions (findings specific to a single paper). This stage runs on structured extractions, not raw text — it is comparing semantics, not prose.

**Stage 5 — Synthesize** (`research_assistant/capabilities/synthesize.py`)

Calls Claude Sonnet (`claude-sonnet-4-6`) with the per-paper extractions and the comparison summary, and writes a flowing prose literature review that answers the original question. Sonnet is used here rather than Haiku because this is the user-facing output and writing quality is consequential. No tool-use — the instruction is for natural prose, 3–5 paragraphs, with papers cited by title.

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
