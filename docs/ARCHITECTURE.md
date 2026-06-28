# Architecture: Research Assistant

## Overview

`research_assistant` is an agentic pipeline for scientific literature. Given a research question it searches for relevant papers, fetches and parses them, extracts structured information per a configurable schema, compares findings across papers, and synthesises a state-of-the-field answer.

The design separates **what to do** (the pipeline definition) from **how to do it** (the capability implementations). This makes individual capabilities testable in isolation, swappable, and extendable to new data sources or domains without touching the orchestration logic.

---

## Core Concepts

### 1. Capability Registry

A global dict maps string names → capability callables. Modules self-register with the `@register("name")` decorator on import. The orchestrator never imports capabilities directly; it resolves them at runtime through the registry.

```
registry = {
    "search":     <function search_papers>,
    "fetch":      <function fetch_papers>,
    "extract":    <function extract_summaries>,
    "compare":    <function compare_papers>,
    "synthesize": <function synthesize>,
}
```

Adding a capability means: write the module, decorate it, and import it before running the orchestrator. Nothing else changes.

### 2. ResearchContext

A typed dataclass that flows through every stage of the pipeline. Each capability reads from context fields set by earlier stages and writes results into later fields. The context is the only shared state — capabilities do not call each other.

```python
@dataclass
class ResearchContext:
    query: str
    found_papers: list       # populated by: search
    summaries: dict          # populated by: fetch + extract
    comparisons: dict        # populated by: compare
    synthesis: str | None    # populated by: synthesize
```

### 3. Orchestrator

Accepts a `ResearchContext` and an ordered list of capability names. Resolves each name against the registry and calls `capability(context)` in sequence. Short-circuits on capability error.

```python
orchestrator.run(context, ["search", "fetch", "extract", "compare", "synthesize"])
```

### 4. Schema-Driven Extraction

The `extract` capability consults a JSON schema describing which fields to pull from a paper. The default schema (`schemas/general_cs_paper.json`) targets CS/ML papers and extracts: `research_question`, `method`, `dataset`, `key_results`, `limitations`.

The schema path is passed through `ResearchContext.extraction_schema` (v1 default: the general CS/ML schema). Domain-specific schemas (e.g., clinical study schema with PICO fields) are a v2 slot.

### 5. Presentation Layer

A thin Streamlit app wraps the Python API. It owns no business logic — it builds a `ResearchContext`, calls `orchestrator.run()`, and renders the result. The Python API is fully usable without Streamlit.

---

## v1 Scope

| Layer | v1 Implementation |
|---|---|
| Search | arXiv API (keyword + category filter) |
| Fetch | arXiv abstract + PDF → BeautifulSoup text extraction |
| Semantic enrichment | Semantic Scholar API (citation count, references) |
| Extraction | Claude API, prompted with the active JSON schema |
| Comparison | Claude API, cross-paper structured diff |
| Synthesis | Claude API, state-of-the-field narrative |
| Presentation | Streamlit single-page app |

---

## v2 Slots (Deliberately Deferred)

- **Papers with Code** integration — link papers to associated repositories and benchmark results
- **Code generation** — generate starter code from the synthesised method description
- **Hyperparameter recommendation** — extract and rank hyperparameters reported across papers
- **Domain schemas** — clinical study (PICO), social science (theory/sample/measure), economics (model/instrument/identification)
- **Incremental / streaming** execution — yield partial results as each capability completes
- **Persistent store** — cache fetched papers and extracted summaries to disk/DB

---

## Data Flow Diagram

```
User query
    │
    ▼
[search]  ──► found_papers: list[PaperMetadata]
    │
    ▼
[fetch]   ──► summaries keys populated with raw text
    │
    ▼
[extract] ──► summaries values populated with structured dicts (schema-driven)
    │
    ▼
[compare] ──► comparisons: dict (cross-paper structured diff)
    │
    ▼
[synthesize] ──► synthesis: str (state-of-the-field narrative)
    │
    ▼
Streamlit presentation layer
```

---

## Directory Layout

```
research_assistant/
├── __init__.py
├── context.py          # ResearchContext dataclass
├── registry.py         # @register decorator + global registry dict
├── orchestrator.py     # run(context, capability_names)
├── capabilities/
│   ├── __init__.py     # imports all capability modules (triggers self-registration)
│   ├── search.py
│   ├── fetch.py
│   ├── extract.py
│   ├── compare.py
│   └── synthesize.py
└── schemas/
    ├── __init__.py
    └── general_cs_paper.json
docs/
    ARCHITECTURE.md     # this file
tests/
    test_skeleton.py
requirements.txt
```
