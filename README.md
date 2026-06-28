# research_assistant

An agentic pipeline for scientific literature: search, fetch, extract, compare, and synthesise — driven by a research question.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## What it does (v1 scope)

Given a natural-language research question, the pipeline:

1. **Searches** arXiv for relevant papers
2. **Fetches** paper text and enriches metadata via Semantic Scholar
3. **Extracts** structured fields from each paper using a configurable JSON schema (default: CS/ML — research question, method, dataset, key results, limitations)
4. **Compares** papers: agreements, contradictions, open questions
5. **Synthesises** a state-of-the-field narrative that answers the original question with citations

A thin Streamlit layer wraps the Python API for interactive use; the API is fully usable without it.

---

## Architecture in brief

- **Capability registry** — a global dict of `name → callable`; modules self-register with `@register("name")` on import. The orchestrator resolves names at runtime; it never imports capabilities directly.
- **ResearchContext** — a typed dataclass that is the only shared state between stages. Each capability reads fields written upstream and writes fields consumed downstream.
- **Schema-driven extraction** — the `extract` capability is parameterised by a JSON schema; swap the schema to target a different paper type without touching any other code.

---

## v2 (deliberately deferred)

- Papers with Code integration (benchmark results, repositories)
- Code generation from synthesised method descriptions
- Hyperparameter recommendation across papers
- Domain-specific extraction schemas (clinical PICO, social science, economics)
- Incremental / streaming execution
- Persistent paper + extraction cache

---

## Status

**Skeleton commit — capability logic next.**

Package structure, registry, orchestrator, and all stubs are in place. Tests confirm the registry wires up correctly and every stub raises `NotImplementedError` as expected. The next commits will replace the stubs with real implementations starting with `search` → `fetch` → `extract`.

---

## Quickstart (once implemented)

```bash
pip install -r requirements.txt
python -m pytest tests/
streamlit run app.py
```
