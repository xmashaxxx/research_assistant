"""Paper fetch-and-parse capability."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord
from research_assistant.registry import register

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_MAX_TEXT_CHARS = 50_000


def _strip_version(arxiv_id: str) -> str:
    """'2506.06962v3' → '2506.06962' for Semantic Scholar lookups."""
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


def _normalize_id(raw: str) -> str:
    """Normalize any arxiv_id form to a clean base ID.

    Handles:
      - Full URLs: 'https://arxiv.org/abs/2506.06962v3' → '2506.06962'
      - Clean IDs with version: '2312.10997v2' → '2312.10997'
      - Clean IDs without version: '2312.10997' → '2312.10997'
    """
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return _strip_version(raw)


def _atom_text(entry: ET.Element, tag: str) -> str:
    el = entry.find(f"{{{_ATOM_NS}}}{tag}")
    return " ".join((el.text or "").split()) if el is not None else ""


def _fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    """Query arXiv by ID and return a metadata dict, or None on failure.

    Retries up to 3 times with exponential back-off to handle transient
    rate limiting from the arXiv API (export.arxiv.org).
    """
    for attempt in range(3):
        if attempt:
            time.sleep(5 if attempt == 1 else 15)  # 5s, 15s
        try:
            resp = requests.get(
                _ARXIV_API,
                params={"id_list": arxiv_id, "max_results": 1},
                timeout=15,
                headers={"User-Agent": "research-assistant/0.1"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            entries = root.findall(f"{{{_ATOM_NS}}}entry")
            if not entries:
                return None
            entry = entries[0]

            authors = [
                " ".join((name.text or "").split())
                for author in entry.findall(f"{{{_ATOM_NS}}}author")
                if (name := author.find(f"{{{_ATOM_NS}}}name")) is not None
            ]

            categories = [
                el.get("term", "")
                for el in entry.findall(f"{{{_ATOM_NS}}}category")
                if el.get("term")
            ]

            pc_el = entry.find(f"{{{_ARXIV_NS}}}primary_category")
            primary_category = (
                pc_el.get("term", "") if pc_el is not None
                else (categories[0] if categories else "")
            )

            return {
                "title": _atom_text(entry, "title"),
                "abstract": _atom_text(entry, "summary"),
                "published": _atom_text(entry, "published"),
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
            }
        except Exception as exc:
            print(f"[fetch]   arXiv metadata attempt {attempt + 1}/3 failed: {exc}")

    return None


def _fetch_full_text(arxiv_id: str) -> str | None:
    """Try the arXiv HTML endpoint; return cleaned body text or None."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "research-assistant/0.1"},
        )
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        container = soup.find("article") or soup.find("body")
        if container is None:
            return None
        text = container.get_text(separator=" ", strip=True)
        return text[:_MAX_TEXT_CHARS] if text else None
    except Exception:
        return None


def _fetch_semantic_scholar(arxiv_id: str) -> tuple[int | None, list | None]:
    """Return (citation_count, references) from Semantic Scholar, or (None, None)."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    try:
        resp = requests.get(
            url,
            params={"fields": "citationCount,references"},
            timeout=10,
            headers={"User-Agent": "research-assistant/0.1"},
        )
        if resp.status_code == 429:
            print("[fetch] Semantic Scholar rate limit; skipping enrichment.")
            return None, None
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        citation_count = data.get("citationCount")
        refs = [
            {"paperId": r.get("paperId"), "title": r.get("title")}
            for r in (data.get("references") or [])
            if r.get("title")
        ]
        return citation_count, refs or None
    except Exception:
        return None, None


_PWC_BASE = "https://paperswithcode.com/api/v1"
_PWC_HEADERS = {"User-Agent": "research-assistant/0.1"}


def _fetch_papers_with_code(
    arxiv_id: str,
) -> tuple[list[dict] | None, str | None]:
    """Return (benchmark_results, code_url) from Papers with Code, or (None, None).

    Makes up to three requests:
      1. Search for the paper by arXiv ID → get the PwC slug
      2. Fetch evaluation results for that slug → extract up to 5 benchmarks
      3. Fetch repositories → pick the one with the most stars

    Any failure (HTTP error, missing data, network timeout) returns (None, None)
    so the pipeline continues without benchmark data.
    """
    try:
        # --- Step 1: find PwC paper ID ---
        resp = requests.get(
            f"{_PWC_BASE}/papers/",
            params={"arxiv_id": arxiv_id},
            timeout=10,
            headers=_PWC_HEADERS,
        )
        if resp.status_code != 200:
            return None, None
        search_data = resp.json()
        items = search_data.get("results") or []
        if not items:
            return None, None
        pwc_id = items[0].get("id")
        if not pwc_id:
            return None, None

        # --- Step 2: benchmark results ---
        benchmark_results: list[dict] | None = None
        resp2 = requests.get(
            f"{_PWC_BASE}/paper/{pwc_id}/results/",
            timeout=10,
            headers=_PWC_HEADERS,
        )
        if resp2.status_code == 200:
            raw = (resp2.json().get("results") or [])[:5]
            parsed = []
            for r in raw:
                task_obj = r.get("task") or {}
                task = task_obj.get("name") or str(task_obj) if task_obj else ""
                dataset_obj = r.get("dataset") or {}
                dataset = dataset_obj.get("name") or str(dataset_obj) if dataset_obj else ""
                metrics = r.get("metrics") or []
                if metrics:
                    m = metrics[0]
                    metric = (m.get("type") or {}).get("name") or ""
                    score = str(m.get("value") or "")
                else:
                    metric = score = ""
                if task or dataset:
                    parsed.append(
                        {"task": task, "dataset": dataset, "metric": metric, "score": score}
                    )
            if parsed:
                benchmark_results = parsed

        # --- Step 3: best code repository ---
        code_url: str | None = None
        resp3 = requests.get(
            f"{_PWC_BASE}/paper/{pwc_id}/repositories/",
            timeout=10,
            headers=_PWC_HEADERS,
        )
        if resp3.status_code == 200:
            repos = resp3.json().get("results") or []
            if repos:
                best = max(repos, key=lambda r: r.get("stars") or 0)
                code_url = best.get("url") or None

        n = len(benchmark_results or [])
        print(f"[fetch] PwC: {n} benchmark results, code: {code_url or 'none'}")
        return benchmark_results, code_url

    except Exception as exc:
        print(f"[fetch] PwC: error — {exc}")
        return None, None


def fetch_papers(ids: list[str]) -> list[PaperRecord]:
    """Fetch complete PaperRecord objects for a list of arXiv IDs.

    Each ID may be a raw arXiv ID ("2312.10997"), a versioned ID
    ("2312.10997v3"), or a full arXiv URL. The stored arxiv_id on each
    returned record is always the clean base ID (no version suffix).

    For each paper:
    - Calls the arXiv API to retrieve metadata (title, authors, abstract,
      categories, primary_category, published date)
    - Tries the arXiv HTML endpoint for full body text; falls back to abstract
    - Queries Semantic Scholar for citation count and reference list

    Papers that fail the arXiv metadata fetch are silently skipped.
    Result order matches the input order.
    """
    if not ids:
        return []

    records = []
    for i, raw_id in enumerate(ids):
        arxiv_id = _normalize_id(raw_id)
        print(f"[fetch] ({i + 1}/{len(ids)}) {arxiv_id}")

        metadata = _fetch_arxiv_metadata(arxiv_id)
        if metadata is None:
            print(f"[fetch]   arXiv metadata fetch failed for {arxiv_id}; skipping.")
            continue

        full_text = _fetch_full_text(arxiv_id)
        if full_text:
            print(f"[fetch]   HTML text: {len(full_text):,} chars")
        else:
            full_text = metadata["abstract"]
            print(f"[fetch]   HTML unavailable — using abstract ({len(full_text):,} chars)")

        citation_count, references = _fetch_semantic_scholar(arxiv_id)
        if citation_count is not None:
            print(f"[fetch]   citations: {citation_count}, references: {len(references or [])}")
        else:
            print("[fetch]   Semantic Scholar: no data")

        time.sleep(0.5)
        benchmark_results, code_url = _fetch_papers_with_code(arxiv_id)

        records.append(
            PaperRecord(
                arxiv_id=arxiv_id,
                title=metadata["title"],
                authors=metadata["authors"],
                abstract=metadata["abstract"],
                published=metadata["published"],
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                categories=metadata["categories"],
                primary_category=metadata["primary_category"],
                source="arxiv",
                full_text=full_text,
                citation_count=citation_count,
                references=references,
                benchmark_results=benchmark_results,
                code_url=code_url,
            )
        )

        # Be polite between papers to avoid rate limits on arXiv and S2.
        if i < len(ids) - 1:
            time.sleep(1)

    print(f"[fetch] Done. {len(records)} papers fetched.")
    return records


@register("fetch")
def _fetch_capability(context: ResearchContext) -> None:
    """Orchestrator wrapper for the fetch stage.

    Extracts arXiv IDs from context.found_papers, calls fetch_papers(),
    replaces context.found_papers with fully-populated records, and writes
    context.summaries[arxiv_id] for the extract stage.
    """
    ids = [
        p.arxiv_id if isinstance(p, PaperRecord) else str(p)
        for p in context.found_papers
    ]
    print(f"[fetch] Fetching {len(ids)} papers...")
    records = fetch_papers(ids)
    context.found_papers = records
    for paper in records:
        context.summaries[paper.arxiv_id] = {
            "raw_text": paper.full_text or paper.abstract,
            "metadata": {
                "citation_count": paper.citation_count,
                "references": paper.references,
            },
        }
    print(f"[fetch] Done. {len(context.summaries)} papers in context.summaries.")
