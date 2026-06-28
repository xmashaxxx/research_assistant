from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchContext:
    """Shared state flowing through every stage of the research pipeline.

    Each capability reads fields written by earlier stages and writes into
    the fields it owns. No capability calls another directly.
    """

    query: str = ""
    found_papers: list = field(default_factory=list)
    summaries: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)
    synthesis: str | None = None
    extraction_schema: str = "general_cs_paper"
