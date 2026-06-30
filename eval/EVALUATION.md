# Evaluation: Research Assistant vs. Gao et al. RAG Survey
Watch the demo ----> https://youtu.be/ypGdeR_sSE8
## Methodology

The evaluation tests whether the research assistant can independently answer a question that a canonical human-authored survey was written to address. The target paper is:

> Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv 2312.10997.

The agent was given the query:

> "What is the current state of retrieval-augmented generation for large language models?"

It searched arXiv autonomously, fetched full text where available, extracted structured findings from each paper using Claude (Haiku), compared those findings across papers, and produced a final synthesis using Claude (Sonnet).

**Corpus independence was confirmed.** arXiv 2312.10997 did not appear in the agent's search results. The 10 papers the agent found and synthesized were discovered and selected entirely independently of the evaluation target — the agent was not summarising the survey it is being compared against.

This matters because it changes what the evaluation is actually measuring. The agent's synthesis represents what a domain-naive system surfaces and concludes from first principles, given the same starting question. Any thematic overlap with Gao et al. is genuine convergence, not retrieval of the answer key.

---

## Coverage Assessment

The agent organized its synthesis around five themes that emerged from the 10 papers it retrieved:

1. **Retrieval mechanism design** — granularity of retrieval, hybrid and iterative strategies, and when to retrieve vs. generate directly.
2. **Knowledge source diversity** — structured databases, code repositories, and heterogeneous corpora beyond unstructured text.
3. **Model size trade-offs** — the differing roles of small vs. large LMs as generators in RAG pipelines.
4. **Security vulnerabilities** — membership inference attacks and corpus poisoning risks specific to RAG architectures.
5. **Evaluation methodology rigor** — dataset choice, metric selection, and benchmark contamination as sources of unreliable evaluation in the literature.

Gao et al. organize their survey along two orthogonal axes: a developmental taxonomy (Naive RAG → Advanced RAG → Modular RAG) and a component framing (Retrieval / Generation / Augmentation). Their coverage spans roughly 100 papers and addresses deployment considerations, domain-specific adaptations, and multimodal RAG that the agent's 10-paper corpus does not reach.

The thematic overlap is real but partial. The agent's retrieval mechanism and knowledge source themes map naturally onto Gao et al.'s Retrieval and Augmentation components. The model size trade-off observation corresponds to discussions of generator capacity in the Advanced RAG section. Security vulnerabilities and evaluation rigor are present but peripheral in Gao et al., whereas the agent's corpus happened to surface them more prominently — likely because those were active areas of publication at query time.

The key structural difference is that Gao et al. present a deliberate, hierarchical taxonomy built retrospectively across a large corpus. The agent's synthesis reflects the themes that happened to be legible from 10 recently published papers. The coverage is narrower by construction, not by error.

---

## Factual Accuracy Spot-Check

Three specific numerical claims in the agent's synthesis were verified manually against the primary source papers.

**Claim 1 — Membership inference attack (arXiv 2502.00306)**

The synthesis stated that the attack successfully avoided detection "76x more frequently" than baseline methods, at a cost of "$0.02 per document," and that "30 queries" were sufficient for successful membership inference.

*Result: verified exact match.* All three figures appear in the paper's abstract with no distortion.

**Claim 2 — FAIR-RAG (arXiv 2510.22344)**

The synthesis reported an F1-score of 0.453 and an 8.3-point absolute improvement over the strongest iterative baseline on HotpotQA.

*Result: verified exact match.* Both the absolute score and the improvement margin are stated correctly.

**Claim 3 — EvoR (arXiv 2402.12317)**

The synthesis claimed EvoR achieved a 2–4× improvement in execution accuracy over baselines.

*Result: verified exact match.* The range is correctly attributed.

**Summary: 3 of 3 spot-checked claims verified accurate**, with no rounding errors, directional errors, or fabricated precision found.

---

## Limitations of This Evaluation

This is a small spot-check, not a comprehensive fact audit. Three verified claims out of dozens in the synthesis is enough to establish that the pipeline does not systematically hallucinate numbers, but it is not enough to rule out occasional errors elsewhere in the text. A production deployment would need automated citation verification — tracing every quantitative claim back to its source passage — rather than manual sampling.

The corpus size is also a hard constraint. With 10 papers, the agent cannot produce a synthesis comparable in scope to a human survey built on 100+. This is expected and not a failure of the pipeline; it is a consequence of running a single query against a keyword search interface. A multi-round search strategy, or seeded search from an initial result set, would expand coverage at the cost of latency and compute.

Finally, the evaluation is a single run against a single query. The search results will vary across runs as new papers are indexed, and the synthesis will vary with them. The verified accuracy of this particular run does not imply consistent accuracy across all possible queries or paper sets.

---

## Conclusion

This evaluation demonstrates two things: the pipeline reliably surfaces relevant, current literature independent of any human-curated reference list, and the specific numerical claims in its synthesis are verifiably accurate against primary sources.

It does not demonstrate that the agent matches the analytical depth or taxonomic coherence of a human-authored survey. Gao et al.'s organizing framework — built from sustained engagement with a large body of work — reflects a kind of structural understanding that a single-pass pipeline over 10 papers cannot replicate. The agent's synthesis is accurate and topically coherent; it is not a substitute for expert literature synthesis.

What this evaluation does establish is that the pipeline is a credible starting point: it finds the right papers, reads them accurately, and reports what they say without distorting the numbers. That is a meaningful baseline for an automated system, and it is what this evaluation was designed to test.
