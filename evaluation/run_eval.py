"""Ablation study + end-to-end checks:  python -m evaluation.run_eval [--out evaluation/results.md]"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from statistics import mean

from rag_pro import RAGPipeline

from .dataset import QA
from .metrics import doc_ranking, mrr, ndcg_at_k, recall_at_k

CONFIGS = [
    ("BM25 only", dict(mode="bm25", rerank=False, multi_query=False)),
    ("Dense (LSA) only", dict(mode="dense", rerank=False, multi_query=False)),
    ("Hybrid (RRF)", dict(mode="hybrid", rerank=False, multi_query=False)),
    ("Hybrid + rerank", dict(mode="hybrid", rerank=True, multi_query=False)),
    ("Hybrid + rerank + multi-query", dict(mode="hybrid", rerank=True, multi_query=True)),
]


def build_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    pipe.ingest_paths([str(Path(__file__).resolve().parent.parent / "data" / "sample_docs")])
    return pipe


def run(out: str | None = None) -> str:
    pipe = build_pipeline()
    answerable = [x for x in QA if x["gold_docs"]]
    lines = [
        f"Corpus: {len(pipe.documents)} documents, {len(pipe.chunks)} chunks. "
        f"{len(answerable)} answerable questions, {len(QA) - len(answerable)} unanswerable.\n",
        "### Retrieval ablation (document-level)\n",
        "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, cfg in CONFIGS:
        r1, r3, r5, rr, nd, ms = [], [], [], [], [], []
        for x in answerable:
            t0 = time.perf_counter()
            hits, _ = pipe.retrieve(x["q"], k=5, **cfg)
            ms.append((time.perf_counter() - t0) * 1000)
            ranked = doc_ranking([h.chunk.doc_id for h in hits])
            g = x["gold_docs"]
            r1.append(recall_at_k(ranked, g, 1)); r3.append(recall_at_k(ranked, g, 3)); r5.append(recall_at_k(ranked, g, 5))
            rr.append(mrr(ranked, g)); nd.append(ndcg_at_k(ranked, g, 5))
        lines.append(f"| {name} | {mean(r1):.2f} | {mean(r3):.2f} | {mean(r5):.2f} | {mean(rr):.2f} | {mean(nd):.2f} | {mean(ms):.1f} |")

    # end-to-end with the default (full) pipeline
    kw, grounded, valid_cites, answered_ok = [], [], [], []
    abstain_correct = []
    for x in QA:
        a = pipe.query(x["q"])
        if not x["gold_docs"]:
            abstain_correct.append(a.abstained)
            continue
        answered_ok.append(not a.abstained)
        text = a.answer.lower()
        kw.append(mean(1.0 if re.search(re.escape(k.lower()), text) else 0.0 for k in x["keywords"]))
        grounded.append(a.grounded)
        valid_cites.append(bool(a.citations) and not a.grounding["invalid_citations"])
    lines += [
        "\n### End-to-end (full pipeline, extractive generator)\n",
        "| Metric | Value |", "|---|---|",
        f"| Answered when answerable | {mean(answered_ok):.2f} |",
        f"| Abstained when unanswerable | {mean(abstain_correct):.2f} |",
        f"| Answer keyword recall | {mean(kw):.2f} |",
        f"| Grounded answers | {mean(grounded):.2f} |",
        f"| Answers with valid citations | {mean(valid_cites):.2f} |",
    ]
    report = "\n".join(lines)
    print(report)
    if out:
        Path(out).write_text(report + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    run(ap.parse_args().out)
