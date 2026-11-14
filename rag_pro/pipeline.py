from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder, make_embedder
from .generation import NO_ANSWER, ExtractiveGenerator, Generator, LLMGenerator, fit_hits
from .grounding import extract_citation_numbers, grounding_report
from .llm import LLM, AnthropicLLM
from .loaders import load_paths
from .query import expand_query
from .rerank import LexicalReranker, Reranker
from .retrieval import HybridRetriever
from .schema import Answer, Chunk, Citation, Document, Hit


class RAGPipeline:
    """ingest -> chunk -> embed + BM25 index -> [multi-query -> hybrid retrieve -> rerank] -> generate -> ground."""

    def __init__(self, settings: Optional[Settings] = None, embedder: Optional[Embedder] = None,
                 generator: Optional[Generator] = None, reranker: Optional[Reranker] = None,
                 llm: Optional[LLM] = None):
        self.s = settings or Settings()
        self.embedder = embedder or make_embedder(self.s.embedder, self.s.embedding_dim, self.s.st_model)
        self.reranker = reranker or LexicalReranker()
        if llm is None and self.s.generator == "anthropic":
            llm = AnthropicLLM(self.s.model)
        self.llm = llm
        self.generator = generator or (LLMGenerator(llm) if llm else ExtractiveGenerator())
        self.documents: Dict[str, Document] = {}
        self.chunks: List[Chunk] = []
        self.matrix: Optional[np.ndarray] = None
        self.retriever: Optional[HybridRetriever] = None

    # ---------------- indexing ----------------
    def add_documents(self, docs: List[Document]) -> Dict[str, int]:
        for d in docs:
            self.documents[d.id] = d
        return self._build(fit=True)

    def ingest_paths(self, paths: List[str]) -> Dict[str, int]:
        return self.add_documents(load_paths(paths))

    def _build(self, fit: bool, matrix: Optional[np.ndarray] = None) -> Dict[str, int]:
        self.chunks = [c for d in self.documents.values()
                       for c in chunk_document(d, self.s.chunk_size, self.s.chunk_overlap)]
        if not self.chunks:
            raise ValueError("no text to index")
        texts = [c.index_text for c in self.chunks]
        if fit:
            self.embedder.fit(texts)
        self.matrix = matrix if matrix is not None else self.embedder.embed(texts)
        self.retriever = HybridRetriever(self.chunks, self.embedder, self.matrix, self.s.rrf_k)
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

    # ---------------- persistence ----------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "documents.json").write_text(json.dumps([vars(x) for x in self.documents.values()]), encoding="utf-8")
        (d / "settings.json").write_text(json.dumps(self.s.to_dict(), indent=2), encoding="utf-8")
        np.save(d / "dense.npy", self.matrix)
        with open(d / "embedder.pkl", "wb") as f:
            pickle.dump(self.embedder, f)

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> "RAGPipeline":
        d = Path(directory)
        settings = Settings(**json.loads((d / "settings.json").read_text(encoding="utf-8")))
        with open(d / "embedder.pkl", "rb") as f:   # only load indexes you created yourself
            embedder = pickle.load(f)
        p = cls(settings, embedder=embedder, **kwargs)
        p.documents = {x["id"]: Document(**x) for x in json.loads((d / "documents.json").read_text(encoding="utf-8"))}
        p._build(fit=False, matrix=np.load(d / "dense.npy"))
        return p

    # ---------------- retrieval ----------------
    def retrieve(self, question: str, k: Optional[int] = None, mode: Optional[str] = None,
                 rerank: Optional[bool] = None, multi_query: Optional[bool] = None,
                 filters: Optional[Dict] = None) -> Tuple[List[Hit], Dict[str, float]]:
        assert self.retriever is not None, "index is empty - ingest documents first"
        k = k or self.s.top_k
        mode = mode or self.s.retrieval_mode
        rerank = self.s.rerank if rerank is None else rerank
        multi_query = self.s.multi_query if multi_query is None else multi_query
        t: Dict[str, float] = {}

        t0 = time.perf_counter()
        queries = expand_query(question, self.llm) if multi_query else [question]
        t["expand"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = self.retriever.search_many(queries, self.s.candidates, mode, filters, self.s.mmr_lambda)
        t["retrieve"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = self.reranker.rerank(question, hits, k) if rerank else hits[:k]
        t["rerank"] = (time.perf_counter() - t0) * 1000
        return hits, t

    # ---------------- answering ----------------
    def query(self, question: str, k: Optional[int] = None, mode: Optional[str] = None,
              filters: Optional[Dict] = None) -> Answer:
        hits, timings = self.retrieve(question, k, mode, filters=filters)
        best_cov = max((h.features.get("coverage", 0.0) for h in hits), default=0.0)
        if not hits or best_cov < self.s.min_coverage:
            return Answer(question, NO_ANSWER, [], 0.0, False, True,
                          {"support_ratio": 0.0, "unsupported_sentences": [], "invalid_citations": [], "ok": False},
                          hits, timings)

        t0 = time.perf_counter()
        used = fit_hits(hits, self.s.max_context_chars)
        text = self.generator.generate(question, used)
        timings["generate"] = (time.perf_counter() - t0) * 1000

        report = grounding_report(text, used)
        citations = [Citation(n, used[n - 1].chunk.id, used[n - 1].chunk.doc_id, used[n - 1].chunk.heading,
                              used[n - 1].chunk.text[:160])
                     for n in extract_citation_numbers(text) if 1 <= n <= len(used)]
        abstained = not text or text.lower().startswith("i don't know")
        confidence = 0.0 if abstained else round(0.6 * min(1.0, best_cov) + 0.4 * report["support_ratio"], 3)
        timings = {k_: round(v, 2) for k_, v in timings.items()}
        return Answer(question, text or NO_ANSWER, citations, confidence, report["ok"] and not abstained,
                      abstained, report, used, timings)
