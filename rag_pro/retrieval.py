from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder
from .schema import Chunk, Hit
from .text import tokenize


def rrf(rank_lists: Sequence[Sequence[int]], k: int = 60, weights: Optional[Sequence[float]] = None) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rank_lists)
    fused: Dict[int, float] = {}
    for w, ranking in zip(weights, rank_lists):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return fused


def mmr(cand: List[int], rel: List[float], matrix: np.ndarray, k: int, lam: float) -> List[int]:
    """Maximal Marginal Relevance: trade relevance against redundancy with already-picked chunks."""
    picked: List[int] = []
    remaining = list(range(len(cand)))
    while remaining and len(picked) < k:
        best, best_val = remaining[0], -1e9
        for r in remaining:
            red = max((float(matrix[cand[r]] @ matrix[cand[p]]) for p in picked), default=0.0)
            val = lam * rel[r] - (1 - lam) * red
            if val > best_val:
                best, best_val = r, val
        picked.append(best)
        remaining.remove(best)
    return [cand[i] for i in picked]


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, matrix: np.ndarray, rrf_k: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = matrix
        self.bm25 = BM25([tokenize(c.index_text) for c in chunks])
        self.rrf_k = rrf_k

    def _mask(self, filters: Optional[Dict]) -> np.ndarray:
        if not filters:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([all(c.metadata.get(k, c.doc_id if k == "doc_id" else None) == v for k, v in filters.items())
                         for c in self.chunks])

    @staticmethod
    def _top(scores: np.ndarray, n: int, mask: np.ndarray, require_positive: bool) -> List[int]:
        s = np.where(mask, scores, -np.inf)
        order = np.argsort(-s)[:n]
        return [int(i) for i in order if np.isfinite(s[i]) and (s[i] > 0 or not require_positive)]

    def search_many(self, queries: Sequence[str], n: int = 30, mode: str = "hybrid",
                    filters: Optional[Dict] = None, mmr_lambda: Optional[float] = None) -> List[Hit]:
        """Retrieve for one or more query variants and fuse every ranking with RRF.

        queries[0] is the original question (weight 1.0); extra variants get weight 0.6.
        """
        mask = self._mask(filters)
        lists: List[List[int]] = []
        weights: List[float] = []
        for qi, q in enumerate(queries):
            w = 1.0 if qi == 0 else 0.6
            if mode in ("dense", "hybrid"):
                qv = self.embedder.embed([q])[0]
                lists.append(self._top(self.matrix @ qv, n, mask, require_positive=False))
                weights.append(w)
            if mode in ("bm25", "hybrid"):
                lists.append(self._top(self.bm25.scores(tokenize(q)), n, mask, require_positive=True))
                weights.append(w)
        fused = rrf(lists, self.rrf_k, weights)
        ranked = sorted(fused, key=fused.get, reverse=True)[:n]
        if not ranked:
            return []
        top = fused[ranked[0]]
        if mmr_lambda is not None:
            ranked = mmr(ranked, [fused[i] / top for i in ranked], self.matrix, n, mmr_lambda)
        return [Hit(self.chunks[i], fused[i] / top) for i in ranked]
