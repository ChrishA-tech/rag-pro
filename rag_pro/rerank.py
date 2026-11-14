from __future__ import annotations

from typing import List, Protocol

from .schema import Hit
from .text import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, hits: List[Hit], top_n: int) -> List[Hit]: ...


def _bigrams(toks: List[str]) -> set:
    return set(zip(toks, toks[1:]))


class LexicalReranker:
    """Cheap offline reranker: query-term coverage + phrase (bigram) match + fused retrieval score.

    `features["coverage"]` (share of query terms present in the chunk) doubles as the
    absolute relevance signal used for abstention.
    """

    def __init__(self, w_cov: float = 0.55, w_bi: float = 0.15, w_ret: float = 0.30):
        self.w = (w_cov, w_bi, w_ret)

    def rerank(self, query: str, hits: List[Hit], top_n: int) -> List[Hit]:
        q = tokenize(query)
        qset, qbi = set(q), _bigrams(q)
        out = []
        for h in hits:
            c = tokenize(h.chunk.index_text)
            cov = len(qset & set(c)) / len(qset) if qset else 0.0
            bi = len(qbi & _bigrams(c)) / len(qbi) if qbi else 0.0
            score = self.w[0] * cov + self.w[1] * bi + self.w[2] * h.score
            out.append(Hit(h.chunk, score, {"coverage": cov, "bigram": bi, "retrieval": h.score}))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_n]


class CrossEncoderReranker:
    """Production reranker (needs `pip install 'rag-pro[st]'`). Scores (query, passage) pairs jointly."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy
        self.model = CrossEncoder(model)
        self._lex = LexicalReranker()

    def rerank(self, query: str, hits: List[Hit], top_n: int) -> List[Hit]:
        scores = self.model.predict([(query, h.chunk.index_text) for h in hits])
        feats = {id(h): self._lex.rerank(query, [h], 1)[0].features for h in hits}
        ranked = sorted(zip(hits, scores), key=lambda x: -float(x[1]))[:top_n]
        return [Hit(h.chunk, float(s), feats[id(h)]) for h, s in ranked]
