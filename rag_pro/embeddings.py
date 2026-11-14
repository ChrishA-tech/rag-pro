from __future__ import annotations

from typing import List, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import tokenize


class Embedder(Protocol):
    def fit(self, texts: List[str]) -> "Embedder": ...
    def embed(self, texts: List[str]) -> np.ndarray: ...


def _analyze(text: str) -> List[str]:
    toks = tokenize(text)
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def _l2(m: np.ndarray) -> np.ndarray:
    return (m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


class LSAEmbedder:
    """Offline dense embeddings: TF-IDF (uni+bigrams) -> truncated SVD (latent semantic analysis).

    Needs no model download. Swap for SentenceTransformerEmbedder for production-grade semantics.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vec = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        x = self.vec.fit_transform(texts)
        n = max(1, min(self.dim, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=0).fit(x)
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        assert self.svd is not None, "call fit() first"
        return _l2(self.svd.transform(self.vec.transform(texts)))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("needs: pip install 'rag-pro[st]'") from e
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        return _l2(np.asarray(self.model.encode(texts, show_progress_bar=False)))


def make_embedder(kind: str, dim: int, st_model: str) -> Embedder:
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(st_model)
    return LSAEmbedder(dim)
