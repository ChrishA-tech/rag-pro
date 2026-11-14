from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


class BM25:
    """Okapi BM25 with an inverted index (no external dependency)."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(query_tokens):
            plist = self.postings.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1.0))
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s
