from __future__ import annotations

import math
from typing import List, Sequence


def doc_ranking(chunk_doc_ids: Sequence[str]) -> List[str]:
    """Collapse chunk-level results to a document ranking (first occurrence wins)."""
    return list(dict.fromkeys(chunk_doc_ids))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
