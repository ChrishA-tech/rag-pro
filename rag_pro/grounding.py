from __future__ import annotations

import re
from typing import Any, Dict, List

from .schema import Hit
from .text import split_sentences, tokenize

_CITE = re.compile(r"\[(\d+)\]")


def extract_citation_numbers(answer: str) -> List[int]:
    return list(dict.fromkeys(int(n) for n in _CITE.findall(answer)))


def grounding_report(answer: str, hits: List[Hit], threshold: float = 0.6) -> Dict[str, Any]:
    """Sentence-level support check: share of each answer sentence's terms found in a retrieved chunk.

    A cheap, deterministic proxy for faithfulness. For high-stakes use, add an NLI model or LLM judge.
    """
    chunk_tokens = [set(tokenize(h.chunk.index_text)) for h in hits]
    sentences = split_sentences(_CITE.sub("", answer))
    unsupported: List[str] = []
    supported = 0
    for s in sentences:
        toks = set(tokenize(s))
        if not toks:
            supported += 1
            continue
        best = max((len(toks & c) / len(toks) for c in chunk_tokens), default=0.0)
        if best >= threshold:
            supported += 1
        else:
            unsupported.append(s)
    invalid = [n for n in extract_citation_numbers(answer) if not 1 <= n <= len(hits)]
    ratio = supported / len(sentences) if sentences else 0.0
    return {"support_ratio": round(ratio, 3), "unsupported_sentences": unsupported,
            "invalid_citations": invalid, "ok": ratio >= 0.8 and not invalid}
