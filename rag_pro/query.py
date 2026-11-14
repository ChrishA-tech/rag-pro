from __future__ import annotations

from typing import List, Optional

from .llm import LLM
from .text import tokenize


def expand_query(question: str, llm: Optional[LLM] = None, max_variants: int = 3) -> List[str]:
    """Multi-query expansion. Always keeps the original first.

    Offline: adds a keyword-only variant. With an LLM: adds paraphrases (better recall on vocabulary mismatch).
    """
    variants = [question.strip()]
    keywords = " ".join(dict.fromkeys(tokenize(question)))
    if keywords and keywords != variants[0].lower():
        variants.append(keywords)
    if llm is not None:
        prompt = (f"Rewrite the search query below in {max_variants} different ways using different vocabulary. "
                  f"One per line, no numbering, no commentary.\n\nQuery: {question}")
        try:
            lines = [ln.strip("-• ").strip() for ln in llm.complete("You rewrite search queries.", prompt).splitlines()]
            variants += [ln for ln in lines if ln][:max_variants]
        except Exception:
            pass  # expansion is best-effort
    return list(dict.fromkeys(variants))
