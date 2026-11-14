from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class Settings:
    # chunking
    chunk_size: int = 600          # characters
    chunk_overlap: int = 100
    # retrieval
    top_k: int = 5
    candidates: int = 30           # per retriever, before fusion
    retrieval_mode: str = "hybrid"  # dense | bm25 | hybrid
    rrf_k: int = 60
    rerank: bool = True
    multi_query: bool = True
    mmr_lambda: float | None = None   # e.g. 0.7 to diversify; None = off
    # answering
    min_coverage: float = 0.25     # abstain if best chunk covers less of the question than this
    max_context_chars: int = 6000
    # components
    embedder: str = "lsa"           # lsa | sentence-transformers
    embedding_dim: int = 128
    generator: str = "extractive"   # extractive | anthropic
    model: str = "claude-sonnet-4-6"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.embedder = os.getenv("RAG_EMBEDDER", s.embedder)
        s.generator = os.getenv("RAG_GENERATOR", s.generator)
        s.model = os.getenv("RAG_MODEL", s.model)
        s.top_k = int(os.getenv("RAG_TOP_K", s.top_k))
        return s

    def to_dict(self) -> dict:
        return asdict(self)
