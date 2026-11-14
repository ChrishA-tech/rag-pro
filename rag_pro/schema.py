from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    heading: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for indexing: the section heading is prepended as context."""
        return f"{self.heading}. {self.text}" if self.heading else self.text


@dataclass
class Hit:
    chunk: Chunk
    score: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    heading: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    grounded: bool
    abstained: bool
    grounding: Dict[str, Any]
    retrieved: List[Hit]
    timings_ms: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["retrieved"] = [
            {"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "heading": h.chunk.heading,
             "score": round(h.score, 4), "features": {k: round(v, 3) for k, v in h.features.items()}}
            for h in self.retrieved
        ]
        return d
