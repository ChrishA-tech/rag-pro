from __future__ import annotations

from typing import List, Tuple

from .schema import Chunk, Document
from .text import split_sentences


def chunk_document(doc: Document, chunk_size: int = 600, overlap: int = 100) -> List[Chunk]:
    """Heading-aware sentence packing with character overlap.

    - markdown headings become chunk context (prepended at index time)
    - sentences are never cut in half unless a single sentence exceeds chunk_size
    - the tail of the previous chunk (<= overlap chars) is repeated at the start of the next
    """
    units: List[Tuple[str, str]] = []   # (sentence, heading)
    heading = ""
    for para in [p.strip() for p in doc.text.replace("\r", "").split("\n\n") if p.strip()]:
        if para.startswith("#"):
            heading = para.lstrip("# ").split("\n")[0].strip()
            rest = "\n".join(para.split("\n")[1:]).strip()
            para = rest
            if not para:
                continue
        for s in split_sentences(para):
            while len(s) > chunk_size:           # hard-split pathological sentences
                units.append((s[:chunk_size], heading))
                s = s[chunk_size:]
            units.append((s, heading))

    chunks: List[Chunk] = []
    cur: List[Tuple[str, str]] = []
    size = 0

    def flush() -> None:
        nonlocal cur, size
        if not cur:
            return
        chunks.append(Chunk(
            id=f"{doc.id}::{len(chunks)}", doc_id=doc.id, text=" ".join(s for s, _ in cur),
            heading=cur[0][1], index=len(chunks), metadata=dict(doc.metadata),
        ))
        tail, total = [], 0
        for u in reversed(cur):
            if total + len(u[0]) > overlap:
                break
            tail.insert(0, u)
            total += len(u[0]) + 1
        cur, size = (tail if len(tail) < len(cur) else []), total

    for unit in units:
        if size + len(unit[0]) > chunk_size and cur:
            flush()
        cur.append(unit)
        size += len(unit[0]) + 1
    if cur and (not chunks or " ".join(s for s, _ in cur) != chunks[-1].text):
        flush()
    return chunks
