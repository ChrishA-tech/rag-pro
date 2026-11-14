from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .schema import Document

_EXT = {".txt", ".md", ".markdown", ".pdf"}


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise ImportError("PDF support needs: pip install 'rag-pro[pdf]'") from e
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return fallback


def load_path(path: str | Path) -> List[Document]:
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in _EXT) if p.is_dir() else [p]
    docs = []
    for f in files:
        text = _read(f)
        if not text.strip():
            continue
        rel = f.relative_to(p) if p.is_dir() else Path(f.name)
        doc_id = rel.with_suffix("").as_posix()
        docs.append(Document(doc_id, text, {"source": str(f), "title": _title(text, doc_id), "type": f.suffix.lower().lstrip(".")}))
    return docs


def load_paths(paths: Iterable[str | Path]) -> List[Document]:
    out: List[Document] = []
    for p in paths:
        out.extend(load_path(p))
    return out
