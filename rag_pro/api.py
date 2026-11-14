import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline
from .schema import Document


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    filters: Optional[Dict[str, Any]] = None


class DocIn(BaseModel):
    id: str
    text: str = Field(min_length=1)
    metadata: Dict[str, Any] = {}


class IngestRequest(BaseModel):
    documents: List[DocIn] = Field(min_length=1)


def create_app(pipeline: Optional[RAGPipeline] = None) -> FastAPI:
    app = FastAPI(title="RAG Pro", version="0.1.0")
    state = {"pipeline": pipeline}
    if state["pipeline"] is None:
        idx = os.getenv("RAG_INDEX_DIR", ".index")
        if Path(idx, "documents.json").exists():
            state["pipeline"] = RAGPipeline.load(idx)

    def need() -> RAGPipeline:
        if state["pipeline"] is None:
            raise HTTPException(503, "No index loaded. POST /ingest or run `rag-pro ingest`.")
        return state["pipeline"]

    @app.get("/health")
    def health():
        p = state["pipeline"]
        return {"status": "ok", "indexed_chunks": len(p.chunks) if p else 0}

    @app.post("/query")
    def query(req: QueryRequest):
        return need().query(req.question, k=req.k, mode=req.mode, filters=req.filters).to_dict()

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline()
        docs = [Document(d.id, d.text, d.metadata) for d in req.documents]
        return state["pipeline"].add_documents(docs)

    return app


app = create_app()
