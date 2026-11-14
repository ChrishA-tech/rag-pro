from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_pro import RAGPipeline
from rag_pro.api import create_app
from rag_pro.schema import Document

DOCS = str(Path(__file__).resolve().parent.parent / "data" / "sample_docs")


@pytest.fixture(scope="module")
def pipe():
    p = RAGPipeline()
    p.ingest_paths([DOCS])
    return p


def test_answer_has_valid_citations(pipe):
    a = pipe.query("What constant is typical for reciprocal rank fusion?")
    assert not a.abstained and a.citations and "60" in a.answer
    assert a.grounded and not a.grounding["invalid_citations"]
    assert a.citations[0].doc_id == "hybrid_search_and_reranking"


def test_abstains_on_offtopic(pipe):
    a = pipe.query("What is the capital of France?")
    assert a.abstained and not a.citations


def test_modes_return_hits(pipe):
    for mode in ("dense", "bm25", "hybrid"):
        hits, _ = pipe.retrieve("how does hybrid search merge rankings", mode=mode, rerank=False, multi_query=False)
        assert hits


def test_metadata_filter(pipe):
    hits, _ = pipe.retrieve("embedding model", filters={"doc_id": "production_deployment"})
    assert hits and all(h.chunk.doc_id == "production_deployment" for h in hits)


def test_save_load_roundtrip(pipe, tmp_path):
    pipe.save(tmp_path)
    loaded = RAGPipeline.load(tmp_path)
    q = "How do I enforce per-user document permissions?"
    assert loaded.query(q).answer == pipe.query(q).answer


def test_add_documents_changes_answers():
    p = RAGPipeline()
    p.add_documents([Document("a", "# Pets\n\nThe office cat is called Biscuit and she sleeps on the printer."),
                     Document("b", "# Cars\n\nThe company car is a blue hatchback with a diesel engine.")])
    a = p.query("What is the office cat called?")
    assert "Biscuit" in a.answer and a.citations[0].doc_id == "a"


def test_api(pipe):
    client = TestClient(create_app(pipe))
    assert client.get("/health").json()["indexed_chunks"] > 0
    r = client.post("/query", json={"question": "What does MRR measure?", "k": 3})
    assert r.status_code == 200 and r.json()["citations"]
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_api_without_index_returns_503():
    client = TestClient(create_app(None))
    assert client.post("/query", json={"question": "hi"}).status_code in (503,)
