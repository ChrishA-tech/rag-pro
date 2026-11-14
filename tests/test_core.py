import numpy as np

from evaluation.metrics import doc_ranking, mrr, ndcg_at_k, recall_at_k
from rag_pro.bm25 import BM25
from rag_pro.chunking import chunk_document
from rag_pro.grounding import grounding_report
from rag_pro.retrieval import mmr, rrf
from rag_pro.schema import Chunk, Document, Hit
from rag_pro.text import split_sentences, tokenize


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The retrieved passages") == tokenize("retrieve passage")
    assert "the" not in tokenize("the cat")


def test_sentence_split():
    assert len(split_sentences("One. Two! Three?")) == 3


def test_chunking_overlap_headings_and_size():
    text = "# Title\n\n## Part A\n\n" + " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_document(Document("d", text), chunk_size=200, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 260 for c in chunks)
    assert all(c.heading == "Part A" for c in chunks)
    # overlap: the first sentence of chunk n+1 appears at the end of chunk n
    first = chunks[1].text.split(". ")[0]
    assert first in chunks[0].text


def test_chunk_ids_unique():
    chunks = chunk_document(Document("d", "A b c. " * 200), chunk_size=100, overlap=20)
    assert len({c.id for c in chunks}) == len(chunks)


def test_bm25_prefers_rare_term():
    docs = [tokenize(t) for t in ["error code e1042 occurred", "general error handling guide", "another guide guide"]]
    bm = BM25(docs)
    assert int(np.argmax(bm.scores(tokenize("e1042 error")))) == 0


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    assert fused[1] > fused[3] and fused[2] > fused[4]


def test_mmr_diversifies():
    m = np.array([[1, 0], [0.99, 0.14], [0, 1]], dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    assert mmr([0, 1, 2], [1.0, 0.95, 0.6], m, 2, lam=0.5) == [0, 2]


def test_grounding_flags_unsupported_and_bad_citation():
    h = Hit(Chunk("c", "d", "Paris is the capital of France.", "", 0), 1.0)
    ok = grounding_report("Paris is the capital of France. [1]", [h])
    bad = grounding_report("Mars has two moons called Phobos and Deimos. [3]", [h])
    assert ok["ok"] and not bad["ok"] and bad["invalid_citations"] == [3]


def test_metrics():
    ranked = doc_ranking(["a", "a", "b", "c"])
    assert ranked == ["a", "b", "c"]
    assert recall_at_k(ranked, ["b", "z"], 2) == 0.5
    assert mrr(ranked, ["c"]) == 1 / 3
    assert ndcg_at_k(["a"], ["a"], 5) == 1.0
