Corpus: 8 documents, 16 chunks. 18 answerable questions, 3 unanswerable.

### Retrieval ablation (document-level)

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | ms/query |
|---|---|---|---|---|---|---|
| BM25 only | 0.86 | 0.94 | 0.94 | 0.92 | 0.92 | 0.1 |
| Dense (LSA) only | 0.81 | 0.94 | 0.94 | 0.89 | 0.90 | 0.5 |
| Hybrid (RRF) | 0.81 | 0.94 | 0.94 | 0.89 | 0.90 | 0.5 |
| Hybrid + rerank | 0.81 | 0.94 | 0.94 | 0.89 | 0.90 | 1.5 |
| Hybrid + rerank + multi-query | 0.81 | 0.94 | 0.94 | 0.89 | 0.90 | 1.9 |

### End-to-end (full pipeline, extractive generator)

| Metric | Value |
|---|---|
| Answered when answerable | 1.00 |
| Abstained when unanswerable | 1.00 |
| Answer keyword recall | 0.81 |
| Grounded answers | 1.00 |
| Answers with valid citations | 1.00 |
