# Evaluating RAG Systems

Retrieval and generation should be measured separately so failures can be located.

## Retrieval metrics

Recall at k asks whether a relevant passage appears in the top k results. Precision at k measures how many of the top k are relevant. Mean reciprocal rank, or MRR, averages the inverse of the rank of the first relevant result, so it rewards putting the right passage near the top. Normalised discounted cumulative gain, nDCG, handles graded relevance and position.

## Generation metrics

Faithfulness checks whether every claim in the answer is supported by the retrieved context. Answer relevance checks whether the answer addresses the question, and context precision checks how much of the retrieved context was actually useful.

## Golden datasets

A golden dataset is a curated set of questions with known source passages and reference answers. Running it on every change in continuous integration catches regressions when chunking, embeddings, or prompts are modified. Automated scores should be complemented by periodic human review.
