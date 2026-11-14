# Hybrid Search and Reranking

## Lexical versus semantic retrieval

BM25 is a lexical ranking function that rewards rare query terms. It excels when exact keywords matter, such as product codes, error identifiers, names, and uncommon technical words, where embeddings may blur the difference. Dense retrieval is stronger on paraphrases and vocabulary mismatch, where the question and the passage share meaning but few words.

## Reciprocal rank fusion

Hybrid search runs both retrievers and merges the rankings. Reciprocal rank fusion gives each document a score of one divided by a constant plus its rank in each list, and sums the scores. The constant is commonly set to 60. Because it uses ranks rather than raw scores, it needs no calibration between BM25 and cosine similarity.

## Reranking and diversity

A cross-encoder reranker reads the query and a passage together, so it judges relevance more accurately than comparing separate embeddings. It is slower because every pair needs a forward pass, so it is applied only to the top 20 to 100 candidates. Maximal marginal relevance, or MMR, improves results by penalising passages that repeat information already selected, which increases diversity.
