# Embeddings and Vector Search

An embedding model maps text to a dense vector so that passages with similar meaning end up close together. Relevance is usually measured with cosine similarity, and vectors are normalised so that a dot product gives the same ranking.

## Approximate nearest neighbour indexes

Exact search compares the query with every vector, which becomes slow at scale. Approximate nearest neighbour indexes such as HNSW graphs and IVF clusters speed up lookup by trading a little recall for much lower latency. Product quantization can further shrink memory use.

## Consistency rules

Queries and documents must be embedded with the same model, otherwise the vectors live in different spaces and similarity scores are meaningless. Embedding dimension affects storage cost and speed. Domain-specific text such as legal or medical content often benefits from a fine-tuned embedding model.
