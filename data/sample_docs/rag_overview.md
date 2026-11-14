# Retrieval-Augmented Generation Overview

Retrieval-augmented generation, or RAG, combines a search step with a language model. When a question arrives, the system retrieves relevant passages from a knowledge base and places them in the prompt before the model writes its answer.

## Why use RAG

Grounding the model in retrieved passages reduces made-up answers, because the model can copy facts from evidence instead of relying on memory. RAG also lets a system use fresh or private knowledge without retraining the model, and it makes source citations possible so users can verify claims.

## Core pipeline

A typical pipeline has an offline indexing phase and an online query phase. Indexing loads documents, splits them into chunks, and stores both vector embeddings and a keyword index. At query time the system retrieves candidates, reranks them, builds a prompt with numbered passages, and generates a cited answer.
