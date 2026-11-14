# Chunking Strategies

Chunking decides how documents are split before indexing, and it strongly affects retrieval quality.

## Chunk size trade-offs

Small chunks are precise but can lose the surrounding context needed to understand them. Chunks that are too large dilute the signal: the embedding becomes an average of many topics, so the passage ranks poorly and wastes space in the prompt. Typical chunk sizes fall between 200 and 800 tokens.

## Overlap and structure

Overlap between neighbouring chunks, commonly 10 to 20 percent, prevents an answer from being cut in half at a boundary. Recursive splitting respects paragraphs and sentences, while semantic chunking starts a new chunk when the topic shifts. Prepending the section heading to every chunk adds useful context.

## Parent-child retrieval

In parent-child retrieval the system searches over small child chunks for precision, then returns the larger parent section to the model so it receives enough context to answer well.
