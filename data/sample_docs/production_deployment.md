# Production Deployment

## Access control and privacy

Permissions must be enforced at retrieval time. Store an access-control list or tenant identifier as chunk metadata and apply it as a filter on every query, so users never see passages they are not allowed to read. Redact personal data before indexing where possible.

## Monitoring

Track retrieval hit rate, the share of questions answered or abstained, latency per stage, token cost, and user feedback such as thumbs up or down. Log the retrieved chunk identifiers with every answer so problems can be traced and replayed.

## Index maintenance

Documents change, so the index needs incremental updates and deletion of stale chunks. If you swap the embedding model you must re-embed the whole corpus, because old and new vectors are not comparable. Version the index together with the model name. Caching frequent queries and embeddings reduces latency and cost.
