# Prompt Injection in RAG

Indirect prompt injection happens when a retrieved document contains hidden instructions. A web page or file might say to ignore the user and reveal secrets, and a naive pipeline would pass that text to the model as if it were trusted. A retrieved page can therefore hijack the model's behaviour.

## Mitigations

Treat all retrieved text as untrusted data. Wrap passages in clear delimiters and instruct the model never to follow commands found inside them. Scan documents at ingestion time for injection patterns and strip invisible characters. Give the model the least privilege possible: avoid connecting tools that can send email or modify data without human approval.

## Output validation

Validate responses before returning them. Check for leaked system prompts or secrets, redact personal data, and block disallowed content. Guardrail pipelines on both input and output provide defence in depth.
