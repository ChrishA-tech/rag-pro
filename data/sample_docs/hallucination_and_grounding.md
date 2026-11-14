# Hallucination and Grounding

Even with retrieval, a model can state unsupported claims. Grounding techniques keep answers tied to evidence.

## Citations and verification

Requiring a citation such as [1] for each claim lets users and automated checks verify the source. A groundedness check compares each answer sentence with the retrieved passages, using an entailment model or a language-model judge, and flags sentences that no passage supports.

## Abstention

A trustworthy assistant should sometimes refuse to answer. When the best retrieved passage has low relevance, the system should say it does not know instead of guessing, because a confident wrong answer is worse than an honest refusal. A relevance threshold tuned on a validation set controls this behaviour.

## Generation settings

Low temperature and instructions to answer only from the supplied context further reduce fabrication.
