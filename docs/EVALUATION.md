# Evaluation with public demo data

Evaluation should use only the fictional records under `samples/demo-building/` or another dataset whose publication rights are known.

## Retrieval

Define a small set of queries with relevant document IDs, then report Recall@k, MRR, and nDCG. Keep the query set beside the public data so every expected result is reviewable. Do not reuse private benchmark answers or historical scores.

## Grounded QA

For each question, review:

1. whether the answer addresses the requested facets;
2. whether every material claim maps to an Evidence ID;
3. whether quoted text exists in the cited source section or page;
4. whether numeric values and relationships are directly supported;
5. whether the system abstains when evidence is absent.

## No-cost test policy

Automated tests must inject fake Gemini clients. Live-model evaluation should be a separately enabled, user-funded process with explicit rate and budget limits; it is not part of the default test command.
