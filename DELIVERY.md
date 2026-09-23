# Discovery to pilot

## Customer problem

An operations specialist needs the applicable escalation procedure while handling a delayed or damaged shipment. Procedures vary by customer and version. A plausible answer citing a different customer's policy can cause the wrong operational action.

## Discovery questions

1. Which decisions must remain with an operator, and what is the cost of an incorrect answer?
2. Who approves SOP updates, and which source records the effective version?
3. How does identity map to customer access, including shared/public policies?
4. Which real questions can operators label without disclosing customer data?
5. What counts as a successful lookup: correct document, exact clause, or completed action?

## Five-minute demonstration

1. Run the default app and ask about delayed shipment escalation. Check the current policy quote, version, and offsets.
2. Compare four retrieval modes. Explain lexical matching versus latent vector similarity and why this dataset does not demonstrate hybrid gains.
3. Ask an unrelated question, such as a chocolate cake recipe, to show abstention.
4. Ask about CEDAR-DEMO-ONLY. Discuss the known false-positive answerability failure without implying a customer-data leak.
5. Run tests to demonstrate fabricated citations, stale source content, and out-of-scope references are rejected.
6. Explain the optional model boundary: models choose evidence; verification checks exact provenance, not truth.

## Proposed pilot acceptance criteria (not achieved claims)

- Build a held-out set with operators: paraphrases, exact IDs, stale documents, ambiguous requests, and genuinely unanswerable questions.
- Select retrieval approach using labeled relevance and cost/latency evidence; retain BM25 if hybrid adds no value.
- Agree answerability and incorrect-answer tolerances before tuning thresholds. Report confidence intervals and slice-level errors.
- Require authenticated server-side scope mapping before handling any real customer material.
- Require document ownership, version approval, deletion propagation, and index refresh checks.
- Measure evidence relevance separately from exact citation validity.

## Rollout and support

Start read-only in shadow mode. Have operators compare evidence to their approved sources; record disagreements with consent. Gate pilot expansion on customer review of errors, access tests, and measured latency under expected load. Keep a rollback to the prior approved index and disable generated responses independently of search. A production integration would need secrets management, audit retention policy, monitoring, rate limits, and a real serving stack before external exposure.

No rollout, customer deployment, or production outcome is claimed by this repository.
