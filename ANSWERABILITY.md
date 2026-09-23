# Answerability experiment

## Run locally

Install the neural requirements from the README, then:

```sh
python app.py --backend neural --qa-reader
# Select Hybrid fusion in the UI, or use the CLI:
python app.py --backend neural --qa-reader --mode hybrid \
  --query 'What is the exact configured invoice tolerance amount?'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_answerability.py
```

This optional mode uses a local extractive QA model with an explicit no-answer prediction. It bypasses lexical coverage, retains identifier and customer-scope checks, and returns only a verified source sentence. The UI shows evidence, not a synthesized operational instruction. Initial model download is required; subsequent inference is local. `--qa-reader` and `--ollama-model` are mutually exclusive.

## Why restrict the reader

Taking the highest QA score across five retrieved chunks produced confident responses from unrelated documents. QA span confidence is not a calibrated document relevance score. The default QA reader therefore considers only retrieved chunks belonging to the highest-ranked eligible document. Unknown literals still cause abstention before inference. Predictions must meet the fixed 0.25 score threshold, match exact source offsets, and pass existing citation validation. Null predictions are normal abstentions.

The threshold was set before measurement. Document restriction was developed after inspecting failures on these same questions. These are development results, not held-out validation or deployment guarantees.

## Results with neural hybrid fusion

`answerability-comparison.json` includes every query, quote, reader score, null prediction, and abstention. The comparison holds retrieval constant.

| Dataset / metric | Lexical gate | QA across candidates | QA within top document |
|---|---:|---:|---:|
| Original: answerable questions receiving evidence | 20/24 | 18/24 | 14/24 |
| Original: unknown questions rejected | 4/4 | 4/4 | 4/4 |
| Original: responses citing a wrong document | 0 | 7 | 0 |
| Challenge: answerable questions receiving evidence | 5/18 | 12/18 | 10/18 |
| Challenge: unknown questions rejected | 6/10 | 8/10 | 10/10 |
| Challenge: responses citing a wrong document | 0 | 4 | 0 |

Wrong-document counts only cover answerable questions. **Response coverage is not accuracy.** A correct document can contain the wrong clause, and one quote may cover only part of a question. The reader also loses six responses on the original set compared with lexical coverage. This mode stays opt-in.

## Passage review: remaining failures

Manual development review of the returned top-document QA quotes found:

| Query | Observed problem |
|---|---|
| q02: unacknowledged carrier exception | Cites initial dispatcher notification instead of the overdue-acknowledgement escalation clause. |
| c01: truck over two hours late | Cites the later acknowledgement escalation instead of the initial delay action. |
| c13: does missing telemetry prove cargo loss? | Cites an action to request an update, omitting the clause rejecting that inference. |
| q17: customs owner and required details | Provides the owner but omits the requested details. |
| q23, c15, c16: two-procedure questions | Provides only one procedure. |

These are explicit counterexamples to treating document precision as answer correctness. `answerability-review.json` records these observations; it is a development review, not an independent adjudication. No end-to-end correctness percentage is claimed.

## Design boundaries and next gate

- Source verification proves provenance, not semantic support or safe operational action.
- A top-document restriction intentionally cannot provide a complete multi-document answer.
- Returning a whole sentence preserves nearby negation but does not establish that it answers the question.
- The model's score is uncalibrated; 0.25 is an experimental threshold, not a probability guarantee.
- A rollout would need passage-level gold labels, a separately authored frozen test set, and review of incorrect-clause and partial-answer rates. Do not deploy based on these small synthetic counts.

## Model attribution

Uses [deepset/minilm-uncased-squad2](https://huggingface.co/deepset/minilm-uncased-squad2), authored by deepset, trained for extractive QA on SQuAD 2.0. Model license: CC BY 4.0, separate from this repository's MIT code license. Revision `934656cdda79824eabf503ed56e15c01ddbdbe3f` is pinned in `qa_reader.py`. Model weights are downloaded unchanged and are not redistributed in this repository. See the linked model card for authors and training details.
