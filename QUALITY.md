# Quality upgrade: measured answerability

The original tokenizer split CEDAR-DEMO-ONLY into ordinary words, matching unrelated demo-policy text. A new literal gate requires uppercase compound identifiers, underscore fields, and quoted phrases to occur in a permitted chunk and selected quote. Unknown literals cause abstention before any model call. Excluded customer documents are never consulted.

This is a conservative rule, not a learned relevance classifier. Lowercase hyphenated prose is not treated as an identifier. Requiring every literal in one chunk may reject legitimate questions spanning multiple chunks.

## Reproduce

```sh
python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python compare_quality.py
python evaluate.py --queries queries-challenge.json --output evaluation-challenge.json
```

The ablation disables only the literal gate. Retrieval, corpus, labels and 40% lexical coverage threshold stay fixed. `quality-comparison.json` contains four retrieval modes, all query outcomes, and query/corpus fingerprints. `evaluation.json` preserves the original historical run.

## LSA hybrid with feature reranking

| Measure | Coverage only | Literal gate |
|---|---:|---:|
| Original unanswerable questions rejected | 3/4 | 4/4 |
| Original answerable questions receiving quotes | 20/24 | 20/24 |
| Challenge unanswerable questions rejected | 3/10 | 6/10 |
| Challenge answerable questions receiving quotes | 5/18 | 5/18 |
| Challenge document Recall@5 | 0.9444 | 0.9444 |
| Challenge document MRR@5 | 0.8472 | 0.8472 |
| Challenge document nDCG@5 | 0.8718 | 0.8718 |

Document-level citation precision on answered answerable questions is 1.0. This weak proxy excludes unanswerable cases and does not check whether the selected passage contains the requested fact. It is not overall answer accuracy.

## Remaining failures

Four challenge questions still receive quotes despite missing facts: exact invoice tolerance, live shipment location, the insurance payer, and the lane's GPS reporting interval. Most answerable paraphrases are rejected by lexical coverage even when retrieval finds the right document. Better retrieval alone will not solve that answerability bottleneck.

Next evaluation should label passage-level support, calibrate a relevance/entailment gate on a development split, then measure a frozen test split. Do not lower a threshold merely to increase response coverage.

## Dataset limitations

There are 28 original and 28 challenge questions. The challenge set has 18 answerable and 10 unanswerable cases covering paraphrases, multiple documents, identifiers, customer scope, and missing facts. Both sets were authored against the synthetic corpus by the same development workflow. Neither is independent, held-out, or customer-validated. Counts are small; results are descriptive, not generalization claims.

## Live neural experiment

CPU inference completed with sentence-transformers 3.4.1, transformers 4.48.3, torch 2.6.0+cpu, and huggingface-hub 0.36.2. The source pins the exact model revisions recorded in `evaluation-neural-challenge.json`. No hosted API or customer data was used. Ollama remains outside the exercised path.

| Mode | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| BM25 | 0.9444 | 0.8333 | 0.8562 |
| MiniLM dense | 1.0000 | 0.8981 | 0.9239 |
| Neural hybrid fusion | 1.0000 | 0.9722 | 0.9795 |
| Neural hybrid + cross-encoder | 1.0000 | 0.9213 | 0.9361 |

Hybrid fusion was the strongest ranking configuration in this small development experiment. Cross-encoder reranking reduced nDCG and increased median query time from about 14 ms to 377 ms on this machine. These are observed timings, not deployment SLOs. The implementation calculates both sparse and dense trace scores even in single-channel modes, so BM25 timing in this report is not standalone BM25 cost. Index/model loading is excluded.

All four neural-run modes answered only 5/18 answerable questions and abstained on 6/10 unanswerable questions. The lexical gate still controls answerability, independently of improved retrieval. There is no demonstrated end-to-end answer-quality gain from neural retrieval here.

```sh
python evaluate.py --backend neural --neural-rerank \
  --queries queries-challenge.json --output evaluation-neural-challenge.json
```
