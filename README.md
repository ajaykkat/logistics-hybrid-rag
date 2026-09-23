# Logistics Hybrid RAG

A local evidence workbench for logistics operators answering questions across customer SOPs: delays, cold-chain excursions, delivery evidence, and integration failures. Built as an FDE portfolio reference implementation using **synthetic data**, not a production deployment.

## Run the demo

Python 3.12 recommended. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8082. Try “When should I escalate a delayed shipment?” Compare BM25, dense, hybrid, and hybrid with reranking. Inspect source quotes and retrieval scores side by side.

```sh
python app.py --query 'What is the free loading time before detention?'
python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python evaluate.py --output evaluation.json
```

## What is implemented

- Exact-offset paragraph chunks with document version and SHA-256 fingerprints.
- Customer and active-version filtering **before** either index is built.
- BM25 sparse retrieval, dense cosine retrieval, reciprocal rank fusion, and reranking.
- An evidence UI with source offsets, quotes, and per-stage retrieval traces.
- Citation validation that rejects fabricated text, changed sources, unknown chunks, and references outside the retrieved scope.
- Reproducible comparisons against labeled queries, plus regression tests for failure paths.

The default dense backend uses TF-IDF + truncated SVD (LSA). It is a small, inspectable dense-vector baseline, **not a pretrained semantic embedding model**. The default reranker is a deterministic feature baseline. Default answers are extracted quotes; no model API or key is required.

## Optional neural and local LLM paths

```sh
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-neural.txt
python app.py --backend neural --neural-rerank
# With an already installed and running Ollama model:
python app.py --backend neural --neural-rerank --ollama-model YOUR_INSTALLED_MODEL
python evaluate.py --backend neural --neural-rerank --output evaluation-neural.json
```

Neural mode uses SentenceTransformers `all-MiniLM-L6-v2`; neural reranking uses `cross-encoder/ms-marco-MiniLM-L6-v2`. Initial use downloads model weights. Ollama runs at localhost:11434 and selects exact evidence quotes in JSON; all selections still pass citation checks. It cannot add unchecked explanatory prose to the displayed answer.

**Validation boundary:** LSA, neural embeddings, and cross-encoder inference were executed. See `evaluation-neural-challenge.json` for the 28-question neural challenge run. Live Ollama inference remains untested; its request/response behavior has mocked contract tests.

On the challenge development set, neural hybrid fusion outperformed BM25 in document retrieval, while the cross-encoder reranker reduced ranking quality and added latency. Use `python app.py --backend neural` and select **Hybrid fusion** in the UI to explore the strongest measured retrieval configuration. Answerability remains limited; this is not a production recommendation.

## Quality upgrade

Unknown structured identifiers and quoted phrases now trigger abstention unless found in permitted source text and selected quotes. The Cedar-code false positive is fixed without inspecting other customers' documents.

Run `python compare_quality.py` for before/after results on 56 synthetic questions. See [QUALITY.md](QUALITY.md) for answer coverage and remaining errors.

## Original retrieval baseline

The checked-in `evaluation.json` contains the full run, corpus fingerprint, runtime versions, and per-query results. There are 24 answerable queries and four unanswerable queries, hand-written against this same synthetic corpus. This is a development set, not an independent benchmark.

| Mode | Recall@5 | MRR@5 | nDCG@5 | Unknown-query abstention |
|---|---:|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 1.0000 | 3/4 |
| Dense LSA | 1.0000 | 1.0000 | 0.9915 | 3/4 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 3/4 |
| Hybrid + feature reranker | 1.0000 | 1.0000 | 0.9967 | 3/4 |

**Hybrid does not outperform BM25 here.** The corpus is too small and lexical to establish a benefit, and the feature reranker slightly worsens ordering. Metrics deduplicate document IDs from the top five retrieved chunks. Timings exclude index loading and are hardware-dependent; they are not service SLOs.

The original baseline returned unrelated quotes for `CEDAR-DEMO-ONLY`. The upgrade now abstains. The historical run remains in `evaluation.json`. Cedar content is excluded, but keyword-based answerability is insufficient. Citation provenance is not proof of relevance, semantic entailment, policy correctness, or prompt-injection resistance. This failure is retained in the evaluation rather than hidden.

## Customer delivery scope

See [DELIVERY.md](DELIVERY.md) for discovery questions, demo steps, acceptance criteria, and a staged pilot proposal. The useful next experiment is a separately authored, harder query set and neural ablation—not adding more components before measuring them.

This is a loopback-only local demo. Tenant selection is a server startup option, **not authentication**. There is no production identity layer, durable audit store, scalable vector database, document connector, deployment hardening, or SLA. All business thresholds in the fixtures are fictional.

## Files

| File | Purpose |
|---|---|
| `retrieval.py` | Chunking, filtering, indexes, fusion, reranking, validation |
| `ollama_adapter.py` | Optional local model evidence selection |
| `app.py`, `index.html` | Local HTTP app and operator interface |
| `corpus.json`, `queries.json` | Synthetic source documents and relevance labels |
| `build_fixtures.py` | Rebuild the demonstration fixtures |
| `evaluate.py`, `evaluation.json` | Reproduce and inspect ablations |
| `test_retrieval.py` | 33 regression tests |
| `compare_quality.py`, `quality-comparison.json` | Before/after answerability ablation |
| `queries-challenge.json`, `QUALITY.md` | Harder development questions and error analysis |

References: [SentenceTransformers retrieve/rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html), [Ollama generate API](https://docs.ollama.com/api/generate).
