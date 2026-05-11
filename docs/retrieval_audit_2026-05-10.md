# Multimodal Page Retrieval Audit

Date: 2026-05-10

## Executive Summary

The main failure mode is not a single FAISS parameter. The repository had multiple incompatible
retrieval stacks writing or expecting similarly named artifacts:

- `build_index_qwen3.py` used `AutoModel` with naive `last_hidden_state.mean(dim=1)`.
- `build_index_qwen3_v2.py` used `SentenceTransformer.encode` for `Qwen/Qwen3-VL-Embedding-2B`.
- `full_pipeline_only_pages.py` encoded queries with the old `AutoModel` path.
- `full_pipeline_only_pages_only_embedder.py` encoded queries with `SentenceTransformer.encode`.
- local `index/` did not contain `pages_qwen3.index`; it contained CLIP and SigLIP indexes.
- `src/core/multimodal_search.py` still described JinaCLIP and looked for `pages_jina.index`.

This makes query/document encoder mismatch highly likely. A Qwen text query searched against a
CLIP/SigLIP or old mean-pooled Qwen index will produce tiny, unseparated scores and layout-like
random results.

## High-Impact Issues

1. Index provenance was missing.
   There was no manifest describing model id, backend, dimensionality, query prompt, normalization,
   or data path. It was too easy to reuse a stale index.

2. Encoder contract was inconsistent.
   The old Qwen script used unsupported mean pooling over all tokens, including image/text/padding
   tokens. The newer script used the model's SentenceTransformer wrapper. Those are different
   embedding spaces.

3. Active code mixed page-image retrieval with text retrieval experiments.
   BM25, E5, JinaCLIP, Docling text, full-page Qwen, and Nemotron experiments were all in scripts
   named like active production paths.

4. Full-page document images are high resolution and text-heavy.
   Pages are roughly 1191x1684. Any model-side processor that tiles/downsamples aggressively can
   lose small text. This should be measured, not guessed.

5. Generator images were downscaled to a max long edge of 800.
   That is likely too small for tables and dense academic pages. The generator now defaults to 1600.

6. Evaluation imported heavy models twice.
   `evaluate_rag_v2.py` created a Qwen generator and then imported a pipeline that also created one.

7. Windows-incompatible timeout logic existed in active scripts.
   `signal.SIGALRM` does not exist on Windows, so active scripts were not portable.

8. Packaging was broken.
   `pyproject.toml` listed only `packages = ["src"]`, while code imports nested modules.

## Root-Cause Hypotheses for Bad Retrieval

Ranked by likelihood:

1. Stale or wrong index artifact.
   Local artifacts are CLIP/SigLIP, while the active Qwen scripts expect `pages_qwen3.index`.

2. Query/document mismatch.
   Index built by old `AutoModel + mean pooling` and query encoded by `SentenceTransformer`, or the
   reverse.

3. Wrong pooling strategy in old index builder.
   `last_hidden_state.mean(dim=1)` is not a safe embedding extraction strategy for Qwen3-VL
   embedding models.

4. Full-page small-text compression.
   Even with the correct wrapper, full-page images may be encoded mostly by layout and large visual
   structure. This must be tested using crop/tiling ablations.

5. Score collapse from non-normalized or incorrectly normalized vectors.
   The new index contract records normalization and uses IP only with normalized vectors.

6. Reranker cannot recover if first-stage recall is bad.
   Nemotron rerank only sees first-stage candidates. If the correct page is absent from top-k, rerank
   cannot fix retrieval.

## New Structure

The clean research pipeline lives in `src/mmrag`:

- `config.py`: explicit paths/model/index/retrieval/reranker/generator configs.
- `dataset.py`: DocBench page and question loading.
- `embeddings.py`: one Qwen3 SentenceTransformer encoder contract.
- `indexing.py`: FAISS index build/load/search with manifest.
- `retrieval.py`: page-only vector retrieval and optional neighbor expansion.
- `rerank.py`: Nemotron VL reranking.
- `pipeline.py`: retrieval/rerank/generation orchestration.
- `diagnostics.py`: vector and score distribution checks.

CLI entrypoints:

- `scripts/build_page_index.py`
- `scripts/diagnose_retrieval.py`
- `scripts/run_page_rag.py`

Compatibility wrappers:

- `scripts/build_index_qwen3_v2.py`
- `scripts/full_pipeline_only_pages_only_embedder.py`

## Required Validation Experiments

1. Rebuild the Qwen3 page index with the new builder.
   Confirm manifest says `Qwen/Qwen3-VL-Embedding-2B`, backend `sentence-transformers`, normalized
   true, and dim matches FAISS.

2. Run retrieval-only diagnostics for known folder questions.
   Measure whether the correct document folder appears in top-1/top-5/top-30 before reranking.

3. Compare full-page vs crops/tiles.
   Build indexes for full page, top/bottom halves, 2x2 tiles, and high-DPI page renders. The goal is
   to test whether small text is lost.

4. Query prompt ablation.
   Compare no prompt, current prompt, and model-recommended query prompt while keeping document
   embeddings fixed.

5. Embedding collapse checks.
   Inspect vector norm min/mean/max, component std, duplicate rows, top1-top5 score gaps, and domain
   distribution for NLP questions.

6. Reranker recall boundary.
   For each question, report if the correct page exists before rerank. If not, reranker changes are
   irrelevant.

7. Generator readability ablation.
   Compare max image long edge 800 vs 1200 vs 1600 for dense table pages after retrieval is correct.
