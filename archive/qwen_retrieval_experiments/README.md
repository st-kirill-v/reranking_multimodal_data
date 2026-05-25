# Qwen Retrieval Experiments Archive

This folder preserves the old Qwen3-VL embedding retrieval path for auditability.

Moved here:

- `scripts/build_index_qwen3_v2.py`
- `scripts/build_page_index.py`
- `scripts/build_page_index_validated.py`
- `scripts/build_qwen3_docapi_index_clean.py`
- `scripts/evaluate_qwen3_docapi_index_clean.py`
- `scripts/evaluate_retrieval.py`
- `scripts/run_page_rag.py`
- `src/mmrag/embeddings.py`
- `src/mmrag/indexing.py`
- `src/mmrag/retrieval.py`
- `src/mmrag/pipeline.py`
- `src/mmrag/diagnostics.py`
- `src/mmrag/tiling.py`

These files depend on the old `Qwen/Qwen3-VL-Embedding-2B` page embedding index.
They are not part of the current publication pipeline.

Current pipeline:

```text
ColPali/ColVision retrieval -> Nemotron VL reranking -> layout-aware context -> Qwen3-VL generation
```
