# Legacy Ablation Scripts

This folder preserves old staged evaluation scripts that are no longer the main publication
pipeline.

Moved here:

- `scripts/analyze_vlm_errors_clean.py`
- `scripts/compare_layout_aware_ablation_clean.py`
- `scripts/compare_vlm_ablation_clean.py`
- `scripts/evaluate_colvision_index_clean.py`
- `scripts/evaluate_colvision_nemotron_rerank_clean.py`
- `scripts/evaluate_colvision_oracle_pages_clean.py`
- `scripts/evaluate_nemotron_rerank_from_candidates_clean.py`
- `scripts/evaluate_vlm_from_page_candidates_clean.py`
- `scripts/evaluate_vlm_layout_aware_from_page_candidates_clean.py`
- `scripts/export_colvision_candidates_clean.py`

Why archived:

- they run cached-candidate or diagnostic workflows;
- they were useful for prompt, crop, retrieval, and reranker ablations;
- they are not required to run the current end-to-end ColPali pipeline.

Current publication pipeline:

```text
scripts/build_colvision_index_clean.py
scripts/evaluate_full_pipeline_layout_aware_clean.py
scripts/run_experiment.py --config configs/experiments/full_colpali_layout_aware.yaml
```

Shared code extracted from the archived scripts now lives in:

- `src/retrieval/colvision.py`
- `src/evaluation/vlm_eval.py`
- `src/cropping/layout_aware_eval.py`
