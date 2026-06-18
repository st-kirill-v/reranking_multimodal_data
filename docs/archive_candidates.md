# Archive Candidates

This document records cleanup decisions made before moving files into `archive/`.

## Moved During This Cleanup

| Path | Target | Reason |
| --- | --- | --- |
| `_archive/` | `archive/old_scripts/_archive_legacy_root/` | Legacy archive directory at repository root. It duplicates the purpose of `archive/` and contains old scripts/tool caches, not active pipeline code. |
| `archived/` | `archive/old_scripts/archived_legacy_root/` | Legacy archive directory at repository root. Keeping it under the canonical `archive/` tree makes the root cleaner. |
| `load_llm.py` | `archive/old_scripts/load_llm.py` | Standalone LLM loading scratch script. It is not imported by current scripts or `src/`. |
| `index_test/` | `archive/debug_or_smoke_runs/index_test/` | Old test index/debug artifact. Current configs use `index/`, `index_colpali_v1_3_merged/`, or generated text indexes, not `index_test/`. |
| `reranking_multimodal_data.egg-info/` | `archive/debug_or_smoke_runs/reranking_multimodal_data.egg-info/` | Local Python packaging/build artifact. It should not live in the publication root. |
| `reports/experiment_summary/figures` | `archive/debug_or_smoke_runs/misnamed_report_figures/experiment_summary_figures.png` | Misnamed local PNG file without extension, likely produced by an interrupted report download/copy. It is not the intended `figures/` directory. |
| `reports/experiment_summary/paper_sections/figures` | `archive/debug_or_smoke_runs/misnamed_report_figures/paper_sections_figures.png` | Misnamed local PNG file without extension, likely produced by an interrupted report download/copy. It is not the intended `figures/` directory. |

## Kept In Place

| Path | Reason |
| --- | --- |
| `scripts/evaluate_full_pipeline_layout_aware_clean.py` | Main full multimodal evaluation script used by `scripts/run_experiment.py`. |
| `scripts/run_experiment.py` | Config-based entrypoint for current experiments. |
| `scripts/build_experiment_summary_tables.py` | Current paper-facing table/figure generator. |
| `scripts/build_colvision_index_clean.py` | Needed to rebuild the ColPali/ColVision visual index. |
| `scripts/build_nemotron_image_index_clean.py` | Needed to rebuild the Nemotron image retrieval index. |
| `scripts/build_docbench_text_encoder_index.py` | Needed to reproduce text-evidence encoder baselines. |
| `scripts/evaluate_text_reranker_308.py` | Needed to reproduce BM25/text reranker baselines. |
| `scripts/extract_docbench_caption_table_text.py` | Needed to rebuild caption/table evidence fields. |
| `scripts/extract_docbench_ocr.py` | Needed to rebuild OCR evidence fields. |
| `configs/experiments/*.yaml` | Kept because configs are small, document experiment provenance, and several are referenced by current reports or README. Older uncertain configs should be reviewed in a later pass only after paper tables are frozen. |
| `reports/tables/` and `reports/experiment_summary/` | Publication-facing tables, markdown summaries, and figures. |

## Needs Later Review

| Path | Reason |
| --- | --- |
| `configs/experiments/baseline.yaml`, `current_full_pipeline.yaml`, `full_colpali_layout_aware.yaml`, `layout_aware.yaml`, `multimodal_reranking.yaml`, `rerank_multimodal.yaml` | Likely older configs, but kept because they are small and may still be useful for reproducing earlier baselines. |
| `scripts/evaluate_docbench_hybrid_bm25.py` | Hybrid 1102/text-meta route is not the current paper focus, but `run_experiment.py` still contains a branch for this evaluator. Kept to avoid breaking that branch. |
| `scripts/generate_error_analysis.py`, `generate_metrics_tables.py`, `generate_plots.py`, `generate_report.py`, `evaluate.py` | Lightweight reporting/evaluation entrypoints. Kept because README references them and they are not harmful. |
| `.env`, `custodian_ed25519` | Local sensitive files. They are ignored by `.gitignore`; they were not moved into `archive/` to avoid accidentally publishing secrets. Remove them locally before any public release. |
