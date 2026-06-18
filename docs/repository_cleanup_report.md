# Repository Cleanup Report

Cleanup date: 2026-06-17

## Goal

Prepare `reranking_multimodal_data` for GitHub publication without breaking the current multimodal DocBench research pipeline or losing historical code. The cleanup follows the rule: do not delete old work; move clearly unused material into `archive/` and document uncertain candidates.

## What Changed

- Rewrote `README.md` as a publication-facing research README with:
  - problem statement;
  - dataset layout;
  - method overview;
  - reproducible setup and run commands;
  - main experiments;
  - best result table;
  - report/figure generation instructions;
  - archive and future-work notes.
- Updated `.gitignore` to keep local data, model caches, indexes, results, secrets, and Python build artifacts out of Git while allowing paper-facing reports under `reports/`.
- Removed already-tracked local cache artifacts from the Git index with `git rm --cached` while keeping them on disk:
  - `cache/modules.npz`
  - `cache/modules/e5_reranker_embeddings.npz`
- Created a canonical archive structure:
  - `archive/old_scripts/`
  - `archive/old_configs/`
  - `archive/old_results/`
  - `archive/debug_or_smoke_runs/`
  - `archive/deprecated_notebooks/`
- Added `docs/archive_candidates.md` with moved files, kept files, and items that need later review.

## Moved To Archive

| Original path | New path | Reason |
| --- | --- | --- |
| `_archive/` | `archive/old_scripts/_archive_legacy_root/` | Root-level legacy archive duplicated the purpose of `archive/`. |
| `archived/` | `archive/old_scripts/archived_legacy_root/` | Root-level legacy archive moved under the canonical archive tree. |
| `load_llm.py` | `archive/old_scripts/load_llm.py` | Standalone LLM scratch loader, not imported by active scripts or `src/`. |
| `index_test/` | `archive/debug_or_smoke_runs/index_test/` | Old test/debug index artifact, not used by current configs. |
| `reranking_multimodal_data.egg-info/` | `archive/debug_or_smoke_runs/reranking_multimodal_data.egg-info/` | Local Python packaging artifact. |

No files were permanently deleted.

## Main Files Kept Active

### Experiment entrypoints

- `scripts/run_experiment.py`
- `scripts/evaluate_full_pipeline_layout_aware_clean.py`
- `scripts/evaluate_text_reranker_308.py`

### Index and evidence builders

- `scripts/build_colvision_index_clean.py`
- `scripts/build_nemotron_image_index_clean.py`
- `scripts/build_docbench_text_encoder_index.py`
- `scripts/extract_docbench_caption_table_text.py`
- `scripts/extract_docbench_ocr.py`

### Reporting

- `scripts/build_experiment_summary_tables.py`
- `scripts/generate_metrics_tables.py`
- `scripts/generate_report.py`
- `src/reporting/`

### Core modules

- `src/retrieval/`
- `src/reranking/`
- `src/generation/`
- `src/evaluation/`
- `src/reporting/`
- `src/mmrag/`

## Publication-Facing Results Kept

The following report locations are intentionally kept in the main tree:

- `reports/tables/`
- `reports/experiment_summary/`
- `reports/experiment_summary/figures/`
- `reports/experiment_summary/paper_sections/`
- `reports/experiment_summary/paper_sections/figures/`

These contain the tables and plots used for the paper-style experiment analysis.

## Commands Checked

The cleanup was designed around these primary commands:

```bash
python scripts/build_experiment_summary_tables.py
python scripts/run_experiment.py --config configs/experiments/image_text_full_308_nemotron_qwen3vl30b.yaml --dry-run
```

Additional syntax checks were run for the main scripts where possible:

```bash
python -m py_compile scripts/build_experiment_summary_tables.py scripts/run_experiment.py scripts/evaluate_full_pipeline_layout_aware_clean.py
```

Local outcomes:

- `py_compile` for main experiment/reporting scripts passed.
- `python scripts/build_experiment_summary_tables.py` passed and regenerated CSV/MD tables. Plot generation was skipped locally because `matplotlib`/`seaborn` are not installed in the local `.venv`.
- `python scripts/run_experiment.py --config configs/experiments/image_text_full_308_nemotron_qwen3vl30b.yaml --dry-run` passed and produced the expected full image+text Nemotron/Qwen3-VL-30B command.
- `python -m pytest tests` could not run because `pytest` is not installed in the local `.venv`.

## Remaining Risks

- The working tree already contains many experiment/config/report changes from previous development sessions. They were not reverted.
- Local sensitive files still exist in the workspace:
  - `.env`
  - `custodian_ed25519`
  They are ignored by `.gitignore`, but should be removed or kept outside the repository before public release.
- Large artifacts are intentionally ignored:
  - `data/`
  - `models/`
  - `hf_cache/`
  - `cache/`
  - visual/text indexes;
  - runtime `results/`.
  A public README must tell users how to rebuild or download these artifacts.
- Some legacy configs and older reporting scripts were kept because they are small and may still be useful for provenance. They can be archived in a later pass after the final paper table is frozen.

## Later Improvements

- Add a small `docs/artifacts.md` file describing which indexes must be built for each experiment family.
- Add CI checks for:
  - `py_compile` on scripts and `src/`;
  - report table generation on a minimal mocked results folder.
- Add a minimal fixture dataset for smoke tests that does not require DocBench or GPU models.
- Move secrets completely outside the repository folder before publication.
