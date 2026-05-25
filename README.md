# Reranking Multimodal Data

Clean research pipeline for multimodal document question answering on DocBench-style PDF pages.

The current system evaluates a multimodal RAG stack:

```text
DocBench pages -> ColPali/ColVision retrieval -> Nemotron reranking -> page/crop selection
-> Qwen3-VL answer generation -> metrics/reporting
```

The repository now keeps publication-facing entrypoints in `scripts/`, reusable code in `src/`,
experiment configs in `configs/experiments/`, and old exploratory code in `archive/`.

## Project Layout

```text
configs/experiments/        reproducible experiment configs
src/data/                   dataset abstraction
src/mmrag/                  shared DocBench schema, dataset loading, and reranker config
src/reranking/              unified reranker interfaces
src/cropping/               layout-aware crop helpers
src/core/generators/        Qwen3-VL generation code
src/evaluation/             metrics and success criteria checks
src/reporting/              tables, plot data, error analysis
scripts/run_experiment.py   one-command experiment runner
scripts/evaluate_full_pipeline_layout_aware_clean.py
                            full ColPali -> Nemotron -> layout-aware -> Qwen3-VL evaluator
scripts/evaluate.py         recompute metrics from predictions.jsonl
scripts/generate_report.py  aggregate report generation
data/                       datasets, intermediate candidates, debug crops
results/                    experiment outputs
reports/                    paper-facing tables/figures/error analysis
archive/                    old diagnostics and exploratory scripts
```

## Fresh Clone Setup

Install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

On Windows PowerShell:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync
```

The experiments require CUDA for the reported latency and quality. The current production-like
pipeline uses `vidore/colpali-v1.3-merged` for ColPali/ColVision retrieval,
`nvidia/llama-nemotron-rerank-vl-1b-v2` for multimodal reranking, and
`Qwen/Qwen3-VL-8B-Instruct` for answer generation.

Recommended reproducibility hardware:

- NVIDIA CUDA GPU;
- 24 GB VRAM minimum for Qwen3-VL-8B fp16 on 3-5 page images;
- 40 GB+ VRAM recommended for stable full runs;
- 32-64 GB system RAM;
- enough disk for DocBench page PNGs, model cache, FAISS indexes, and results.

## Data Layout

DocBench pages should be available as PNG files:

```text
data/datasets/docbench/<document_id>/extracted/pages/page_<N>.png
data/datasets/docbench/<document_id>/<document_id>_qa.jsonl
```

Each document may also contain:

```text
data/datasets/docbench/<document_id>/extracted/doc_report.json
```

## Build ColPali/ColVision Page Index

Build the current ColPali/ColVision page index:

```bash
python scripts/build_colvision_index_clean.py \
  --data-dir data/datasets/docbench \
  --index-dir index_colpali_v1_3_merged \
  --index-name pages_colpali_v1_3_merged_clean \
  --model-id vidore/colpali-v1.3-merged \
  --device cuda
```

Expected artifacts:

```text
index_colpali_v1_3_merged/metadata_pages_colpali_v1_3_merged_clean.json
index_colpali_v1_3_merged/manifest_pages_colpali_v1_3_merged_clean.json
index_colpali_v1_3_merged/shards/*.pt
```

## Full Production-Like Evaluation

This is the main honest evaluation path. It starts from the question, retrieves pages with
ColPali/ColVision, reranks them with Nemotron, applies layout-aware page/crop selection, generates
the answer with Qwen3-VL, and then computes metrics.

```bash
python scripts/evaluate_full_pipeline_layout_aware_clean.py \
  --first-stage-top-k 30 \
  --rerank-top-k 10 \
  --adaptive-policy text_top3_visual_top5 \
  --text-top-pages 3 \
  --visual-top-pages 5 \
  --visual-crop-policy layout_aware_v2 \
  --layout-context-mode full_page_plus_crop \
  --prompt-style concise \
  --output data/eval_full_pipeline_colpali_nemotron_layout_aware_v2_308.json
```

## Reproduce Current Candidate-Based Results

The current prompt/crop ablations use cached reranked page candidates:

```text
data/eval_vlm_reranked_adaptive_clean_rerun_full_308.json
```

Run the current full ColPali pipeline as a single command:

```bash
python scripts/run_experiment.py --config configs/experiments/full_colpali_layout_aware.yaml
```

Outputs:

```text
results/full_colpali_layout_aware/
  predictions.jsonl
  predictions_raw.json
  metrics.json
  metrics_table.csv
  metrics_table.md
  config.yaml
  run.log
  error_cases.csv
  summary.md
```

Older cached-candidate prompt/crop ablations were moved to
`archive/legacy_ablation_scripts/`. They are preserved for auditability, but they are no longer
the main publication pipeline because they start from precomputed candidates rather than full
retrieval.

## Metrics

Recompute metrics from any experiment:

```bash
python scripts/evaluate.py --predictions results/baseline/predictions.jsonl
```

Implemented metrics:

- exact match;
- token F1;
- F1 > 0.5 accuracy;
- relaxed exact;
- containment metrics;
- numeric any/all/precision/recall;
- latency mean/p50/p95;
- crop used rate;
- crop type mismatch rate;
- caption match rate.

## Success Criteria

A question is successfully processed if:

- no runtime error occurs;
- retrieved pages exist;
- the VLM returns a non-empty answer;
- latency is recorded;
- prediction is saved;
- metrics are computed;
- missing answers are represented exactly as `NOT FOUND`.

See `docs/success_criteria.md`.

## Reporting

Generate aggregate tables and plot data:

```bash
python scripts/generate_report.py --results-dir results --reports-dir reports
python scripts/generate_metrics_tables.py --results-dir results --output reports/tables/metrics_by_experiment.md
python scripts/generate_plots.py --results-dir results --output-dir reports/figures
```

Generate error analysis:

```bash
python scripts/generate_error_analysis.py \
  --predictions results/baseline/predictions.jsonl \
  --output reports/error_analysis/baseline_errors.csv
```

## Archive Policy

Old diagnostics, duplicated experiments, and exploratory scripts are preserved under `archive/`.
They are not part of the publication pipeline, but remain available for auditability.

Qwen embedding retrieval experiments were moved to:

```text
archive/qwen_retrieval_experiments/
```

That archive contains the old Qwen3-VL embedding index builders, Qwen retrieval evaluators, and the
old page-RAG pipeline that depended on `Qwen/Qwen3-VL-Embedding-2B`. They are kept only for audit
history; the current pipeline uses ColPali/ColVision retrieval.

Legacy staged ablation scripts were moved to:

```text
archive/legacy_ablation_scripts/
```

That archive contains cached-candidate VLM evaluators, prompt/crop comparison scripts, ColVision
diagnostics, candidate export, and standalone Nemotron rerank diagnostics. The code needed by the
current pipeline was extracted into `src/retrieval/colvision.py`, `src/evaluation/vlm_eval.py`, and
`src/cropping/layout_aware_eval.py`.
