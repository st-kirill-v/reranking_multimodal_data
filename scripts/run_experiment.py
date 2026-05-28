from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import enrich_rows, write_metrics_artifacts
from src.reporting.error_analysis import write_error_cases
from src.utils.config import dump_config, load_config


SUCCESS_CRITERIA = [
    "question processed without runtime error",
    "retrieved pages exist",
    "VLM returned a non-empty answer",
    "latency recorded",
    "prediction saved",
    "metrics computed",
    'missing answer is represented exactly as "NOT FOUND"',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible experiment from config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    return bool(value) and str(value).lower() not in {"0", "false", "no", "none"}


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_eval_command(config: dict[str, Any], raw_output: Path) -> list[str]:
    eval_cfg = config.get("evaluation", {})
    crop_cfg = config.get("cropping", {})
    generation_cfg = config.get("generation", {})
    dataset_cfg = config.get("dataset", {})
    retrieval_cfg = config.get("retrieval", {})
    rerank_cfg = config.get("reranking", {})

    evaluator = eval_cfg.get("evaluator", "layout_aware")
    if evaluator == "hybrid_bm25":
        text_cfg = config.get("text_pipeline", {})
        command = [
            sys.executable,
            "scripts/evaluate_docbench_hybrid_bm25.py",
            "--data-dir",
            str(dataset_cfg.get("data_dir", "data/datasets/docbench")),
            "--types",
            *_as_list(dataset_cfg.get("types"), ["all"]),
            "--routing-mode",
            str(config.get("routing_mode", "by_official_type")),
            "--text-index-dir",
            str(text_cfg.get("text_index_dir", "data/indexes/docbench_bm25")),
            "--text-top-k",
            str(text_cfg.get("text_top_k", 5)),
            "--text-context-max-chars",
            str(text_cfg.get("text_context_max_chars", 12000)),
            "--index-dir",
            str(retrieval_cfg.get("index_dir", "index_colpali_v1_3_merged")),
            "--index-name",
            str(retrieval_cfg.get("index_name", "pages_colpali_v1_3_merged_clean")),
            "--retriever-model-id",
            str(retrieval_cfg.get("model_id", "vidore/colpali-v1.3-merged")),
            "--first-stage-top-k",
            str(retrieval_cfg.get("first_stage_top_k", 30)),
            "--rerank-top-k",
            str(rerank_cfg.get("top_k", 10)),
            "--adaptive-policy",
            str(eval_cfg.get("adaptive_policy", "text_top3_visual_top5")),
            "--text-top-pages",
            str(eval_cfg.get("text_top_pages", 3)),
            "--visual-top-pages",
            str(eval_cfg.get("visual_top_pages", 5)),
            "--visual-crop-policy",
            str(crop_cfg.get("visual_crop_policy", "layout_aware_v2")),
            "--layout-context-mode",
            str(crop_cfg.get("layout_context_mode", "full_page_plus_crop")),
            "--debug-crop-dir",
            str(crop_cfg.get("debug_crop_dir", "data/debug_crops/docbench_hybrid_bm25")),
            "--prompt-style",
            str(generation_cfg.get("prompt_style", "concise")),
            "--prompt-profile",
            str(generation_cfg.get("prompt_profile", config.get("prompt_profile", "legacy"))),
            "--max-new-tokens",
            str(generation_cfg.get("max_new_tokens", 192)),
            "--max-context-images",
            str(generation_cfg.get("max_context_images", 5)),
            "--max-image-long-edge",
            str(generation_cfg.get("max_image_long_edge", 1600)),
            "--answer-refine",
            str(generation_cfg.get("answer_refine", "none")),
            "--output",
            str(raw_output),
        ]
        if _as_bool(generation_cfg.get("no_4bit", False)):
            command.append("--no-4bit")
        if _as_bool(generation_cfg.get("do_sample", False)):
            command.append("--do-sample")
        if eval_cfg.get("limit"):
            command.extend(["--limit", str(eval_cfg["limit"])])
        return command

    if evaluator == "full_pipeline":
        command = [
            sys.executable,
            "scripts/evaluate_full_pipeline_layout_aware_clean.py",
            "--data-dir",
            str(dataset_cfg.get("data_dir", "data/datasets/docbench")),
            "--index-dir",
            str(retrieval_cfg.get("index_dir", "index_colpali_v1_3_merged")),
            "--index-name",
            str(retrieval_cfg.get("index_name", "pages_colpali_v1_3_merged_clean")),
            "--retriever-model-id",
            str(retrieval_cfg.get("model_id", "vidore/colpali-v1.3-merged")),
            "--first-stage-top-k",
            str(retrieval_cfg.get("first_stage_top_k", 30)),
            "--rerank-top-k",
            str(rerank_cfg.get("top_k", 10)),
            "--adaptive-policy",
            str(eval_cfg.get("adaptive_policy", "text_top3_visual_top5")),
            "--text-top-pages",
            str(eval_cfg.get("text_top_pages", 3)),
            "--visual-top-pages",
            str(eval_cfg.get("visual_top_pages", 5)),
            "--visual-crop-policy",
            str(crop_cfg.get("visual_crop_policy", "layout_aware_v2")),
            "--layout-context-mode",
            str(crop_cfg.get("layout_context_mode", "full_page_plus_crop")),
            "--debug-crop-dir",
            str(crop_cfg.get("debug_crop_dir", "data/debug_crops/full_pipeline_layout_aware_v2")),
            "--prompt-style",
            str(generation_cfg.get("prompt_style", "concise")),
            "--max-new-tokens",
            str(generation_cfg.get("max_new_tokens", 192)),
            "--max-context-images",
            str(generation_cfg.get("max_context_images", 5)),
            "--max-image-long-edge",
            str(generation_cfg.get("max_image_long_edge", 1600)),
            "--answer-refine",
            str(generation_cfg.get("answer_refine", "none")),
            "--output",
            str(raw_output),
        ]
        if _as_bool(generation_cfg.get("no_4bit", False)):
            command.append("--no-4bit")
        if _as_bool(generation_cfg.get("do_sample", False)):
            command.append("--do-sample")
        if eval_cfg.get("limit"):
            command.extend(["--limit", str(eval_cfg["limit"])])
        return command

    script = (
        "archive/legacy_ablation_scripts/evaluate_vlm_layout_aware_from_page_candidates_clean.py"
        if evaluator == "layout_aware"
        else "archive/legacy_ablation_scripts/evaluate_vlm_from_page_candidates_clean.py"
    )
    command = [
        sys.executable,
        script,
        "--input",
        str(
            dataset_cfg.get(
                "input_candidates", "data/eval_vlm_reranked_adaptive_clean_rerun_full_308.json"
            )
        ),
        "--mode",
        str(eval_cfg.get("mode", "reranked")),
        "--adaptive-policy",
        str(eval_cfg.get("adaptive_policy", "text_top3_visual_top5")),
        "--text-top-pages",
        str(eval_cfg.get("text_top_pages", 3)),
        "--visual-top-pages",
        str(eval_cfg.get("visual_top_pages", 5)),
        "--visual-crop-policy",
        str(crop_cfg.get("visual_crop_policy", "visual_main")),
        "--prompt-style",
        str(generation_cfg.get("prompt_style", "concise")),
        "--output",
        str(raw_output),
    ]
    if evaluator == "layout_aware":
        command.extend(
            [
                "--layout-context-mode",
                str(crop_cfg.get("layout_context_mode", "crop_only")),
                "--debug-crop-dir",
                str(crop_cfg.get("debug_crop_dir", "data/debug_crops")),
            ]
        )
    if _as_bool(generation_cfg.get("no_4bit", False)):
        command.append("--no-4bit")
    if eval_cfg.get("limit"):
        command.extend(["--limit", str(eval_cfg["limit"])])
    return command


def page_labels(candidates: list[dict[str, Any]]) -> list[str]:
    labels = []
    for candidate in candidates or []:
        folder = candidate.get("folder")
        page = candidate.get("page")
        labels.append(
            f"{folder}/{page}" if folder is not None and page is not None else str(candidate)
        )
    return labels


def selected_crop(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("layout_aware_selected_crop") or {}


def normalize_prediction_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    crop_cfg = config.get("cropping", {})
    generation_cfg = config.get("generation", {})
    retrieval_cfg = config.get("retrieval", {})
    rerank_cfg = config.get("reranking", {})
    dataset_cfg = config.get("dataset", {})
    crop = selected_crop(row)
    retrieved_pages = page_labels(row.get("retrieved_candidates") or [])
    reranked_pages = page_labels(row.get("reranked_candidates") or [])
    selected_pages = page_labels(row.get("pages") or []) or row.get("selected_pages") or []
    return {
        **row,
        "expected_answer": row.get("expected"),
        "generated_answer": row.get("generated"),
        "exact_match": row.get("exact_match", row.get("exact")),
        "original_type": row.get("original_type") or row.get("type"),
        "pipeline_used": row.get("pipeline_used"),
        "prompt_profile": row.get("prompt_profile")
        or generation_cfg.get("prompt_profile")
        or config.get("prompt_profile"),
        "prompt_name": row.get("prompt_name"),
        "dataset": dataset_cfg.get("name", "docbench"),
        "dataset_split": dataset_cfg.get("split") or dataset_cfg.get("subset"),
        "document_id": row.get("expected_folder"),
        "retrieved_pages": retrieved_pages or row.get("retrieved_pages") or [],
        "retrieved_text_pages": row.get("retrieved_text_pages") or [],
        "reranked_pages": reranked_pages,
        "selected_pages": selected_pages,
        "crop_used": bool(row.get("crop_used") or row.get("crop_path")),
        "crop_path": row.get("crop_path") or crop.get("crop_path"),
        "fallback_used": row.get("fallback_used"),
        "prompt_style": generation_cfg.get("prompt_style", "concise"),
        "retrieval_mode": retrieval_cfg.get("mode", "colpali_colvision_multivector"),
        "reranker_mode": rerank_cfg.get("mode", "nemotron_vl_cross_encoder"),
        "crop_policy": crop_cfg.get("crop_policy") or crop_cfg.get("visual_crop_policy"),
        "context_mode": row.get("context_mode")
        or crop_cfg.get("context_mode")
        or crop_cfg.get("layout_context_mode"),
        "latency_total": row.get("latency"),
        "latency_retrieval": row.get("retrieval_latency"),
        "latency_rerank": row.get("rerank_latency"),
        "latency_context": row.get("context_latency"),
        "latency_vlm": row.get("vlm_latency"),
        "latency_generation": row.get("latency_generation") or row.get("vlm_latency"),
    }


def validate_prediction(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    row = normalize_prediction_row(row, config)
    errors = []
    generated = row.get("generated_answer")
    pages = row.get("selected_pages") or row.get("retrieved_text_pages") or []
    if row.get("runtime_error"):
        errors.append("runtime_error")
    if not pages:
        errors.append("no_retrieved_pages")
    if generated is None or str(generated).strip() == "":
        errors.append("empty_answer")
    if generated == "NOT FOUND":
        pass
    if row.get("latency") is None:
        errors.append("missing_latency")
    if row.get("f1") is None:
        errors.append("missing_metrics")
    item = dict(row)
    item["status"] = "success" if not errors else "failed"
    item["success"] = not errors
    item["success_errors"] = errors
    item["prediction"] = generated
    item["modality"] = row.get("type")
    return item


def convert_raw_output(
    raw_output: Path, results_dir: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = json.loads(raw_output.read_text(encoding="utf-8"))
    rows = payload.get("results") or payload.get("rows") or []
    predictions = [validate_prediction(row, config) for row in enrich_rows(rows)]
    pred_path = results_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return predictions


def write_summary(results_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = [
        f"# {config.get('experiment_name', results_dir.name)}",
        "",
        "## Success Criteria",
        *[f"- {criterion}" for criterion in SUCCESS_CRITERIA],
        "",
        "## Key Metrics",
        f"- total questions: {metrics.get('total_questions')}",
        f"- exact_match: {metrics.get('exact_match')}",
        f"- mean_f1: {metrics.get('mean_f1')}",
        f"- f1 > 0.5: {metrics.get('accuracy_f1_gt_0_5')}",
        f"- multimodal-t mean_f1: {metrics.get('by_modality', {}).get('multimodal-t', {}).get('mean_f1')}",
        f"- multimodal-f mean_f1: {metrics.get('by_modality', {}).get('multimodal-f', {}).get('mean_f1')}",
        f"- latency mean: {metrics.get('latency_seconds', {}).get('mean')}",
        "",
    ]
    (results_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def log_header(
    config_path: Path, results_dir: Path, config: dict[str, Any], command: list[str]
) -> str:
    retrieval = config.get("retrieval", {})
    reranking = config.get("reranking", {})
    cropping = config.get("cropping", {})
    generation = config.get("generation", {})
    lines = [
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        f"Config: {config_path}",
        f"Output directory: {results_dir}",
        "Command:",
        " ".join(command),
        "",
        f"VLM model: {generation.get('model_id')}",
        f"Prompt style: {generation.get('prompt_style')}",
        f"Retrieval: {retrieval.get('mode')} model={retrieval.get('model_id')} top_k={retrieval.get('first_stage_top_k')}",
        f"Reranker: {reranking.get('mode')} model={reranking.get('model_id')} top_k={reranking.get('top_k')}",
        f"Crop/context: crop={cropping.get('crop_policy') or cropping.get('visual_crop_policy')} context={cropping.get('context_mode') or cropping.get('layout_context_mode')}",
        "",
    ]
    return "\n".join(lines)


def append_footer(
    log_path: Path, predictions: list[dict[str, Any]], metrics: dict[str, Any]
) -> None:
    successful = sum(1 for row in predictions if row.get("status") == "success")
    failed = len(predictions) - successful
    runtime_errors = [row for row in predictions if row.get("success_errors")]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"Total questions: {len(predictions)}\n")
        handle.write(f"Successful questions: {successful}\n")
        handle.write(f"Failed questions: {failed}\n")
        handle.write(f"Runtime/validation errors: {len(runtime_errors)}\n")
        handle.write(f"mean_f1: {metrics.get('mean_f1')}\n")
        handle.write(f"accuracy_f1_gt_0_5: {metrics.get('accuracy_f1_gt_0_5')}\n")
        handle.write(f"exact_match: {metrics.get('exact_match')}\n")
        handle.write(f"doc_hit_at_1: {metrics.get('doc_hit_at_1')}\n")
        handle.write(f"page_hit_at_1: {metrics.get('page_hit_at_1')}\n")
        handle.write(f"latency_mean: {metrics.get('latency_seconds', {}).get('mean')}\n")
        if runtime_errors:
            handle.write("\nFailed cases:\n")
            for row in runtime_errors:
                handle.write(
                    json.dumps(
                        {
                            "question_id": row.get("question_id"),
                            "status": row.get("status"),
                            "errors": row.get("success_errors"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_name = str(config.get("experiment_name") or args.config.stem)
    output_cfg = config.get("output", {})
    results_dir = Path(
        output_cfg.get("directory") or Path(config.get("results_root", "results")) / experiment_name
    )
    raw_output = results_dir / "raw_output.json"

    command = build_eval_command(config, raw_output)
    if args.dry_run:
        print(" ".join(command))
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, results_dir / "config.yaml")
    log_path = results_dir / "run.log"
    log_path.write_text(log_header(args.config, results_dir, config, command), encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"Experiment failed with exit code {return_code}; see {log_path}")

    predictions = convert_raw_output(raw_output, results_dir, config)
    metrics = write_metrics_artifacts(predictions, results_dir)
    write_error_cases(predictions, results_dir / "error_cases.csv")
    write_summary(results_dir, config, metrics)
    shutil.copy2(raw_output, results_dir / "predictions_raw.json")
    append_footer(log_path, predictions, metrics)
    successful = sum(1 for row in predictions if row.get("status") == "success")
    failed = len(predictions) - successful
    print(
        json.dumps(
            {
                "total_questions": len(predictions),
                "successful_questions": successful,
                "failed_questions": failed,
                "mean_f1": metrics.get("mean_f1"),
                "accuracy_f1_gt_0_5": metrics.get("accuracy_f1_gt_0_5"),
                "exact_match": metrics.get("exact_match"),
                "doc_hit_at_1": metrics.get("doc_hit_at_1"),
                "page_hit_at_1": metrics.get("page_hit_at_1"),
                "latency_mean": metrics.get("latency_seconds", {}).get("mean"),
                "results_dir": str(results_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Experiment saved to {results_dir}")


if __name__ == "__main__":
    main()
