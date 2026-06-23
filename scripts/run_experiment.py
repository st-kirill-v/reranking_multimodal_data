from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--types", nargs="*", default=None)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    return bool(value) and str(value).lower() not in {"0", "false", "no", "none"}


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def append_fusion_args(command: list[str], fusion_cfg: dict[str, Any]) -> None:
    command.extend(["--fusion-mode", str(fusion_cfg.get("mode", "none"))])
    if not fusion_cfg:
        return
    command.extend(
        [
            "--fusion-alpha",
            str(fusion_cfg.get("alpha", 1.0)),
            "--fusion-beta",
            str(fusion_cfg.get("beta", 0.2)),
            "--fusion-gamma",
            str(fusion_cfg.get("gamma", 1.0)),
            "--fusion-lambda-number",
            str(fusion_cfg.get("lambda_number", 0.05)),
            "--fusion-lambda-keyword",
            str(fusion_cfg.get("lambda_keyword", 0.05)),
            "--fusion-lambda-exact-phrase",
            str(fusion_cfg.get("lambda_exact_phrase", 0.05)),
            "--fusion-lambda-table-header",
            str(fusion_cfg.get("lambda_table_header", 0.20)),
            "--fusion-text-reranker-model-id",
            str(fusion_cfg.get("text_reranker_model_id", "BAAI/bge-reranker-large")),
            "--fusion-text-reranker-device",
            str(fusion_cfg.get("text_reranker_device", "cuda")),
            "--fusion-text-reranker-batch-size",
            str(fusion_cfg.get("text_reranker_batch_size", 4)),
            "--fusion-text-reranker-max-length",
            str(fusion_cfg.get("text_reranker_max_length", 512)),
            "--fusion-text-reranker-backend",
            str(fusion_cfg.get("text_reranker_backend", "cross_encoder")),
        ]
    )
    if fusion_cfg.get("source_fields"):
        command.extend(
            [
                "--fusion-source-fields",
                *_as_list(
                    fusion_cfg.get("source_fields"),
                    ["page_text", "caption", "table_text"],
                ),
            ]
        )
    if not _as_bool(fusion_cfg.get("trust_remote_code", True)):
        command.append("--fusion-no-trust-remote-code")


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
        openai_vlm_cfg = generation_cfg.get("openai_vlm", config.get("openai_vlm", {}))
        command = [
            sys.executable,
            "-u",
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
            "--retriever-backend",
            str(retrieval_cfg.get("backend", "colvision")),
            "--retriever-model-id",
            str(retrieval_cfg.get("model_id", "vidore/colpali-v1.3-merged")),
            "--retrieval-device",
            str(retrieval_cfg.get("device", "cuda")),
            "--score-batch-size",
            str(retrieval_cfg.get("score_batch_size", 1)),
            "--first-stage-top-k",
            str(retrieval_cfg.get("first_stage_top_k", 30)),
            "--rerank-top-k",
            str(rerank_cfg.get("top_k", 10)),
            "--rerank-device",
            str(rerank_cfg.get("device", "cuda")),
            "--rerank-batch-size",
            str(rerank_cfg.get("batch_size", 1)),
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
            "--multimodal-generation-backend",
            str(generation_cfg.get("multimodal_generation_backend", "local_vlm")),
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
        if openai_vlm_cfg:
            command.extend(
                [
                    "--openai-vlm-base-url",
                    str(openai_vlm_cfg.get("base_url", "")),
                    "--openai-vlm-model",
                    str(openai_vlm_cfg.get("model", "openai/qwen3-vl-30b")),
                    "--openai-vlm-api-key-env",
                    str(openai_vlm_cfg.get("api_key_env", "OPENAI_COMPAT_API_KEY")),
                    "--openai-vlm-temperature",
                    str(openai_vlm_cfg.get("temperature", 0)),
                    "--openai-vlm-max-tokens",
                    str(
                        openai_vlm_cfg.get("max_tokens", generation_cfg.get("max_new_tokens", 192))
                    ),
                ]
            )
            if openai_vlm_cfg.get("api_key"):
                command.extend(["--openai-vlm-api-key", str(openai_vlm_cfg["api_key"])])
            if openai_vlm_cfg.get("timeout"):
                command.extend(["--openai-vlm-timeout", str(openai_vlm_cfg["timeout"])])
        if _as_bool(generation_cfg.get("no_4bit", False)):
            command.append("--no-4bit")
        if _as_bool(generation_cfg.get("do_sample", False)):
            command.append("--do-sample")
        if _as_bool(text_cfg.get("enable_text_tools", False)):
            command.append("--enable-text-tools")
        if eval_cfg.get("limit"):
            command.extend(["--limit", str(eval_cfg["limit"])])
        return command

    if evaluator == "full_pipeline":
        reranker_mode = str(rerank_cfg.get("mode", "nemotron_vl_cross_encoder"))
        if reranker_mode in {"none", "no_reranking"}:
            reranker_cli_mode = "none"
        elif reranker_mode in {"nemotron_text_image", "text_image"}:
            reranker_cli_mode = "nemotron_text_image"
        elif reranker_mode in {"adaptive", "adaptive_reranking"}:
            reranker_cli_mode = "adaptive"
        elif reranker_mode in {"threshold_skip", "threshold_skip_reranking"}:
            reranker_cli_mode = "threshold_skip"
        else:
            reranker_cli_mode = "nemotron"
        openai_vlm_cfg = generation_cfg.get("openai_vlm", config.get("openai_vlm", {}))
        command = [
            sys.executable,
            "-u",
            "scripts/evaluate_full_pipeline_layout_aware_clean.py",
            "--data-dir",
            str(dataset_cfg.get("data_dir", "data/datasets/docbench")),
            "--types",
            *_as_list(dataset_cfg.get("types"), ["multimodal-t", "multimodal-f"]),
            "--index-dir",
            str(retrieval_cfg.get("index_dir", "index_colpali_v1_3_merged")),
            "--index-name",
            str(retrieval_cfg.get("index_name", "pages_colpali_v1_3_merged_clean")),
            "--retriever-backend",
            str(retrieval_cfg.get("backend", "colvision")),
            "--retriever-model-id",
            str(retrieval_cfg.get("model_id", "vidore/colpali-v1.3-merged")),
            "--retrieval-device",
            str(retrieval_cfg.get("device", "cuda")),
            "--score-batch-size",
            str(retrieval_cfg.get("score_batch_size", 1)),
            "--first-stage-top-k",
            str(retrieval_cfg.get("first_stage_top_k", 30)),
            "--reranker-mode",
            reranker_cli_mode,
            "--rerank-top-k",
            str(rerank_cfg.get("top_k", 10)),
            "--rerank-device",
            str(rerank_cfg.get("device", "cuda")),
            "--rerank-batch-size",
            str(rerank_cfg.get("batch_size", 1)),
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
            "--device",
            str(generation_cfg.get("device", "cuda")),
            "--max-new-tokens",
            str(generation_cfg.get("max_new_tokens", 192)),
            "--max-context-images",
            str(generation_cfg.get("max_context_images", 5)),
            "--max-image-long-edge",
            str(generation_cfg.get("max_image_long_edge", 1600)),
            "--answer-refine",
            str(generation_cfg.get("answer_refine", "none")),
            "--multimodal-generation-backend",
            str(generation_cfg.get("multimodal_generation_backend", "local_vlm")),
            "--output",
            str(raw_output),
        ]
        if reranker_cli_mode == "nemotron_text_image":
            command.extend(
                [
                    "--rerank-text-max-chars",
                    str(rerank_cfg.get("text_max_chars", 4096)),
                ]
            )
            if rerank_cfg.get("text_source_fields"):
                command.extend(
                    [
                        "--rerank-text-source-fields",
                        *_as_list(
                            rerank_cfg.get("text_source_fields"),
                            ["table_text", "caption", "page_text", "ocr"],
                        ),
                    ]
                )
        elif reranker_cli_mode == "adaptive":
            adaptive_cfg = config.get("adaptive_reranking", {})
            command.extend(
                [
                    "--rerank-text-max-chars",
                    str(rerank_cfg.get("text_max_chars", 4096)),
                    "--adaptive-high-confidence-strategy",
                    str(adaptive_cfg.get("high_confidence_strategy", "no_reranker")),
                ]
            )
            if rerank_cfg.get("text_source_fields"):
                command.extend(
                    [
                        "--rerank-text-source-fields",
                        *_as_list(
                            rerank_cfg.get("text_source_fields"),
                            ["table_text", "caption", "page_text", "ocr"],
                        ),
                    ]
                )
            if adaptive_cfg.get("threshold_top1") not in (None, {}, ""):
                command.extend(
                    ["--adaptive-threshold-top1", str(adaptive_cfg.get("threshold_top1"))]
                )
            if adaptive_cfg.get("threshold_gap") not in (None, {}, ""):
                command.extend(["--adaptive-threshold-gap", str(adaptive_cfg.get("threshold_gap"))])
        elif reranker_cli_mode == "threshold_skip":
            threshold_cfg = config.get("threshold_skip", {})
            command.extend(
                [
                    "--rerank-text-max-chars",
                    str(rerank_cfg.get("text_max_chars", 4096)),
                    "--threshold-skip-top1",
                    str(threshold_cfg.get("threshold_top1", 0.8)),
                    "--threshold-skip-gap",
                    str(threshold_cfg.get("threshold_gap", 0.1)),
                ]
            )
            if rerank_cfg.get("text_source_fields"):
                command.extend(
                    [
                        "--rerank-text-source-fields",
                        *_as_list(
                            rerank_cfg.get("text_source_fields"),
                            ["table_text", "caption", "page_text", "ocr"],
                        ),
                    ]
                )
        append_fusion_args(command, config.get("fusion", {}))
        vlm_text_context_cfg = generation_cfg.get("vlm_text_context", {})
        if vlm_text_context_cfg:
            command.extend(
                [
                    "--vlm-text-context-mode",
                    str(vlm_text_context_cfg.get("mode", "none")),
                    "--vlm-text-max-chars",
                    str(vlm_text_context_cfg.get("max_chars", 12000)),
                ]
            )
            if vlm_text_context_cfg.get("source_fields"):
                command.extend(
                    [
                        "--vlm-text-source-fields",
                        *_as_list(
                            vlm_text_context_cfg.get("source_fields"),
                            ["ocr", "page_text", "caption", "table_text"],
                        ),
                    ]
                )
        if openai_vlm_cfg:
            command.extend(
                [
                    "--openai-vlm-base-url",
                    str(openai_vlm_cfg.get("base_url", "")),
                    "--openai-vlm-model",
                    str(openai_vlm_cfg.get("model", "openai/qwen3-vl-30b")),
                    "--openai-vlm-api-key-env",
                    str(openai_vlm_cfg.get("api_key_env", "OPENAI_COMPAT_API_KEY")),
                    "--openai-vlm-temperature",
                    str(openai_vlm_cfg.get("temperature", 0)),
                    "--openai-vlm-max-tokens",
                    str(
                        openai_vlm_cfg.get("max_tokens", generation_cfg.get("max_new_tokens", 192))
                    ),
                ]
            )
            if openai_vlm_cfg.get("api_key"):
                command.extend(["--openai-vlm-api-key", str(openai_vlm_cfg["api_key"])])
            if openai_vlm_cfg.get("timeout"):
                command.extend(["--openai-vlm-timeout", str(openai_vlm_cfg["timeout"])])
        if _as_bool(generation_cfg.get("no_4bit", False)):
            command.append("--no-4bit")
        if _as_bool(generation_cfg.get("do_sample", False)):
            command.append("--do-sample")
        if eval_cfg.get("limit"):
            command.extend(["--limit", str(eval_cfg["limit"])])
        return command

    if evaluator == "text_reranker_308":
        text_cfg = config.get("text_pipeline", {})
        openai_vlm_cfg = generation_cfg.get("openai_vlm", config.get("openai_vlm", {}))
        text_retriever_backend = str(retrieval_cfg.get("backend", "bm25"))
        if text_retriever_backend == "text_page_bm25":
            text_retriever_backend = "bm25"
        command = [
            sys.executable,
            "-u",
            "scripts/evaluate_text_reranker_308.py",
            "--data-dir",
            str(dataset_cfg.get("data_dir", "data/datasets/docbench")),
            "--types",
            *_as_list(dataset_cfg.get("types"), ["multimodal-t", "multimodal-f"]),
            "--text-retriever-backend",
            text_retriever_backend,
            "--first-stage-top-k",
            str(retrieval_cfg.get("first_stage_top_k", 30)),
            "--rerank-top-k",
            str(rerank_cfg.get("top_k", 10)),
            "--context-top-pages",
            str(text_cfg.get("context_top_pages", eval_cfg.get("text_top_pages", 3))),
            "--text-context-max-chars",
            str(text_cfg.get("text_context_max_chars", 12000)),
            "--text-reranker-model-id",
            str(rerank_cfg.get("model_id", "BAAI/bge-reranker-base")),
            "--text-reranker-device",
            str(rerank_cfg.get("device", "cuda")),
            "--text-reranker-batch-size",
            str(rerank_cfg.get("batch_size", 8)),
            "--text-reranker-max-length",
            str(rerank_cfg.get("max_length", 4096)),
            "--text-reranker-backend",
            str(rerank_cfg.get("backend", "cross_encoder")),
            "--openai-vlm-base-url",
            str(openai_vlm_cfg.get("base_url", "")),
            "--openai-vlm-model",
            str(openai_vlm_cfg.get("model", "openai/qwen3-vl-30b")),
            "--openai-vlm-api-key-env",
            str(openai_vlm_cfg.get("api_key_env", "OPENAI_COMPAT_API_KEY")),
            "--openai-vlm-temperature",
            str(openai_vlm_cfg.get("temperature", 0)),
            "--openai-vlm-max-tokens",
            str(openai_vlm_cfg.get("max_tokens", generation_cfg.get("max_new_tokens", 192))),
            "--openai-vlm-timeout",
            str(openai_vlm_cfg.get("timeout", 180)),
            "--output",
            str(raw_output),
        ]
        if text_retriever_backend == "text_encoder":
            command.extend(
                [
                    "--text-encoder-index-dir",
                    str(
                        retrieval_cfg.get(
                            "index_dir", "data/indexes/docbench_text_encoder_bge_base_en_v1_5"
                        )
                    ),
                    "--text-encoder-model-id",
                    str(retrieval_cfg.get("model_id", "BAAI/bge-base-en-v1.5")),
                    "--text-encoder-device",
                    str(retrieval_cfg.get("device", "cuda")),
                    "--text-encoder-batch-size",
                    str(retrieval_cfg.get("batch_size", 32)),
                    "--text-encoder-max-length",
                    str(retrieval_cfg.get("max_length", 512)),
                ]
            )
            if not _as_bool(retrieval_cfg.get("normalize", True)):
                command.append("--no-text-encoder-normalize")
        if text_cfg.get("source_fields"):
            command.extend(
                ["--text-source-fields", *_as_list(text_cfg.get("source_fields"), ["page_text"])]
            )
        if openai_vlm_cfg.get("api_key"):
            command.extend(["--openai-vlm-api-key", str(openai_vlm_cfg["api_key"])])
        if not _as_bool(rerank_cfg.get("trust_remote_code", True)):
            command.append("--no-trust-remote-code")
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
        "-u",
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
        "route_subtype": row.get("route_subtype"),
        "tool_used": row.get("tool_used"),
        "normalized_answer": row.get("normalized_answer"),
        "raw_generated_answer": row.get("raw_generated_answer"),
        "postprocessed_answer": row.get("postprocessed_answer") or row.get("generated"),
        "generation_backend": row.get("generation_backend")
        or generation_cfg.get("multimodal_generation_backend")
        or "local_vlm",
        "model_name": row.get("model_name") or generation_cfg.get("model_id"),
        "image_paths_sent": row.get("image_paths_sent") or [],
        "num_images_sent": row.get("num_images_sent"),
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
        "retriever_backend": row.get("retriever_backend") or retrieval_cfg.get("backend"),
        "colpali_used": (
            row.get("colpali_used")
            if row.get("colpali_used") is not None
            else retrieval_cfg.get("backend", "colvision") == "colvision"
        ),
        "top30_retrieved_pages": row.get("top30_retrieved_pages") or retrieved_pages[:30],
        "retrieval_scores": row.get("retrieval_scores") or [],
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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def write_adaptive_reranking_report(
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    results_dir: Path,
    config: dict[str, Any],
) -> None:
    report_dir = Path("reports/adaptive_reranking")
    report_dir.mkdir(parents=True, exist_ok=True)
    routes: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        routes.setdefault(str(row.get("adaptive_route") or "unknown"), []).append(row)
    skipped = [row for row in predictions if row.get("adaptive_skipped_reranker") is True]
    baselines = [
        ("Best quality baseline", 0.7023, 13.6441),
        ("Fast baseline", 0.6575, 2.5080),
        ("No-reranker full image+text baseline", 0.6784, 3.4263),
    ]
    mean_f1 = metrics.get("mean_f1")
    latency = metrics.get("latency_seconds", {}).get("mean")
    comparison_rows = []
    for name, baseline_f1, baseline_latency in baselines:
        delta_f1 = float(mean_f1) - baseline_f1 if mean_f1 is not None else None
        delta_latency = float(latency) - baseline_latency if latency is not None else None
        comparison_rows.append(
            f"| {name} | {baseline_f1:.4f} | {baseline_latency:.4f} | "
            f"{_fmt(delta_f1)} | {_fmt(delta_latency)} |"
        )

    route_rows = []
    for route, route_items in sorted(routes.items()):
        route_rows.append(
            f"| {route} | {len(route_items)} | {_fmt(_mean_optional(route_items, 'f1'))} | "
            f"{_fmt(_mean_optional(route_items, 'latency'))} | "
            f"{_fmt(_mean_optional(route_items, 'rerank_latency'))} |"
        )

    text = [
        "# Adaptive Reranking Summary",
        "",
        "## 1. Что было реализовано",
        "",
        "Добавлен отдельный экспериментальный режим `adaptive`, который выбирает стратегию реранкинга для каждого вопроса. Существующие конфиги, результаты и реализации реранкеров не изменяются.",
        "",
        "## 2. Routing strategies",
        "",
        "- `high_confidence`: retrieval считается достаточно уверенным, дорогой VL reranker пропускается.",
        "- `table_or_text`: используется full image+text Nemotron VL reranker.",
        "- `visual`: используется image-only Nemotron VL reranker.",
        "- `unknown`: используется default best strategy, full image+text Nemotron VL reranker.",
        "",
        "## 3. Route distribution",
        "",
        "| Route | Questions | Mean F1 | Mean latency | Mean rerank latency |",
        "|---|---:|---:|---:|---:|",
        *route_rows,
        "",
        "## 4. Skipped reranker",
        "",
        f"- skipped questions: {len(skipped)}",
        f"- skipped share: {_fmt(len(skipped) / max(len(predictions), 1))}",
        "",
        "## 5. Итоговые метрики",
        "",
        f"- total questions: {metrics.get('total_questions')}",
        f"- Mean F1: {_fmt(metrics.get('mean_f1'))}",
        f"- Exact Match: {_fmt(metrics.get('exact_match'))}",
        f"- F1 > 0.5: {_fmt(metrics.get('accuracy_f1_gt_0_5'))}",
        f"- MM-T F1: {_fmt(metrics.get('by_modality', {}).get('multimodal-t', {}).get('mean_f1'))}",
        f"- MM-F F1: {_fmt(metrics.get('by_modality', {}).get('multimodal-f', {}).get('mean_f1'))}",
        f"- latency: {_fmt(metrics.get('latency_seconds', {}).get('mean'))} sec",
        "",
        "## 6. Сравнение с baseline",
        "",
        "| Baseline | Baseline Mean F1 | Baseline latency | Adaptive delta F1 | Adaptive delta latency |",
        "|---|---:|---:|---:|---:|",
        *comparison_rows,
        "",
        "## 7. Вывод",
        "",
        "Заполнено автоматически после запуска эксперимента. Интерпретация: если `delta F1` около нуля или положительный, а `delta latency` отрицательный относительно best quality baseline, adaptive routing полезен для защиты как quality/latency improvement. Если качество заметно ниже, результат лучше использовать как дополнительную абляцию, а не как замену лучшей конфигурации статьи.",
        "",
        "## Reproducibility",
        "",
        f"- config: `configs/experiments/adaptive_reranking_308_qwen3vl30b.yaml`",
        f"- results: `{results_dir}`",
        f"- reranking mode: `{config.get('reranking', {}).get('mode')}`",
        f"- thresholds: `{config.get('adaptive_reranking', {})}`",
        "",
    ]
    (report_dir / "adaptive_reranking_summary.md").write_text("\n".join(text), encoding="utf-8")


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
    if args.limit is not None:
        config.setdefault("evaluation", {})["limit"] = args.limit
    if args.types:
        config.setdefault("dataset", {})["types"] = _as_list(args.types, [])
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
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
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
    if str(config.get("reranking", {}).get("mode", "")).lower() in {
        "adaptive",
        "adaptive_reranking",
    }:
        write_adaptive_reranking_report(predictions, metrics, results_dir, config)
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
