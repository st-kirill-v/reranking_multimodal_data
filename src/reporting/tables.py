from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.config import load_config


REPORT_COLUMNS = [
    "experiment",
    "dataset",
    "retriever",
    "reranker",
    "evidence",
    "vlm",
    "total",
    "successful",
    "failed",
    "exact_match",
    "mean_f1",
    "f1_gt_0_5",
    "multimodal_t_mean_f1",
    "multimodal_f_mean_f1",
    "doc_hit_at_1",
    "page_hit_at_1",
    "latency_mean",
    "latency_retrieval_mean",
    "latency_rerank_mean",
    "latency_context_mean",
    "latency_vlm_mean",
    "results_dir",
]

PAPER_EXCLUDE_NAME_PATTERNS = (
    "smoke",
    "bad",
    "current_full_pipeline",
)

PAPER_COLUMNS = [
    "method_group",
    "experiment",
    "retriever",
    "reranker",
    "evidence",
    "vlm",
    "mean_f1",
    "f1_gt_0_5",
    "exact_match",
    "multimodal_t_mean_f1",
    "multimodal_f_mean_f1",
    "doc_hit_at_1",
    "page_hit_at_1",
    "latency_mean",
    "results_dir",
]


def _get_nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_result_config(result_dir: Path, configs_dir: Path | None = None) -> dict[str, Any]:
    for name in ("config.yaml", "config.json"):
        path = result_dir / name
        if path.exists():
            try:
                return load_config(path)
            except Exception:
                return _load_json(path)
    if configs_dir is not None:
        path = configs_dir / f"{result_dir.name}.yaml"
        if path.exists():
            try:
                return load_config(path)
            except Exception:
                return {}
    return {}


def _by_type(metrics: dict[str, Any], question_type: str) -> dict[str, Any]:
    return (
        _get_nested(metrics, ["by_type", question_type], {})
        or _get_nested(metrics, ["by_modality", question_type], {})
        or _get_nested(metrics, ["by_original_type", question_type], {})
        or {}
    )


def _latency_mean(metrics: dict[str, Any], key: str | None = None) -> Any:
    if key is None:
        return _first(
            _get_nested(metrics, ["latency_seconds", "mean"]),
            _get_nested(metrics, ["latency", "mean"]),
            metrics.get("latency_mean"),
        )
    return _first(
        _get_nested(metrics, ["latency_breakdown", key, "mean"]),
        _get_nested(metrics, ["latency_seconds", key, "mean"]),
        _get_nested(metrics, [f"latency_{key}", "mean"]),
        metrics.get(f"latency_{key}_mean"),
    )


def _config_label(config: dict[str, Any], section: str, default: str = "") -> str:
    cfg = config.get(section, {})
    if not isinstance(cfg, dict):
        return default
    parts = [
        cfg.get("mode"),
        cfg.get("backend"),
        cfg.get("model_id"),
    ]
    label = " / ".join(str(item) for item in parts if item)
    return label or default


def _evidence_label(config: dict[str, Any]) -> str:
    text_cfg = config.get("text_pipeline", {})
    crop_cfg = config.get("crop", {}) or config.get("cropping", {})
    if isinstance(text_cfg, dict) and text_cfg.get("source_fields"):
        fields = text_cfg.get("source_fields")
        if isinstance(fields, list):
            return "+".join(str(item) for item in fields)
        return str(fields)
    if isinstance(crop_cfg, dict):
        crop_policy = crop_cfg.get("visual_crop_policy") or crop_cfg.get("crop_policy")
        context_mode = crop_cfg.get("layout_context_mode") or crop_cfg.get("context_mode")
        if crop_policy or context_mode:
            return " / ".join(str(item) for item in [crop_policy, context_mode] if item)
    return ""


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _vlm_label(config: dict[str, Any]) -> str:
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        return ""
    openai_vlm = generation.get("openai_vlm", {})
    if isinstance(openai_vlm, dict) and openai_vlm.get("model"):
        return str(openai_vlm["model"])
    return str(generation.get("model_id") or generation.get("model") or "")


def _method_group(row: dict[str, Any]) -> str:
    experiment = str(row.get("experiment") or "")
    retriever = str(row.get("retriever") or "").lower()
    reranker = str(row.get("reranker") or "").lower()
    evidence = str(row.get("evidence") or "").lower()
    vlm = str(row.get("vlm") or "").lower()
    if "image_text_fusion" in experiment:
        if "nemotron" in experiment:
            return "Image+text fusion, Nemotron visual"
        return "Image+text fusion, ColPali visual"
    if "text_evidence_encoder" in experiment:
        if "none" in reranker:
            return "Text evidence encoder, no reranker"
        return "Text evidence encoder + text reranker"
    if experiment.startswith("text_reranker_"):
        if "none" in reranker:
            return "BM25 text, no reranker"
        return "BM25 text + text reranker"
    if "nemotron_image_retriever" in experiment:
        if "nemotron_vl_cross_encoder" in reranker:
            return "Nemotron image retriever + VL reranker"
        return "Nemotron image retriever, no reranker"
    if "colpali" in retriever or "colvision" in retriever:
        if "qwen3-vl-30b" in vlm:
            if "none" in reranker:
                return "ColPali visual + Qwen30B, no reranker"
            return "ColPali visual + Qwen30B + VL reranker"
        if "none" in reranker:
            return "ColPali visual + Qwen8B, no reranker"
        return "ColPali visual + Qwen8B + VL reranker"
    if "layout_aware" in evidence:
        return "Visual layout-aware"
    return "Other"


def collect_metric_rows(
    results_dir: Path,
    *,
    configs_dir: Path | None = Path("configs/experiments"),
) -> list[dict[str, Any]]:
    rows = []
    if not results_dir.exists():
        return rows
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        result_dir = metrics_path.parent
        metrics = _load_json(metrics_path)
        config = _load_result_config(result_dir, configs_dir=configs_dir)
        dataset = config.get("dataset", {})
        if not isinstance(dataset, dict):
            dataset = {}
        multimodal_t = _by_type(metrics, "multimodal-t")
        multimodal_f = _by_type(metrics, "multimodal-f")
        row = {
            "experiment": result_dir.name,
            "dataset": dataset.get("split") or dataset.get("subset") or dataset.get("name") or "",
            "retriever": _config_label(config, "retrieval"),
            "reranker": _config_label(config, "reranking"),
            "evidence": _evidence_label(config),
            "vlm": _vlm_label(config),
            "total": _first(metrics.get("total"), metrics.get("total_questions")),
            "successful": metrics.get("successful_questions"),
            "failed": metrics.get("failed_questions"),
            "exact_match": _round(metrics.get("exact_match")),
            "mean_f1": _round(metrics.get("mean_f1")),
            "f1_gt_0_5": _round(metrics.get("accuracy_f1_gt_0_5")),
            "multimodal_t_mean_f1": _round(multimodal_t.get("mean_f1")),
            "multimodal_f_mean_f1": _round(multimodal_f.get("mean_f1")),
            "doc_hit_at_1": _round(metrics.get("doc_hit_at_1")),
            "page_hit_at_1": _round(metrics.get("page_hit_at_1")),
            "latency_mean": _round(_latency_mean(metrics)),
            "latency_retrieval_mean": _round(_latency_mean(metrics, "retrieval")),
            "latency_rerank_mean": _round(_latency_mean(metrics, "rerank")),
            "latency_context_mean": _round(_latency_mean(metrics, "context")),
            "latency_vlm_mean": _round(_latency_mean(metrics, "vlm")),
            "results_dir": result_dir.as_posix(),
        }
        row["method_group"] = _method_group(row)
        rows.append(row)
    return rows


def filter_paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        experiment = str(row.get("experiment") or "").lower()
        dataset = str(row.get("dataset") or "")
        total = row.get("total")
        if any(pattern in experiment for pattern in PAPER_EXCLUDE_NAME_PATTERNS):
            continue
        if dataset != "multimodal_308":
            continue
        if total not in {308, "308"}:
            continue
        filtered.append(row)
    return filtered


def _headers_for_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    return [header for header in columns if any(header in row for row in rows)]


def write_markdown_table(
    rows: list[dict[str, Any]], path: Path, *, columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = _headers_for_rows(rows, columns or REPORT_COLUMNS)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(
    rows: list[dict[str, Any]], path: Path, *, columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _headers_for_rows(rows, columns or REPORT_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})
