from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "answer",
    "question",
}


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text)


def normalize_number_token(token: str) -> str:
    token = token.lower().replace(",", "").strip()
    token = token.replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
    token = token.rstrip(".")
    if token.endswith("%"):
        token = token[:-1]
    try:
        value = float(token)
    except ValueError:
        return token
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: str) -> list[str]:
    return [
        normalize_number_token(match.group(0))
        for match in re.finditer(r"[$€£₹]?\d[\d,]*(?:\.\d+)?%?", text or "")
    ]


def token_f1(generated: str, expected: str) -> float:
    if generated in {"NOT FOUND", "ERROR", "TIMEOUT"}:
        return 0.0
    gen = normalize_answer(generated)
    exp = normalize_answer(expected)
    gen_numbers = re.findall(r"\d+(?:\.\d+)?%?", gen)
    exp_numbers = re.findall(r"\d+(?:\.\d+)?%?", exp)
    gen_words = {
        word
        for word in re.findall(r"[a-z][a-z0-9_\-]*", gen)
        if word not in STOP_WORDS and len(word) > 2
    }
    exp_words = {
        word
        for word in re.findall(r"[a-z][a-z0-9_\-]*", exp)
        if word not in STOP_WORDS and len(word) > 2
    }
    word_overlap = gen_words & exp_words
    word_precision = len(word_overlap) / len(gen_words) if gen_words else 0.0
    word_recall = len(word_overlap) / len(exp_words) if exp_words else 0.0
    word_score = (
        2 * word_precision * word_recall / (word_precision + word_recall)
        if word_precision + word_recall
        else 0.0
    )
    num_overlap = set(gen_numbers) & set(exp_numbers)
    num_precision = len(num_overlap) / len(gen_numbers) if gen_numbers else 0.0
    num_recall = len(num_overlap) / len(exp_numbers) if exp_numbers else 0.0
    num_score = (
        2 * num_precision * num_recall / (num_precision + num_recall)
        if num_precision + num_recall
        else 0.0
    )
    return 0.4 * word_score + 0.6 * num_score if exp_numbers else word_score


def row_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    generated = row.get("generated") or row.get("prediction") or ""
    expected = row.get("expected") or row.get("expected_answer") or ""
    gen_norm = normalize_answer(generated)
    exp_norm = normalize_answer(expected)
    gen_compact = re.sub(r"[^a-z0-9]+", "", gen_norm)
    exp_compact = re.sub(r"[^a-z0-9]+", "", exp_norm)
    gen_numbers = extract_numbers(generated)
    exp_numbers = extract_numbers(expected)
    expected_counts = Counter(exp_numbers)
    generated_counts = Counter(gen_numbers)
    matched = sum(min(count, generated_counts[number]) for number, count in expected_counts.items())
    return {
        "exact_match": float(
            gen_norm.replace(" ", "").replace("%", "") == exp_norm.replace(" ", "").replace("%", "")
        ),
        "token_f1": token_f1(generated, expected),
        "relaxed_exact": float(gen_compact == exp_compact),
        "answer_contains_expected": float(bool(exp_compact) and exp_compact in gen_compact),
        "expected_contains_answer": float(bool(gen_compact) and gen_compact in exp_compact),
        "numeric_any_match": float(
            bool(expected_counts) and any(n in generated_counts for n in expected_counts)
        ),
        "numeric_all_recall": (
            float(matched == sum(expected_counts.values()))
            if expected_counts
            else float(not generated_counts)
        ),
        "numeric_precision": (
            matched / sum(generated_counts.values())
            if generated_counts
            else float(not expected_counts)
        ),
        "numeric_recall": (
            matched / sum(expected_counts.values())
            if expected_counts
            else float(not generated_counts)
        ),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def _optional_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1 - frac) + values[high] * frac


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        metrics = row_metrics(item)
        item.setdefault("f1", metrics["token_f1"])
        item.update({k: item.get(k, v) for k, v in metrics.items()})
        enriched.append(item)
    return enriched


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = enrich_rows(rows)
    latencies = [float(row.get("latency", 0.0)) for row in rows if row.get("latency") is not None]
    retrieval_latencies = [
        float(row.get("latency_retrieval", row.get("retrieval_latency", 0.0)))
        for row in rows
        if row.get("latency_retrieval", row.get("retrieval_latency")) is not None
    ]
    rerank_latencies = [
        float(row.get("latency_rerank", row.get("rerank_latency", 0.0)))
        for row in rows
        if row.get("latency_rerank", row.get("rerank_latency")) is not None
    ]
    context_latencies = [
        float(row.get("latency_context", row.get("context_latency", 0.0)))
        for row in rows
        if row.get("latency_context", row.get("context_latency")) is not None
    ]
    vlm_latencies = [
        float(row.get("latency_vlm", row.get("vlm_latency", 0.0)))
        for row in rows
        if row.get("latency_vlm", row.get("vlm_latency")) is not None
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("modality") or row.get("type") or "unknown"].append(row)
    crop_used = [bool(row.get("crop_used") or row.get("crop_path")) for row in rows]
    crop_mismatch = [bool(row.get("crop_type_mismatch")) for row in rows]
    caption_match = [bool(row.get("caption_match")) for row in rows]
    total = max(len(rows), 1)
    latency_seconds = {
        "mean": mean(latencies) if latencies else 0.0,
        "p50": median(latencies) if latencies else 0.0,
        "p95": _percentile(latencies, 0.95),
    }
    return {
        "total": len(rows),
        "total_questions": len(rows),
        "exact_match": _mean(rows, "exact_match"),
        "mean_f1": _mean(rows, "f1"),
        "accuracy_f1_gt_0_5": (
            mean([float(row.get("f1", 0.0)) > 0.5 for row in rows]) if rows else 0.0
        ),
        "relaxed_exact": _mean(rows, "relaxed_exact"),
        "answer_contains_expected": _mean(rows, "answer_contains_expected"),
        "expected_contains_answer": _mean(rows, "expected_contains_answer"),
        "numeric_any_match": _mean(rows, "numeric_any_match"),
        "numeric_all_recall": _mean(rows, "numeric_all_recall"),
        "numeric_precision": _mean(rows, "numeric_precision"),
        "numeric_recall": _mean(rows, "numeric_recall"),
        "numeric_exact_match": _optional_mean(rows, "numeric_exact_match"),
        "numeric_relaxed_match": _optional_mean(rows, "numeric_relaxed_match"),
        "unit_match": _optional_mean(rows, "unit_match"),
        "entity_match": _optional_mean(rows, "entity_match"),
        "doc_hit_at_1": _optional_mean(rows, "doc_hit_at_1"),
        "doc_hit_at_5": _optional_mean(rows, "doc_hit_at_5"),
        "page_hit_at_1": _optional_mean(rows, "page_hit_at_1"),
        "page_hit_at_5": _optional_mean(rows, "page_hit_at_5"),
        "crop_used_rate": sum(crop_used) / total,
        "fallback_rate": sum([bool(row.get("fallback_used")) for row in rows]) / total,
        "latency_seconds": latency_seconds,
        "latency": latency_seconds,
        "latency_breakdown": {
            "retrieval": {
                "mean": mean(retrieval_latencies) if retrieval_latencies else 0.0,
                "p50": median(retrieval_latencies) if retrieval_latencies else 0.0,
                "p95": _percentile(retrieval_latencies, 0.95),
            },
            "rerank": {
                "mean": mean(rerank_latencies) if rerank_latencies else 0.0,
                "p50": median(rerank_latencies) if rerank_latencies else 0.0,
                "p95": _percentile(rerank_latencies, 0.95),
            },
            "context": {
                "mean": mean(context_latencies) if context_latencies else 0.0,
                "p50": median(context_latencies) if context_latencies else 0.0,
                "p95": _percentile(context_latencies, 0.95),
            },
            "vlm": {
                "mean": mean(vlm_latencies) if vlm_latencies else 0.0,
                "p50": median(vlm_latencies) if vlm_latencies else 0.0,
                "p95": _percentile(vlm_latencies, 0.95),
            },
        },
        "by_modality": {
            key: {
                "count": len(value),
                "mean_f1": _mean(value, "f1"),
                "accuracy_f1_gt_0_5": (
                    mean([float(row.get("f1", 0.0)) > 0.5 for row in value]) if value else 0.0
                ),
            }
            for key, value in grouped.items()
        },
        "grounding": {
            "crop_used_rate": sum(crop_used) / total,
            "crop_type_mismatch_rate": sum(crop_mismatch) / total,
            "caption_match_rate": sum(caption_match) / total,
        },
    }


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_metrics_artifacts(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_rows(rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_rows = [
        ("total questions", metrics["total_questions"]),
        ("exact_match", metrics["exact_match"]),
        ("mean_f1", metrics["mean_f1"]),
        ("f1 > 0.5", metrics["accuracy_f1_gt_0_5"]),
        ("multimodal-t mean_f1", metrics["by_modality"].get("multimodal-t", {}).get("mean_f1")),
        ("multimodal-f mean_f1", metrics["by_modality"].get("multimodal-f", {}).get("mean_f1")),
        ("latency mean", metrics["latency_seconds"]["mean"]),
    ]
    csv_text = "metric,value\n" + "\n".join(f"{name},{value}" for name, value in flat_rows) + "\n"
    (output_dir / "metrics_table.csv").write_text(csv_text, encoding="utf-8")
    md_text = (
        "| metric | value |\n|---|---:|\n"
        + "\n".join(f"| {name} | {value} |" for name, value in flat_rows)
        + "\n"
    )
    (output_dir / "metrics_table.md").write_text(md_text, encoding="utf-8")
    return metrics
