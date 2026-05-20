from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze VLM answer errors for MRAG experiments.")
    parser.add_argument("--vlm", type=Path, required=True, help="VLM evaluation JSON.")
    parser.add_argument(
        "--ranking",
        type=Path,
        default=None,
        help="Optional reranker evaluation JSON with oracle pages.",
    )
    parser.add_argument(
        "--baseline-vlm",
        type=Path,
        default=None,
        help="Optional baseline VLM JSON for paired comparison.",
    )
    parser.add_argument("--worst-k", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/analyze_vlm_errors_clean.json"))
    parser.add_argument("--markdown", type=Path, default=Path("data/analyze_vlm_errors_clean.md"))
    return parser.parse_args()


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
    "been",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "can",
    "answer",
    "question",
    "according",
    "table",
    "figure",
}


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("−", "-")
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text: str) -> list[str]:
    text = normalize_text(text)
    return re.findall(r"-?\d+(?:\.\d+)?%?", text)


def extract_words(text: str) -> set[str]:
    text = normalize_text(text)
    return {
        word
        for word in re.findall(r"[a-z][a-z0-9_+\-]*", text)
        if len(word) > 2 and word not in STOP_WORDS
    }


def f1_from_sets(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = pred & gold
    precision = len(inter) / len(pred)
    recall = len(inter) / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def question_key(row: dict[str, Any]) -> str:
    return normalize_text(row.get("question", ""))


def oracle_pages_from_ranking_row(row: dict[str, Any]) -> list[int]:
    pages = []
    for item in row.get("oracle_pages", []):
        if isinstance(item, int):
            pages.append(item)
        elif isinstance(item, dict) and "page" in item:
            pages.append(int(item["page"]))
    return pages


def page_key(folder: str | int | None, page: str | int | None) -> str:
    return f"{folder}_{page}"


def selected_page_keys(row: dict[str, Any]) -> set[str]:
    return {page_key(page.get("folder"), page.get("page")) for page in row.get("pages", [])}


def selected_folders(row: dict[str, Any]) -> set[str]:
    return {
        str(page.get("folder")) for page in row.get("pages", []) if page.get("folder") is not None
    }


def likely_truncated(answer: str) -> bool:
    stripped = (answer or "").strip()
    if not stripped:
        return True
    if stripped in {"ERROR", "NOT FOUND", "TIMEOUT"}:
        return False
    if stripped.endswith((".", "%", ")", "]", '"', "'")):
        return False
    words = stripped.split()
    if len(words) <= 3:
        return True
    return bool(re.search(r"\b(the|a|an|of|from|to|with|and|or|by|in|for)$", stripped.lower()))


def has_internal_thought(answer: str) -> bool:
    return "internal thought" in (answer or "").lower()


def classify_error(vlm_row: dict[str, Any], ranking_row: dict[str, Any] | None) -> list[str]:
    labels = []
    answer = vlm_row.get("generated", "")
    expected = vlm_row.get("expected", "")
    expected_folder = str(vlm_row.get("expected_folder"))
    folders = selected_folders(vlm_row)

    if answer in {"ERROR", "NOT FOUND", "TIMEOUT"}:
        labels.append("runtime_or_not_found")
    if has_internal_thought(answer):
        labels.append("prompt_leaks_internal_thought")
    if likely_truncated(answer):
        labels.append("likely_truncated_answer")

    if expected_folder and expected_folder != "None" and expected_folder not in folders:
        labels.append("selected_pages_missing_expected_folder")

    oracle_pages = oracle_pages_from_ranking_row(ranking_row) if ranking_row else []
    if oracle_pages:
        oracle_keys = {page_key(expected_folder, page) for page in oracle_pages}
        if not selected_page_keys(vlm_row) & oracle_keys:
            labels.append("selected_pages_missing_oracle_page")
        else:
            labels.append("oracle_page_available_to_vlm")

    generated_numbers = set(extract_numbers(answer))
    expected_numbers = set(extract_numbers(expected))
    generated_number_stripped = {num.rstrip("%") for num in generated_numbers}
    expected_number_stripped = {num.rstrip("%") for num in expected_numbers}
    if expected_numbers:
        if not generated_numbers:
            labels.append("expected_numeric_answer_but_no_generated_number")
        elif generated_number_stripped & expected_number_stripped:
            labels.append("expected_number_present")
        else:
            labels.append("numeric_mismatch")

    word_f1 = f1_from_sets(extract_words(answer), extract_words(expected))
    if word_f1 >= 0.7 and vlm_row.get("f1", 0.0) < 0.5:
        labels.append("metric_undercredits_paraphrase")

    if vlm_row.get("f1", 0.0) < 0.5 and "oracle_page_available_to_vlm" in labels:
        labels.append("vlm_failed_despite_oracle_page")
    if vlm_row.get("f1", 0.0) < 0.5 and "selected_pages_missing_oracle_page" in labels:
        labels.append("context_selection_failure")

    if not labels:
        labels.append("mostly_correct_or_uncategorized")
    return labels


def bin_f1(score: float) -> str:
    if score >= 0.9:
        return "0.90-1.00"
    if score >= 0.7:
        return "0.70-0.89"
    if score >= 0.5:
        return "0.50-0.69"
    if score >= 0.25:
        return "0.25-0.49"
    if score > 0:
        return "0.01-0.24"
    return "0.00"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("results") or payload.get("rows") or []


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def compact_example(
    row: dict[str, Any], labels: list[str], ranking_row: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "type": row.get("type"),
        "expected_folder": row.get("expected_folder"),
        "pages": [f"{page.get('folder')}/{page.get('page')}" for page in row.get("pages", [])],
        "oracle_pages": oracle_pages_from_ranking_row(ranking_row) if ranking_row else [],
        "expected": row.get("expected"),
        "generated": row.get("generated"),
        "f1": row.get("f1"),
        "exact": row.get("exact"),
        "labels": labels,
    }


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    worst: list[dict[str, Any]],
    grouped_examples: dict[str, list[dict[str, Any]]],
) -> None:
    lines = []
    lines.append("# VLM Error Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total: {summary['total']}")
    lines.append(f"- Mean F1: {summary['mean_f1']:.4f}")
    lines.append(f"- Accuracy F1>0.5: {summary['accuracy_f1_gt_0_5']:.4f}")
    lines.append(f"- Exact match: {summary['exact_match']:.4f}")
    lines.append("")
    lines.append("## Error Labels")
    lines.append("")
    for label, count in summary["label_counts"]:
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("## F1 Bins")
    lines.append("")
    for label, count in summary["f1_bins"]:
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("## Worst Examples")
    for idx, row in enumerate(worst, start=1):
        lines.append("")
        lines.append(f"### {idx}. F1={row['f1']} `{', '.join(row['labels'])}`")
        lines.append(f"- Type: {row['type']}")
        lines.append(f"- Expected folder: {row['expected_folder']}")
        lines.append(f"- Pages: {row['pages']}")
        if row.get("oracle_pages"):
            lines.append(f"- Oracle pages: {row['oracle_pages']}")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Expected: {row['expected']}")
        lines.append(f"- Generated: {row['generated']}")
    lines.append("")
    lines.append("## Examples By Label")
    for label, examples in grouped_examples.items():
        lines.append("")
        lines.append(f"### {label}")
        for row in examples[:5]:
            lines.append("")
            lines.append(f"- F1={row['f1']} pages={row['pages']}")
            lines.append(f"  - Q: {row['question']}")
            lines.append(f"  - E: {row['expected']}")
            lines.append(f"  - G: {row['generated']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vlm_rows = load_rows(args.vlm)

    ranking_by_question = {}
    if args.ranking:
        ranking_rows = load_rows(args.ranking)
        ranking_by_question = {question_key(row): row for row in ranking_rows}

    baseline_by_question = {}
    if args.baseline_vlm:
        baseline_rows = load_rows(args.baseline_vlm)
        baseline_by_question = {question_key(row): row for row in baseline_rows}

    label_counts: Counter[str] = Counter()
    f1_bins: Counter[str] = Counter()
    by_type: dict[str, list[float]] = defaultdict(list)
    grouped_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    analyzed_rows = []
    paired_delta_rows = []

    for row in vlm_rows:
        key = question_key(row)
        ranking_row = ranking_by_question.get(key)
        labels = classify_error(row, ranking_row)
        for label in labels:
            label_counts[label] += 1
        f1_bins[bin_f1(float(row.get("f1", 0.0)))] += 1
        by_type[row.get("type", "")].append(float(row.get("f1", 0.0)))

        compact = compact_example(row, labels, ranking_row)
        analyzed_rows.append(compact)
        for label in labels:
            if len(grouped_examples[label]) < args.worst_k:
                grouped_examples[label].append(compact)

        baseline = baseline_by_question.get(key)
        if baseline:
            paired_delta_rows.append(
                {
                    **compact,
                    "baseline_f1": baseline.get("f1"),
                    "reranked_f1": row.get("f1"),
                    "delta_f1": float(row.get("f1", 0.0)) - float(baseline.get("f1", 0.0)),
                    "baseline_generated": baseline.get("generated"),
                    "baseline_pages": [
                        f"{page.get('folder')}/{page.get('page')}"
                        for page in baseline.get("pages", [])
                    ],
                }
            )

    f1_values = [float(row.get("f1", 0.0)) for row in vlm_rows]
    exact_values = [float(row.get("exact", 0.0)) for row in vlm_rows]
    latency_values = [
        float(row.get("latency", 0.0)) for row in vlm_rows if row.get("latency") is not None
    ]

    worst = sorted(analyzed_rows, key=lambda row: (row["f1"], row["exact"]))[: args.worst_k]
    paired_improved = sorted(paired_delta_rows, key=lambda row: row["delta_f1"], reverse=True)[
        : args.worst_k
    ]
    paired_regressed = sorted(paired_delta_rows, key=lambda row: row["delta_f1"])[: args.worst_k]

    summary = {
        "total": len(vlm_rows),
        "mean_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "exact_match": float(np.mean(exact_values)) if exact_values else 0.0,
        "accuracy_f1_gt_0_5": (
            float(np.mean([score > 0.5 for score in f1_values])) if f1_values else 0.0
        ),
        "latency": summarize_numeric(latency_values),
        "by_type": {
            key: {
                "count": len(values),
                "mean_f1": float(np.mean(values)) if values else 0.0,
                "accuracy_f1_gt_0_5": (
                    float(np.mean([score > 0.5 for score in values])) if values else 0.0
                ),
            }
            for key, values in sorted(by_type.items())
        },
        "label_counts": label_counts.most_common(),
        "f1_bins": sorted(f1_bins.items()),
    }

    output = {
        "summary": summary,
        "worst_examples": worst,
        "examples_by_label": grouped_examples,
        "paired_improved_vs_baseline": paired_improved,
        "paired_regressed_vs_baseline": paired_regressed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    write_markdown(args.markdown, summary, worst, grouped_examples)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if paired_delta_rows:
        mean_delta = float(np.mean([row["delta_f1"] for row in paired_delta_rows]))
        improved = sum(1 for row in paired_delta_rows if row["delta_f1"] > 0.05)
        regressed = sum(1 for row in paired_delta_rows if row["delta_f1"] < -0.05)
        print(
            json.dumps(
                {
                    "paired_vs_baseline": {
                        "mean_delta_f1": mean_delta,
                        "improved_by_gt_0_05": improved,
                        "regressed_by_lt_minus_0_05": regressed,
                    }
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    print(f"Saved: {args.output}")
    print(f"Saved markdown: {args.markdown}")


if __name__ == "__main__":
    main()
