from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign one primary failure mode to every non-perfect VLM answer."
    )
    parser.add_argument("--vlm", type=Path, required=True, help="VLM evaluation JSON.")
    parser.add_argument(
        "--ranking", type=Path, default=None, help="Reranker evaluation JSON with oracle pages."
    )
    parser.add_argument(
        "--baseline-vlm", type=Path, default=None, help="Optional baseline VLM JSON."
    )
    parser.add_argument("--output", type=Path, default=Path("data/diagnose_vlm_failure_modes.json"))
    parser.add_argument("--markdown", type=Path, default=Path("data/diagnose_vlm_failure_modes.md"))
    parser.add_argument("--examples-per-mode", type=int, default=12)
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
    "score",
    "value",
}


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def rows_from(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("results") or payload.get("rows") or []


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("−", "-")
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_number(number: str) -> str:
    value = number.strip().lower().replace(",", "")
    if value.endswith("%"):
        return value
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: str) -> list[str]:
    return [
        normalize_number(num) for num in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", text or "")
    ]


def extract_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9_+\-]*", normalize_text(text))
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def f1_sets(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    intersection = pred & gold
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def question_key(row: dict[str, Any]) -> str:
    return normalize_text(row.get("question", ""))


def page_key(folder: str | int | None, page: str | int | None) -> str:
    return f"{folder}_{page}"


def selected_pages(row: dict[str, Any]) -> set[str]:
    return {page_key(page.get("folder"), page.get("page")) for page in row.get("pages", [])}


def selected_folders(row: dict[str, Any]) -> set[str]:
    return {
        str(page.get("folder")) for page in row.get("pages", []) if page.get("folder") is not None
    }


def oracle_pages(row: dict[str, Any] | None) -> list[int]:
    if not row:
        return []
    pages = []
    for item in row.get("oracle_pages", []):
        if isinstance(item, int):
            pages.append(item)
        elif isinstance(item, dict) and "page" in item:
            pages.append(int(item["page"]))
    return pages


def is_truncated(answer: str) -> bool:
    answer = (answer or "").strip()
    if not answer or answer in {"ERROR", "NOT FOUND", "TIMEOUT"}:
        return False
    if answer.endswith((".", "%", ")", "]", '"', "'")):
        return False
    if len(answer.split()) <= 4:
        return True
    return bool(re.search(r"\b(the|a|an|of|from|to|with|and|or|by|in|for)$", answer.lower()))


def has_reasoning_leak(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if "internal thought" in text:
        return True
    reasoning_starts = (
        "locate ",
        "scan ",
        "identify ",
        "find ",
        "calculate ",
        "reviewing ",
        "the question asks",
        "the document discusses",
        "the table ",
        "table ",
        "figure ",
        "image ",
    )
    if text.startswith(reasoning_starts):
        return True
    return bool(
        re.search(
            r"\b(locate|scan|identify|find)\s+(the|row|column|table|figure|value)\b",
            text,
        )
    )


def number_sets(generated: str, expected: str) -> tuple[set[str], set[str], set[str]]:
    gen_nums = set(extract_numbers(generated))
    exp_nums = set(extract_numbers(expected))
    overlap = gen_nums & exp_nums
    return gen_nums, exp_nums, overlap


def assign_failure_mode(
    vlm_row: dict[str, Any],
    ranking_row: dict[str, Any] | None,
    baseline_row: dict[str, Any] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    f1 = float(vlm_row.get("f1", 0.0))
    answer = vlm_row.get("generated", "") or ""
    expected = vlm_row.get("expected", "") or ""
    expected_folder = str(vlm_row.get("expected_folder"))
    folders = selected_folders(vlm_row)
    pages = selected_pages(vlm_row)
    oracle = oracle_pages(ranking_row)
    oracle_keys = {page_key(expected_folder, page) for page in oracle}
    gen_nums, exp_nums, num_overlap = number_sets(answer, expected)
    word_f1 = f1_sets(extract_words(answer), extract_words(expected))
    baseline_f1 = float(baseline_row.get("f1", 0.0)) if baseline_row else None
    delta = f1 - baseline_f1 if baseline_f1 is not None else None

    evidence = {
        "f1": f1,
        "word_f1": word_f1,
        "generated_numbers": sorted(gen_nums),
        "expected_numbers": sorted(exp_nums),
        "number_overlap": sorted(num_overlap),
        "selected_folders": sorted(folders),
        "selected_pages": sorted(pages),
        "oracle_pages": oracle,
        "baseline_f1": baseline_f1,
        "delta_vs_baseline": delta,
    }

    secondary = []
    reasoning_leak = has_reasoning_leak(answer)
    if reasoning_leak:
        secondary.append("prompt_leak")
    if is_truncated(answer):
        secondary.append("truncated")
    if baseline_f1 is not None:
        if delta is not None and delta > 0.05:
            secondary.append("reranker_helped_answer")
        elif delta is not None and delta < -0.05:
            secondary.append("reranker_hurt_answer")

    if f1 >= 0.999:
        return "correct", secondary, evidence

    if answer in {"ERROR", "NOT FOUND", "TIMEOUT"}:
        return "runtime_or_not_found", secondary, evidence

    if reasoning_leak:
        return "prompt_leak_or_reasoning_output", secondary, evidence

    if is_truncated(answer):
        return "truncated_generation", secondary, evidence

    if expected_folder and expected_folder != "None" and expected_folder not in folders:
        return "wrong_document_in_vlm_context", secondary, evidence

    if oracle and not (pages & oracle_keys):
        return "answer_page_missing_from_vlm_context", secondary, evidence

    if exp_nums:
        if not gen_nums:
            return "missing_numeric_answer", secondary, evidence
        if exp_nums <= gen_nums or {num.rstrip("%") for num in exp_nums} <= {
            num.rstrip("%") for num in gen_nums
        }:
            if f1 < 0.5:
                return "metric_or_format_undercredits_correct_number", secondary, evidence
            return "partially_correct_format_mismatch", secondary, evidence
        if num_overlap:
            return "partial_numeric_answer", secondary, evidence
        return "wrong_number_or_wrong_table_cell", secondary, evidence

    if word_f1 >= 0.75:
        return "metric_undercredits_paraphrase", secondary, evidence

    if oracle and pages & oracle_keys:
        return "vlm_grounding_or_reasoning_failure", secondary, evidence

    return "semantic_mismatch_uncategorized", secondary, evidence


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1_values = [float(row["f1"]) for row in rows]
    exact_values = [float(row["exact"]) for row in rows]
    by_mode: Counter[str] = Counter(row["failure_mode"] for row in rows)
    by_type: dict[str, list[float]] = defaultdict(list)
    by_type_mode: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        qtype = row.get("type", "")
        by_type[qtype].append(float(row["f1"]))
        by_type_mode[qtype][row["failure_mode"]] += 1
    metric_keys = [
        "relaxed_exact",
        "answer_contains_expected",
        "expected_contains_answer",
        "numeric_any_match",
        "numeric_all_recall",
        "numeric_precision",
        "numeric_recall",
    ]
    present_metric_keys = [key for key in metric_keys if any(key in row for row in rows)]

    summary = {
        "total": len(rows),
        "non_perfect": sum(1 for row in rows if row["failure_mode"] != "correct"),
        "mean_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "exact_match": float(np.mean(exact_values)) if exact_values else 0.0,
        "accuracy_f1_gt_0_5": (
            float(np.mean([score > 0.5 for score in f1_values])) if f1_values else 0.0
        ),
        "failure_modes": by_mode.most_common(),
        "by_type": {
            qtype: {
                "count": len(values),
                "mean_f1": float(np.mean(values)) if values else 0.0,
                "accuracy_f1_gt_0_5": (
                    float(np.mean([score > 0.5 for score in values])) if values else 0.0
                ),
                "failure_modes": by_type_mode[qtype].most_common(),
            }
            for qtype, values in sorted(by_type.items())
        },
    }
    if present_metric_keys:
        summary["extended_metrics"] = {
            key: float(np.mean([float(row.get(key, 0.0)) for row in rows]))
            for key in present_metric_keys
        }
    return summary


def compact_row(
    row: dict[str, Any],
    mode: str,
    secondary: list[str],
    evidence: dict[str, Any],
    ranking_row: dict[str, Any] | None,
    baseline_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "type": row.get("type"),
        "expected_folder": row.get("expected_folder"),
        "pages": [f"{page.get('folder')}/{page.get('page')}" for page in row.get("pages", [])],
        "oracle_pages": oracle_pages(ranking_row),
        "expected": row.get("expected"),
        "generated": row.get("generated"),
        "f1": row.get("f1"),
        "exact": row.get("exact"),
        "failure_mode": mode,
        "secondary": secondary,
        "evidence": evidence,
        "baseline_f1": baseline_row.get("f1") if baseline_row else None,
        "baseline_generated": baseline_row.get("generated") if baseline_row else None,
    }


def write_markdown(
    path: Path, summary: dict[str, Any], examples_by_mode: dict[str, list[dict[str, Any]]]
) -> None:
    lines = ["# VLM Failure Modes", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total: {summary['total']}")
    lines.append(f"- Non-perfect: {summary['non_perfect']}")
    lines.append(f"- Mean F1: {summary['mean_f1']:.4f}")
    lines.append(f"- Accuracy F1>0.5: {summary['accuracy_f1_gt_0_5']:.4f}")
    lines.append("")
    lines.append("## Failure Modes")
    lines.append("")
    for mode, count in summary["failure_modes"]:
        lines.append(f"- {mode}: {count}")

    for mode, examples in examples_by_mode.items():
        lines.append("")
        lines.append(f"## {mode}")
        for item in examples:
            lines.append("")
            lines.append(f"- F1={item['f1']} secondary={item['secondary']}")
            lines.append(f"  - Pages: {item['pages']}")
            if item["oracle_pages"]:
                lines.append(f"  - Oracle pages: {item['oracle_pages']}")
            lines.append(f"  - Q: {item['question']}")
            lines.append(f"  - Expected: {item['expected']}")
            lines.append(f"  - Generated: {item['generated']}")
            if item.get("baseline_generated") is not None:
                lines.append(f"  - Baseline generated: {item['baseline_generated']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vlm_payload = load_payload(args.vlm)
    vlm_rows = rows_from(vlm_payload)

    ranking_by_question = {}
    if args.ranking:
        ranking_by_question = {
            question_key(row): row for row in rows_from(load_payload(args.ranking))
        }

    baseline_by_question = {}
    if args.baseline_vlm:
        baseline_by_question = {
            question_key(row): row for row in rows_from(load_payload(args.baseline_vlm))
        }

    diagnosed = []
    examples_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vlm_rows:
        key = question_key(row)
        ranking_row = ranking_by_question.get(key)
        baseline_row = baseline_by_question.get(key)
        mode, secondary, evidence = assign_failure_mode(row, ranking_row, baseline_row)
        item = compact_row(row, mode, secondary, evidence, ranking_row, baseline_row)
        diagnosed.append(item)
        if mode != "correct" and len(examples_by_mode[mode]) < args.examples_per_mode:
            examples_by_mode[mode].append(item)

    summary = summarize(diagnosed)
    output = {
        "summary": summary,
        "examples_by_mode": examples_by_mode,
        "rows": diagnosed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    write_markdown(args.markdown, summary, examples_by_mode)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Saved markdown: {args.markdown}")


if __name__ == "__main__":
    main()
