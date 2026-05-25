from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.metrics import vqa_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute VLM answer/retrieval metrics for an existing evaluation JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing VLM evaluation JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the augmented JSON. Defaults to overwriting --input.",
    )
    return parser.parse_args()


def candidate_pages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("pages") or row.get("selected_pages") or []


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("results") or payload.get("rows") or []
    for row in rows:
        generated = row.get("generated") or row.get("answer") or ""
        expected = row.get("expected") or row.get("gold") or row.get("answer_expected") or ""
        if "exact" not in row or "f1" not in row:
            exact, f1 = vqa_metrics.compute_similarity(generated, expected)
            row["exact"] = exact
            row["f1"] = f1
        answer_metrics = vqa_metrics.compute_extended_metrics(generated, expected)
        for key in (
            "relaxed_exact",
            "answer_contains_expected",
            "expected_contains_answer",
            "numeric_any_match",
            "numeric_all_recall",
            "numeric_precision",
            "numeric_recall",
            "numeric_exact_match",
            "numeric_relaxed_match",
            "unit_match",
            "entity_match",
        ):
            if key not in row or key in {
                "numeric_exact_match",
                "numeric_relaxed_match",
                "unit_match",
                "entity_match",
            }:
                row[key] = answer_metrics[key]
        row.update(vqa_metrics.compute_retrieval_metrics(row, candidate_pages(row)))

    latencies = [float(row["latency"]) for row in rows if row.get("latency") is not None]
    payload["summary"] = vqa_metrics.summarize_answer_results(rows, latencies)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
