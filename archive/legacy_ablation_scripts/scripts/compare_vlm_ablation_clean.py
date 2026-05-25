from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs prompt-ablation VLM evaluation outputs."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prompt_ablation_compare"))
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results") or payload.get("rows") or []
    return payload, rows


def ensure_summary(_payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group_rows),
            "exact_match": (
                mean([float(row.get("exact", 0.0)) for row in group_rows]) if group_rows else 0.0
            ),
            "mean_f1": (
                mean([float(row.get("f1", 0.0)) for row in group_rows]) if group_rows else 0.0
            ),
            "accuracy_f1_gt_0_5": (
                mean([float(row.get("f1", 0.0)) > 0.5 for row in group_rows]) if group_rows else 0.0
            ),
        }

    summary = {
        "total": len(rows),
        "exact_match": mean([float(row.get("exact", 0.0)) for row in rows]) if rows else 0.0,
        "mean_f1": mean([float(row.get("f1", 0.0)) for row in rows]) if rows else 0.0,
        "accuracy_f1_gt_0_5": (
            mean([float(row.get("f1", 0.0)) > 0.5 for row in rows]) if rows else 0.0
        ),
        "latency_seconds": {
            "mean": mean([float(row.get("latency", 0.0)) for row in rows]) if rows else 0.0,
        },
        "by_type": {},
    }
    for group_name in sorted({row.get("type", "unknown") for row in rows}):
        summary["by_type"][group_name] = summarize_group(
            [row for row in rows if row.get("type", "unknown") == group_name]
        )
    bad_rows = [row for row in rows if float(row.get("f1", 0.0)) < 0.5]
    summary["f1_lt_0_5"] = summarize_group(bad_rows)
    return summary


def metric_delta(experiment: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    exp_value = experiment.get(key)
    base_value = baseline.get(key)
    if exp_value is None or base_value is None:
        return None
    return float(exp_value) - float(base_value)


def compact_row(baseline: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": baseline.get("question") or experiment.get("question"),
        "type": baseline.get("type") or experiment.get("type"),
        "expected": baseline.get("expected") or experiment.get("expected"),
        "baseline_generated": baseline.get("generated"),
        "experiment_generated": experiment.get("generated"),
        "baseline_f1": float(baseline.get("f1", 0.0)),
        "experiment_f1": float(experiment.get("f1", 0.0)),
        "delta_f1": float(experiment.get("f1", 0.0)) - float(baseline.get("f1", 0.0)),
        "baseline_exact": baseline.get("exact"),
        "experiment_exact": experiment.get("exact"),
        "baseline_latency": baseline.get("latency"),
        "experiment_latency": experiment.get("latency"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "type",
        "expected",
        "baseline_generated",
        "experiment_generated",
        "baseline_f1",
        "experiment_f1",
        "delta_f1",
        "baseline_exact",
        "experiment_exact",
        "baseline_latency",
        "experiment_latency",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_payload, baseline_rows = load_results(args.baseline)
    experiment_payload, experiment_rows = load_results(args.experiment)

    experiment_by_question = {
        normalize_question(row.get("question", "")): row for row in experiment_rows
    }

    paired = []
    missing = []
    for baseline_row in baseline_rows:
        key = normalize_question(baseline_row.get("question", ""))
        experiment_row = experiment_by_question.get(key)
        if experiment_row is None:
            missing.append(baseline_row.get("question", ""))
            continue
        paired.append(compact_row(baseline_row, experiment_row))

    improved = sorted(
        [row for row in paired if row["delta_f1"] > 0],
        key=lambda row: row["delta_f1"],
        reverse=True,
    )
    regressed = sorted(
        [row for row in paired if row["delta_f1"] < 0], key=lambda row: row["delta_f1"]
    )
    still_bad = sorted(
        [row for row in paired if row["experiment_f1"] < 0.5], key=lambda row: row["experiment_f1"]
    )

    baseline_summary = ensure_summary(baseline_payload, baseline_rows)
    experiment_summary = ensure_summary(experiment_payload, experiment_rows)

    metric_keys = [
        "exact_match",
        "mean_f1",
        "accuracy_f1_gt_0_5",
    ]

    def group(summary: dict[str, Any], name: str) -> dict[str, Any]:
        return summary.get("by_type", {}).get(name, {})

    comparison = {
        "baseline": str(args.baseline),
        "experiment": str(args.experiment),
        "paired_questions": len(paired),
        "missing_questions": missing,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(paired) - len(improved) - len(regressed),
        "still_bad_count": len(still_bad),
        "overall_delta": {
            key: metric_delta(experiment_summary, baseline_summary, key) for key in metric_keys
        },
        "latency_mean_delta": metric_delta(
            experiment_summary.get("latency_seconds", {}),
            baseline_summary.get("latency_seconds", {}),
            "mean",
        ),
        "multimodal_t_delta": {
            key: metric_delta(
                group(experiment_summary, "multimodal-t"),
                group(baseline_summary, "multimodal-t"),
                key,
            )
            for key in metric_keys
        },
        "multimodal_f_delta": {
            key: metric_delta(
                group(experiment_summary, "multimodal-f"),
                group(baseline_summary, "multimodal-f"),
                key,
            )
            for key in metric_keys
        },
        "baseline_summary": baseline_summary,
        "experiment_summary": experiment_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cases_improved.csv", improved)
    write_csv(args.output_dir / "cases_regressed.csv", regressed)
    write_csv(args.output_dir / "cases_still_bad.csv", still_bad)
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
