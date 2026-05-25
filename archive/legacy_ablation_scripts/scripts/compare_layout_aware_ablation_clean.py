from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare visual_main vs layout-aware crop runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/layout_aware_ablation"))
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, payload.get("results") or payload.get("rows") or []


def by_question(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_question(row.get("question", "")): row for row in rows}


def group_mean_f1(rows: list[dict[str, Any]], group: str) -> float | None:
    selected = [float(row.get("f1", 0.0)) for row in rows if row.get("type") == group]
    return mean(selected) if selected else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency", 0.0)) for row in rows]
    return {
        "total": len(rows),
        "mean_f1": mean([float(row.get("f1", 0.0)) for row in rows]) if rows else 0.0,
        "accuracy_f1_gt_0_5": (
            mean([float(row.get("f1", 0.0)) > 0.5 for row in rows]) if rows else 0.0
        ),
        "multimodal_t_mean_f1": group_mean_f1(rows, "multimodal-t"),
        "multimodal_f_mean_f1": group_mean_f1(rows, "multimodal-f"),
        "latency_mean": mean(latencies) if latencies else 0.0,
    }


def metric_delta(exp: dict[str, Any], base: dict[str, Any], key: str) -> float | None:
    if exp.get(key) is None or base.get(key) is None:
        return None
    return float(exp[key]) - float(base[key])


def crop_debug_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(rows), 1)
    crop_used = [bool(row.get("crop_path")) and not bool(row.get("fallback_used")) for row in rows]
    mismatch = [bool(row.get("crop_type_mismatch")) for row in rows]
    caption_match = [bool(row.get("caption_match")) for row in rows]
    fallback = [bool(row.get("fallback_used")) for row in rows]
    mismatch_rows = [row for row, flag in zip(rows, mismatch) if flag]
    caption_rows = [row for row, flag in zip(rows, caption_match) if flag]
    return {
        "crop_used_rate": sum(crop_used) / total,
        "crop_type_mismatch_rate": sum(mismatch) / total,
        "caption_match_rate": sum(caption_match) / total,
        "fallback_rate": sum(fallback) / total,
        "mean_f1_when_crop_type_mismatch_true": (
            mean([float(row.get("f1", 0.0)) for row in mismatch_rows]) if mismatch_rows else None
        ),
        "mean_f1_when_caption_match_true": (
            mean([float(row.get("f1", 0.0)) for row in caption_rows]) if caption_rows else None
        ),
    }


def compact_case(base: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": exp.get("question") or base.get("question"),
        "type": exp.get("type") or base.get("type"),
        "expected": exp.get("expected") or base.get("expected"),
        "baseline_generated": base.get("generated"),
        "experiment_generated": exp.get("generated"),
        "baseline_f1": float(base.get("f1", 0.0)),
        "experiment_f1": float(exp.get("f1", 0.0)),
        "delta_f1": float(exp.get("f1", 0.0)) - float(base.get("f1", 0.0)),
        "baseline_latency": base.get("latency"),
        "experiment_latency": exp.get("latency"),
        "question_crop_intent": exp.get("question_crop_intent"),
        "explicit_reference": exp.get("explicit_reference"),
        "selected_crop_type": exp.get("selected_crop_type"),
        "selected_crop_caption": exp.get("selected_crop_caption"),
        "selected_crop_score": exp.get("selected_crop_score"),
        "crop_type_mismatch": exp.get("crop_type_mismatch"),
        "caption_match": exp.get("caption_match"),
        "fallback_used": exp.get("fallback_used"),
        "crop_used": exp.get("crop_used"),
        "full_page_plus_crop": exp.get("full_page_plus_crop"),
        "crop_path": exp.get("crop_path"),
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
        "baseline_latency",
        "experiment_latency",
        "question_crop_intent",
        "explicit_reference",
        "selected_crop_type",
        "selected_crop_caption",
        "selected_crop_score",
        "crop_type_mismatch",
        "caption_match",
        "fallback_used",
        "crop_used",
        "full_page_plus_crop",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _baseline_payload, baseline_rows = load_rows(args.baseline)
    _experiment_payload, experiment_rows = load_rows(args.experiment)

    baseline_by_question = by_question(baseline_rows)
    paired = []
    missing = []
    for exp in experiment_rows:
        base = baseline_by_question.get(normalize_question(exp.get("question", "")))
        if not base:
            missing.append(exp.get("question", ""))
            continue
        paired.append(compact_case(base, exp))

    improved = sorted(
        [row for row in paired if row["delta_f1"] > 0],
        key=lambda row: row["delta_f1"],
        reverse=True,
    )
    regressed = sorted(
        [row for row in paired if row["delta_f1"] < 0], key=lambda row: row["delta_f1"]
    )

    baseline_summary = summary(baseline_rows)
    experiment_summary = summary(experiment_rows)
    metric_keys = [
        "mean_f1",
        "accuracy_f1_gt_0_5",
        "multimodal_t_mean_f1",
        "multimodal_f_mean_f1",
        "latency_mean",
    ]
    output = {
        "baseline": str(args.baseline),
        "experiment": str(args.experiment),
        "paired_questions": len(paired),
        "missing_questions": missing,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(paired) - len(improved) - len(regressed),
        "metric_delta": {
            key: metric_delta(experiment_summary, baseline_summary, key) for key in metric_keys
        },
        "baseline_summary": baseline_summary,
        "experiment_summary": experiment_summary,
        "crop_debug": crop_debug_summary(experiment_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "improved_cases.csv", improved)
    write_csv(args.output_dir / "regressed_cases.csv", regressed)
    with (args.output_dir / "crop_debug_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
