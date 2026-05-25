from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_metric_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": metrics_path.parent.name,
                "exact_match": metrics.get("exact_match"),
                "mean_f1": metrics.get("mean_f1"),
                "f1>0.5": metrics.get("accuracy_f1_gt_0_5"),
                "multimodal-t mean_f1": metrics.get("by_modality", {})
                .get("multimodal-t", {})
                .get("mean_f1"),
                "multimodal-f mean_f1": metrics.get("by_modality", {})
                .get("multimodal-f", {})
                .get("mean_f1"),
                "latency": metrics.get("latency_seconds", {}).get("mean"),
            }
        )
    return rows


def write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
