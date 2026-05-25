from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lightweight plot data for paper figures."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(args.results_dir.glob("**/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": metrics_path.parent.name,
                "mean_f1": metrics.get("mean_f1"),
                "multimodal_t_mean_f1": metrics.get("by_modality", {})
                .get("multimodal-t", {})
                .get("mean_f1"),
                "multimodal_f_mean_f1": metrics.get("by_modality", {})
                .get("multimodal-f", {})
                .get("mean_f1"),
                "latency_mean": metrics.get("latency_seconds", {}).get("mean"),
            }
        )
    (args.output_dir / "plot_data.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved plot data to {args.output_dir / 'plot_data.json'}")


if __name__ == "__main__":
    main()
