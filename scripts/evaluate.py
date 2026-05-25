from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import load_predictions_jsonl, write_metrics_artifacts
from src.reporting.error_analysis import write_error_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions.jsonl artifacts.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.predictions.parent
    rows = load_predictions_jsonl(args.predictions)
    metrics = write_metrics_artifacts(rows, output_dir)
    write_error_cases(rows, output_dir / "error_cases.csv")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {output_dir}")


if __name__ == "__main__":
    main()
