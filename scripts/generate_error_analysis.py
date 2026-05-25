from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import load_predictions_jsonl
from src.reporting.error_analysis import write_error_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate f1<0.5 error analysis table.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/error_analysis/error_cases.csv")
    )
    args = parser.parse_args()
    write_error_cases(load_predictions_jsonl(args.predictions), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
