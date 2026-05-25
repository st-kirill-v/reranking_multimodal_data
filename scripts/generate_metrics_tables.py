from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.tables import collect_metric_rows, write_markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics tables across experiments.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/tables/metrics_by_experiment.md")
    )
    args = parser.parse_args()
    write_markdown_table(collect_metric_rows(args.results_dir), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
