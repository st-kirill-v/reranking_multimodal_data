from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.plots import write_placeholder_plots_note
from src.reporting.tables import collect_metric_rows, write_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-facing reports from results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_metric_rows(args.results_dir)
    write_markdown_table(rows, args.reports_dir / "tables" / "metrics_by_experiment.md")
    write_placeholder_plots_note(args.reports_dir / "figures" / "README.md")
    print(f"Report artifacts saved under {args.reports_dir}")


if __name__ == "__main__":
    main()
