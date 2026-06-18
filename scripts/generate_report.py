from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.plots import write_placeholder_plots_note
from src.reporting.tables import (
    PAPER_COLUMNS,
    collect_metric_rows,
    filter_paper_rows,
    write_csv_table,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-facing reports from results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_metric_rows(args.results_dir)
    paper_rows = filter_paper_rows(rows)
    write_markdown_table(rows, args.reports_dir / "tables" / "metrics_by_experiment.md")
    write_csv_table(rows, args.reports_dir / "tables" / "metrics_by_experiment.csv")
    write_markdown_table(
        paper_rows,
        args.reports_dir / "tables" / "paper_multimodal_308.md",
        columns=PAPER_COLUMNS,
    )
    write_csv_table(
        paper_rows,
        args.reports_dir / "tables" / "paper_multimodal_308.csv",
        columns=PAPER_COLUMNS,
    )
    write_placeholder_plots_note(args.reports_dir / "figures" / "README.md")
    print(f"Report artifacts saved under {args.reports_dir}")


if __name__ == "__main__":
    main()
