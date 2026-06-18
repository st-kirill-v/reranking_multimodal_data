from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.tables import (
    PAPER_COLUMNS,
    collect_metric_rows,
    filter_paper_rows,
    write_csv_table,
    write_markdown_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics tables across experiments.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/tables/metrics_by_experiment.md")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("reports/tables/metrics_by_experiment.csv")
    )
    parser.add_argument(
        "--paper-output", type=Path, default=Path("reports/tables/paper_multimodal_308.md")
    )
    parser.add_argument(
        "--paper-csv-output", type=Path, default=Path("reports/tables/paper_multimodal_308.csv")
    )
    args = parser.parse_args()
    rows = collect_metric_rows(args.results_dir, configs_dir=args.configs_dir)
    paper_rows = filter_paper_rows(rows)
    write_markdown_table(rows, args.output)
    write_csv_table(rows, args.csv_output)
    write_markdown_table(paper_rows, args.paper_output, columns=PAPER_COLUMNS)
    write_csv_table(paper_rows, args.paper_csv_output, columns=PAPER_COLUMNS)
    print(f"Saved {args.output}")
    print(f"Saved {args.csv_output}")
    print(f"Saved {args.paper_output}")
    print(f"Saved {args.paper_csv_output}")


if __name__ == "__main__":
    main()
