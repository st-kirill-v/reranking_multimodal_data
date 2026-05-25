from __future__ import annotations

from pathlib import Path


def write_placeholder_plots_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Plot generation is intentionally lightweight in the base environment. "
        "Use reports/tables/*.md or install matplotlib to render bar plots.\n",
        encoding="utf-8",
    )
