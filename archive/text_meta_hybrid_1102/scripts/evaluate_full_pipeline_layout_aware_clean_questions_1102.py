from __future__ import annotations

import sys

from evaluate_full_pipeline_layout_aware_clean import main


def ensure_default_args() -> None:
    if "--types" not in sys.argv:
        sys.argv.extend(["--types", "all"])
    if "--output" not in sys.argv:
        sys.argv.extend(
            [
                "--output",
                "data/eval_full_pipeline_colpali_nemotron_layout_aware_v2_1102.json",
            ]
        )


if __name__ == "__main__":
    ensure_default_args()
    main()
