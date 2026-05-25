from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a candidate-input subset for the VLM evaluator using questions whose "
            "previous eval F1 is below a threshold."
        )
    )
    parser.add_argument("--eval-json", type=Path, required=True, help="Previous VLM eval JSON.")
    parser.add_argument(
        "--source-candidates",
        type=Path,
        default=None,
        help=(
            "Original candidates JSON used by evaluate_vlm_from_page_candidates_clean.py. "
            "If omitted, the subset is rebuilt from the already selected pages in --eval-json."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vlm_f1_lt_0_5_candidates_subset.json"),
    )
    return parser.parse_args()


def normalize_question(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results") or payload.get("rows") or []
    return payload, rows


def main() -> None:
    args = parse_args()
    eval_payload, eval_rows = load_rows(args.eval_json)

    bad_questions = {
        normalize_question(row.get("question", ""))
        for row in eval_rows
        if float(row.get("f1", 0.0)) < args.threshold
    }

    if args.source_candidates:
        _source_payload, source_rows = load_rows(args.source_candidates)
        subset_rows = [
            row
            for row in source_rows
            if normalize_question(row.get("question", "")) in bad_questions
        ]
    else:
        subset_rows = []
        for row in eval_rows:
            if normalize_question(row.get("question", "")) not in bad_questions:
                continue
            pages = row.get("pages") or []
            subset_rows.append(
                {
                    "question": row.get("question", ""),
                    "answer": row.get("expected", ""),
                    "type": row.get("type", ""),
                    "expected_folder": row.get("expected_folder"),
                    "oracle_pages": row.get("oracle_pages", []),
                    "top10_reranked": pages,
                    "reranked": pages,
                    "previous_generated": row.get("generated", ""),
                    "previous_f1": row.get("f1"),
                    "previous_exact": row.get("exact"),
                }
            )

    missing = sorted(
        bad_questions - {normalize_question(row.get("question", "")) for row in subset_rows}
    )
    output = {
        "summary": {
            "source_eval": str(args.eval_json),
            "source_candidates": str(args.source_candidates) if args.source_candidates else None,
            "rebuilt_from_eval_pages": args.source_candidates is None,
            "threshold": args.threshold,
            "bad_questions": len(bad_questions),
            "subset_rows": len(subset_rows),
            "missing_questions": len(missing),
            "source_summary": eval_payload.get("summary"),
        },
        "rows": subset_rows,
        "missing": missing,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(
        "Run subset with: "
        "python scripts/evaluate_vlm_from_page_candidates_clean.py "
        f"--input {args.output} --mode reranked --output data/eval_vlm_f1_lt_0_5_subset.json"
    )


if __name__ == "__main__":
    main()
