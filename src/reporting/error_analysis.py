from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def suspected_error_type(row: dict[str, Any]) -> str:
    generated = (row.get("generated") or "").strip()
    if generated == "NOT FOUND":
        return "not_found"
    if row.get("f1", 0.0) >= 0.5:
        return ""
    if row.get("crop_type_mismatch"):
        return (
            "wrong_table" if row.get("question_crop_intent") == "table" else "visual_reading_error"
        )
    if row.get("caption_match") is False and row.get("explicit_reference"):
        return (
            "wrong_table"
            if "table" in str(row.get("explicit_reference"))
            else "visual_reading_error"
        )
    if row.get("numeric_any_match") == 0 and any(
        char.isdigit() for char in str(row.get("expected", ""))
    ):
        return "wrong_row_or_column"
    if generated and generated != "ERROR":
        return "formatting_issue"
    return "wrong_page"


def write_error_cases(rows: list[dict[str, Any]], path: Path, threshold: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "expected_answer",
        "generated_answer",
        "f1",
        "exact_match",
        "document_id",
        "retrieved_pages",
        "reranked_pages",
        "selected_pages",
        "crop_used",
        "crop_path",
        "fallback_used",
        "latency_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            if float(row.get("f1", 0.0)) >= threshold:
                continue
            writer.writerow(
                {
                    "question_id": row.get("question_id") or idx,
                    "question": row.get("question"),
                    "expected_answer": row.get("expected_answer") or row.get("expected"),
                    "generated_answer": row.get("generated_answer") or row.get("generated"),
                    "f1": row.get("f1"),
                    "exact_match": (
                        row.get("exact_match")
                        if row.get("exact_match") is not None
                        else row.get("exact")
                    ),
                    "document_id": row.get("document_id") or row.get("expected_folder"),
                    "retrieved_pages": row.get("retrieved_pages"),
                    "reranked_pages": row.get("reranked_pages"),
                    "selected_pages": row.get("selected_pages") or row.get("pages"),
                    "crop_used": bool(row.get("crop_used") or row.get("crop_path")),
                    "crop_path": row.get("crop_path"),
                    "fallback_used": row.get("fallback_used"),
                    "latency_total": row.get("latency_total") or row.get("latency"),
                }
            )
