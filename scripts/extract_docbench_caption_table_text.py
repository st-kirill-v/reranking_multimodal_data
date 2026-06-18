#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.text_page_retriever import find_page_text_path, load_page_text_rows  # noqa: E402


CAPTION_RE = re.compile(
    r"^\s*(?P<label>Table|Figure|Fig\.?|Chart|Graph)\s+"
    r"(?P<number>\d+[A-Za-z]?)\s*[:.\-]?\s*(?P<body>.*)$",
    re.IGNORECASE,
)
TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s+\d+[A-Za-z]?\s*[:.\-]?\s*.*$",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Z])\s+[A-Z][A-Za-z ,:/&()\-]{2,}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic caption and table_text sidecar files from DocBench pages_text.json."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--max-caption-continuation-lines", type=int, default=3)
    parser.add_argument("--table-lines-before", type=int, default=28)
    parser.add_argument("--table-lines-after", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root for sidecar output. By default writes to each "
            "doc_id/extracted directory. With this option writes to "
            "output_root/doc_id/extracted."
        ),
    )
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line or "").strip()


def page_lines(text: str) -> list[str]:
    return [clean_line(line) for line in (text or "").splitlines() if clean_line(line)]


def caption_reference(line: str) -> str:
    match = CAPTION_RE.match(line)
    if not match:
        return ""
    label = match.group("label").lower().replace(".", "")
    if label == "fig":
        label = "figure"
    return f"{label} {match.group('number').lower()}"


def should_continue_caption(line: str) -> bool:
    if CAPTION_RE.match(line):
        return False
    if SECTION_RE.match(line):
        return False
    if len(line) <= 2:
        return False
    return True


def extract_captions(text: str, max_continuation_lines: int) -> list[str]:
    lines = page_lines(text)
    captions: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if not CAPTION_RE.match(line):
            continue
        parts = [line]
        for offset in range(1, max_continuation_lines + 1):
            next_idx = idx + offset
            if next_idx >= len(lines):
                break
            next_line = lines[next_idx]
            if not should_continue_caption(next_line):
                break
            parts.append(next_line)
        caption = clean_line(" ".join(parts))
        key = caption.lower()
        if caption and key not in seen:
            seen.add(key)
            captions.append(caption)
    return captions


def trim_table_block(lines: list[str], caption_idx: int, before: int, after: int) -> str:
    start = max(0, caption_idx - before)
    end = min(len(lines), caption_idx + after + 1)
    block_lines = lines[start:end]
    return "\n".join(block_lines).strip()


def extract_table_blocks(text: str, before: int, after: int) -> list[str]:
    lines = page_lines(text)
    blocks: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if not TABLE_CAPTION_RE.match(line):
            continue
        block = trim_table_block(lines, idx, before, after)
        key = re.sub(r"\s+", " ", block.lower())
        if block and key not in seen:
            seen.add(key)
            blocks.append(block)
    return blocks


def numeric_doc_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def write_json(path: Path, rows: list[dict[str, Any]], *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    folders = numeric_doc_folders(args.dataset_root)
    if args.limit_docs > 0:
        folders = folders[: args.limit_docs]

    total_pages = 0
    total_caption_pages = 0
    total_table_pages = 0
    total_captions = 0
    total_table_blocks = 0
    warnings: list[str] = []

    print(f"[TextEvidence] Dataset root: {args.dataset_root}")
    print(f"[TextEvidence] Document folders: {len(folders)}")
    print(
        "[TextEvidence] Extracting captions/table_text from pages_text.json "
        f"dry_run={args.dry_run}"
    )

    for doc_idx, doc_folder in enumerate(folders, start=1):
        extracted_dir = doc_folder / "extracted"
        output_extracted_dir = (
            args.output_root / doc_folder.name / "extracted"
            if args.output_root is not None
            else extracted_dir
        )
        page_text_path = find_page_text_path(doc_folder)
        if page_text_path is None:
            warnings.append(f"Missing pages_text.json for doc {doc_folder.name}")
            continue
        try:
            page_rows = load_page_text_rows(page_text_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read {page_text_path}: {type(exc).__name__}: {exc}")
            continue

        caption_rows: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []
        for page_row in page_rows:
            try:
                page = int(page_row.get("page"))
            except (TypeError, ValueError):
                warnings.append(f"Invalid page in {page_text_path}: {page_row.get('page')!r}")
                continue
            text = str(page_row.get("text") or "")
            total_pages += 1
            captions = extract_captions(text, args.max_caption_continuation_lines)
            table_blocks = extract_table_blocks(
                text, args.table_lines_before, args.table_lines_after
            )
            if captions:
                total_caption_pages += 1
                total_captions += len(captions)
                caption_rows.append(
                    {
                        "page": page,
                        "caption": captions,
                        "caption_references": [caption_reference(item) for item in captions],
                    }
                )
            if table_blocks:
                total_table_pages += 1
                total_table_blocks += len(table_blocks)
                table_rows.append(
                    {
                        "page": page,
                        "table_text": table_blocks,
                    }
                )

        write_json(output_extracted_dir / "page_captions.json", caption_rows, dry_run=args.dry_run)
        write_json(
            output_extracted_dir / "pages_tables_text.json", table_rows, dry_run=args.dry_run
        )
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": page_text_path.as_posix(),
            "caption_file": (output_extracted_dir / "page_captions.json").as_posix(),
            "table_text_file": (output_extracted_dir / "pages_tables_text.json").as_posix(),
            "pages_total": len(page_rows),
            "pages_with_captions": len(caption_rows),
            "pages_with_table_text": len(table_rows),
            "captions": sum(len(row.get("caption", [])) for row in caption_rows),
            "table_blocks": sum(len(row.get("table_text", [])) for row in table_rows),
            "method": "regex_caption_and_caption_centered_table_blocks_from_pages_text",
        }
        if not args.dry_run:
            output_extracted_dir.mkdir(parents=True, exist_ok=True)
            (output_extracted_dir / "text_evidence_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(
            f"[TextEvidence] [{doc_idx}/{len(folders)}] doc={doc_folder.name} "
            f"caption_pages={len(caption_rows)} table_pages={len(table_rows)}"
        )

    summary = {
        "dataset_root": args.dataset_root.as_posix(),
        "documents_processed": len(folders),
        "pages_processed": total_pages,
        "pages_with_captions": total_caption_pages,
        "pages_with_table_text": total_table_pages,
        "captions": total_captions,
        "table_blocks": total_table_blocks,
        "warnings": warnings,
        "dry_run": args.dry_run,
    }
    print("[TextEvidence] Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
