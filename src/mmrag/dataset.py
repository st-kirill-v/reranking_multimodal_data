from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from src.mmrag.schema import PageRecord


def parse_page_number(path: Path) -> int | None:
    match = re.fullmatch(r"(?:page_)?(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def iter_docbench_pages(data_dir: Path) -> Iterable[PageRecord]:
    index = 0
    for folder in sorted(data_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        pages_dir = folder / "extracted" / "pages"
        if not pages_dir.exists():
            continue
        for image_path in sorted(pages_dir.glob("*.png"), key=lambda p: parse_page_number(p) or 0):
            page = parse_page_number(image_path)
            if page is None:
                continue
            yield PageRecord(folder=folder.name, page=page, path=image_path.resolve(), index=index)
            index += 1


def load_page_records(path: Path) -> list[PageRecord]:
    with path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [PageRecord.from_json(row) for row in rows]


def save_page_records(path: Path, records: list[PageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([record.to_json() for record in records], fh, indent=2, ensure_ascii=False)


def load_docbench_questions(data_dir: Path, question_types: set[str] | None = None) -> list[dict]:
    questions: list[dict] = []
    for jsonl_file in sorted(data_dir.glob("*/*_qa.jsonl")):
        folder = jsonl_file.parent.name
        if not folder.isdigit():
            continue
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if question_types and row.get("type") not in question_types:
                    continue
                questions.append(
                    {
                        "folder": folder,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "type": row.get("type", ""),
                    }
                )
    return questions


def load_document_domains(data_dir: Path) -> dict[str, str]:
    domains: dict[str, str] = {}
    for folder in data_dir.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        report_path = folder / "extracted" / "doc_report.json"
        if not report_path.exists():
            continue
        with report_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
        domains[folder.name] = report.get("domain", "unknown")
    return domains
