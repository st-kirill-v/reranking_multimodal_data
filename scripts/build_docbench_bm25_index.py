#!/usr/bin/env python
"""Build a BM25 index over DocBench extracted page text.

This script intentionally uses only extracted page text files and does not read
answers, evidence, gold pages, OCR artifacts, or multimodal pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bm25s


PAGE_TEXT_FILENAMES = ("page_text.json", "pages_text.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DocBench BM25 index from page text.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets/docbench"),
        help="Path to DocBench dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes/docbench_bm25"),
        help="Directory where corpus, metadata, and BM25 index will be saved.",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Optional query for a top-5 smoke search after indexing.",
    )
    return parser.parse_args()


def numeric_doc_folders(dataset_root: Path, warnings: list[str]) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    folders: list[Path] = []
    for path in dataset_root.iterdir():
        if not path.is_dir():
            continue
        if path.name.isdigit():
            folders.append(path)
        else:
            warnings.append(f"Skipped non-numeric dataset folder: {path}")
    return sorted(folders, key=lambda p: int(p.name))


def find_page_text_path(doc_folder: Path) -> Path | None:
    extracted = doc_folder / "extracted"
    for filename in PAGE_TEXT_FILENAMES:
        candidate = extracted / filename
        if candidate.exists():
            return candidate
    return None


def load_page_text_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("pages", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Expected a JSON list of page records or a dict with pages/records/data list")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview_text(text: str, max_chars: int = 300) -> str:
    preview = " ".join(text.split())[:max_chars]
    encoding = sys.stdout.encoding or "utf-8"
    return preview.encode(encoding, errors="replace").decode(encoding, errors="replace")


def build_corpus(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: list[str] = []
    doc_folders = numeric_doc_folders(dataset_root, warnings)

    print(f"[BM25] Dataset root: {dataset_root}")
    print(f"[BM25] Found document folders: {len(doc_folders)}")

    corpus: list[dict[str, Any]] = []
    folders_with_page_text = 0
    folders_missing_page_text = 0
    empty_pages_count = 0
    text_length_sum = 0

    last_doc_id = doc_folders[-1].name if doc_folders else "0"
    for index, doc_folder in enumerate(doc_folders, start=1):
        doc_id = doc_folder.name
        print(f"[BM25] Processing doc {doc_id}/{last_doc_id} ...")

        page_text_path = find_page_text_path(doc_folder)
        if page_text_path is None:
            folders_missing_page_text += 1
            warning = f"Missing extracted page text for doc {doc_id}"
            warnings.append(warning)
            print(f"[BM25] Warning: {warning}")
            continue

        try:
            page_records = load_page_text_records(page_text_path)
        except Exception as exc:  # noqa: BLE001 - keep indexing robust over imperfect data dumps.
            folders_missing_page_text += 1
            warning = f"Could not read {page_text_path}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            print(f"[BM25] Warning: {warning}")
            continue

        folders_with_page_text += 1
        indexed_for_doc = 0
        empty_for_doc = 0
        source_path = page_text_path.as_posix()

        for page_record in page_records:
            page = page_record.get("page")
            text = str(page_record.get("text") or "").strip()

            if not text:
                empty_pages_count += 1
                empty_for_doc += 1
                continue

            try:
                page_number = int(page)
            except (TypeError, ValueError):
                warning = f"Skipped doc {doc_id} page with invalid page value: {page!r}"
                warnings.append(warning)
                print(f"[BM25] Warning: {warning}")
                continue

            corpus.append(
                {
                    "doc_id": doc_id,
                    "page": page_number,
                    "text": text,
                    "source_path": source_path,
                }
            )
            indexed_for_doc += 1
            text_length_sum += len(text)

        print(
            f"[BM25] Indexed doc {doc_id}: {indexed_for_doc} pages, " f"{empty_for_doc} empty pages"
        )

    avg_text_length_chars = text_length_sum / len(corpus) if corpus else 0.0
    metadata = {
        "dataset": "docbench",
        "dataset_root": dataset_root.as_posix(),
        "num_doc_folders_found": len(doc_folders),
        "num_doc_folders_with_page_text": folders_with_page_text,
        "num_doc_folders_missing_page_text": folders_missing_page_text,
        "num_pages_indexed": len(corpus),
        "num_empty_pages": empty_pages_count,
        "avg_text_length_chars": avg_text_length_chars,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bm25_backend": "bm25s",
        "warnings": warnings,
    }
    return corpus, metadata


def build_bm25_index(corpus: list[dict[str, Any]], index_dir: Path) -> bm25s.BM25:
    texts = [record["text"] for record in corpus]
    print(f"[BM25] Tokenizing {len(texts)} pages ...")
    tokenized_corpus = bm25s.tokenize(texts, show_progress=True)

    print("[BM25] Building BM25 index ...")
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus, show_progress=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BM25] Saving BM25 index to: {index_dir}")
    retriever.save(index_dir, corpus=corpus, show_progress=True)
    return retriever


def run_smoke_search(retriever: bm25s.BM25, corpus: list[dict[str, Any]], query: str) -> None:
    print(f"[BM25] Test query: {query}")
    query_tokens = bm25s.tokenize(query, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=5, show_progress=False)

    print("[BM25] Top-5 results:")
    for rank, (record, score) in enumerate(zip(results[0], scores[0], strict=False), start=1):
        print(
            f"[BM25] rank={rank} doc_id={record['doc_id']} page={record['page']} "
            f"score={float(score):.4f}"
        )
        print(f"[BM25] text={preview_text(record['text'])}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    index_dir = output_dir / "bm25_index"

    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, metadata = build_corpus(dataset_root)
    corpus_path = output_dir / "corpus.jsonl"
    metadata_path = output_dir / "metadata.json"

    write_jsonl(corpus_path, corpus)
    retriever = build_bm25_index(corpus, index_dir)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[BM25] Done.")
    print(
        "[BM25] Documents with page_text: "
        f"{metadata['num_doc_folders_with_page_text']} / {metadata['num_doc_folders_found']}"
    )
    print(
        "[BM25] Missing page_text: "
        f"{metadata['num_doc_folders_missing_page_text']} / {metadata['num_doc_folders_found']}"
    )
    print(f"[BM25] Pages indexed: {metadata['num_pages_indexed']}")
    print(f"[BM25] Empty pages skipped: {metadata['num_empty_pages']}")
    print(f"[BM25] Output saved to: {output_dir}")

    if args.test_query:
        run_smoke_search(retriever, corpus, args.test_query)


if __name__ == "__main__":
    main()
