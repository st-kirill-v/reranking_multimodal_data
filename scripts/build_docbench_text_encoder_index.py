#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.text_page_retriever import (  # noqa: E402
    collect_text_evidence_records,
    normalize_source_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DocBench dense text-encoder FAISS index.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes/docbench_text_encoder_bge_base_en_v1_5"),
    )
    parser.add_argument("--model-id", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--source-fields",
        nargs="*",
        default=["ocr", "page_text", "caption", "table_text"],
        help="Text evidence fields to concatenate per page.",
    )
    parser.add_argument("--test-query", default="")
    return parser.parse_args()


def safe_preview(text: str, max_chars: int = 240) -> str:
    preview = re.sub(r"\s+", " ", text or "").strip()[:max_chars]
    encoding = sys.stdout.encoding or "utf-8"
    return preview.encode(encoding, errors="replace").decode(encoding, errors="replace")


def iter_doc_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def build_corpus(
    dataset_root: Path, source_fields: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    folders = iter_doc_folders(dataset_root)
    print(f"[TextEncoderIndex] Dataset root: {dataset_root}")
    print(f"[TextEncoderIndex] Found document folders: {len(folders)}")
    corpus = collect_text_evidence_records(dataset_root, source_fields=source_fields)
    counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    for record in corpus:
        counts[str(record["doc_id"])] = counts.get(str(record["doc_id"]), 0) + 1
        for field in record.get("available_source_fields") or []:
            available_counts[field] = available_counts.get(field, 0) + 1
    for idx, doc_folder in enumerate(folders, start=1):
        indexed = counts.get(doc_folder.name, 0)
        if indexed == 0:
            warnings.append(f"No text evidence records indexed for doc {doc_folder.name}")
        print(f"[TextEncoderIndex] [{idx}/{len(folders)}] doc={doc_folder.name} pages={indexed}")
    print(f"[TextEncoderIndex] Available evidence counts: {available_counts}")
    return corpus, warnings


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

    source_fields = normalize_source_fields(args.source_fields)
    corpus, warnings = build_corpus(args.dataset_root, source_fields)
    texts = [record["text"] for record in corpus]
    if not texts:
        raise RuntimeError("No page text records found; cannot build text encoder index.")

    from sentence_transformers import SentenceTransformer

    print(f"[TextEncoderIndex] Loading encoder: {args.model_id} on {args.device}")
    model = SentenceTransformer(args.model_id, device=args.device, trust_remote_code=True)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.model_max_length = args.max_length

    print(
        f"[TextEncoderIndex] Encoding {len(texts)} pages "
        f"batch_size={args.batch_size} max_length={args.max_length} normalize={normalize}"
    )
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "pages.faiss"))
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    metadata = {
        "dataset": "docbench",
        "dataset_root": args.dataset_root.as_posix(),
        "num_pages_indexed": len(corpus),
        "model_id": args.model_id,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "requested_source_fields": source_fields,
        "available_source_field_counts": {
            field: sum(
                1 for record in corpus if field in (record.get("available_source_fields") or [])
            )
            for field in source_fields
        },
        "normalize_embeddings": normalize,
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_index": "IndexFlatIP",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[TextEncoderIndex] Saved: {output_dir}")
    print(f"[TextEncoderIndex] Pages indexed: {len(corpus)}")
    print(f"[TextEncoderIndex] Warnings: {len(warnings)}")

    if args.test_query:
        query_embedding = model.encode(
            [args.test_query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, indices = index.search(query_embedding, 5)
        print(f"[TextEncoderIndex] Test query: {args.test_query}")
        for rank, (score, corpus_idx) in enumerate(
            zip(scores[0], indices[0], strict=False), start=1
        ):
            row = corpus[int(corpus_idx)]
            print(
                f"[TextEncoderIndex] rank={rank} doc_id={row['doc_id']} "
                f"page={row['page']} score={float(score):.4f}"
            )
            print(f"[TextEncoderIndex] text={safe_preview(row['text'])}")


if __name__ == "__main__":
    main()
