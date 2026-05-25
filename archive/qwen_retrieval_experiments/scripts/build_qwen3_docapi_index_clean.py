from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean self-contained Qwen3-VL page index builder with mandatory validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="pages_qwen3_docapi_clean")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--validate-samples", type=int, default=50)
    parser.add_argument("--validate-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def parse_page_number(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("page_"):
        stem = stem.removeprefix("page_")
    return int(stem) if stem.isdigit() else None


def iter_page_records(data_dir: Path) -> list[dict]:
    records = []
    index = 0
    folders = sorted(
        [path for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    for folder in folders:
        pages_dir = folder / "extracted" / "pages"
        if not pages_dir.exists():
            continue
        image_paths = sorted(
            [path for path in pages_dir.glob("*.png") if parse_page_number(path) is not None],
            key=lambda path: parse_page_number(path) or 0,
        )
        for image_path in image_paths:
            records.append(
                {
                    "folder": folder.name,
                    "page": parse_page_number(image_path),
                    "path": str(image_path.resolve()),
                    "index": index,
                }
            )
            index += 1
    return records


def load_images(records: list[dict]) -> list[Image.Image]:
    images = []
    for record in records:
        with Image.open(record["path"]) as img:
            images.append(img.convert("RGB").copy())
    return images


def encode_documents(model: Any, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
    embeddings = model.encode_document(
        images,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('pages_')}.json"


def validate_saved_index(
    *,
    model: Any,
    index: Any,
    records: list[dict],
    samples: int,
    top_k: int,
    batch_size: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    sample_indices = rng.sample(range(index.ntotal), min(samples, index.ntotal))
    hits_at_1 = 0
    hits_at_k = 0
    rows = []

    for idx in sample_indices:
        record = records[idx]
        fresh = encode_documents(model, load_images([record]), batch_size=batch_size)
        scores, indices = index.search(fresh, top_k)
        retrieved = [int(value) for value in indices[0] if value >= 0]
        rank = retrieved.index(idx) + 1 if idx in retrieved else None
        hits_at_1 += 1 if rank == 1 else 0
        hits_at_k += 1 if rank is not None else 0

        stored = np.empty((index.d,), dtype=np.float32)
        index.reconstruct(idx, stored)
        row = {
            "index": idx,
            "folder": record["folder"],
            "page": record["page"],
            "self_rank": rank,
            "stored_vs_fresh": float(np.dot(stored, fresh[0])),
            "top_indices": retrieved,
            "top_scores": [float(score) for score in scores[0]],
        }
        rows.append(row)
        print(
            f"[validate] idx={idx} folder={record['folder']} page={record['page']} "
            f"self_rank={rank} stored_vs_fresh={row['stored_vs_fresh']:.6f} "
            f"top1={retrieved[0] if retrieved else None}"
        )

    total = max(len(rows), 1)
    return {
        "samples": len(rows),
        "top_k": top_k,
        "self_recall_at_1": hits_at_1 / total,
        "self_recall_at_k": hits_at_k / total,
        "stored_vs_fresh_mean": (
            float(np.mean([row["stored_vs_fresh"] for row in rows])) if rows else None
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    args.index_dir.mkdir(parents=True, exist_ok=True)
    records = iter_page_records(args.data_dir)
    if args.max_pages > 0:
        records = records[: args.max_pages]
    if not records:
        raise RuntimeError(f"No page PNG files found under {args.data_dir}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    model_kwargs = {"dtype": dtype} if dtype is not None else {}
    model = SentenceTransformer(
        args.model_id,
        trust_remote_code=True,
        device=args.device,
        model_kwargs=model_kwargs,
    )

    index = None
    start = time.time()
    saved_records = []
    for offset in tqdm(range(0, len(records), args.batch_size), desc="Embedding pages"):
        batch = records[offset : offset + args.batch_size]
        embeddings = encode_documents(model, load_images(batch), batch_size=args.batch_size)
        if index is None:
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        if embeddings.shape[1] != index.d:
            raise ValueError(f"Embedding dim changed from {index.d} to {embeddings.shape[1]}")
        index.add(embeddings)
        saved_records.extend(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if index is None:
        raise RuntimeError("No embeddings were produced")

    index_path = args.index_dir / f"{args.index_name}.index"
    metadata_path = args.index_dir / metadata_filename(args.index_name)
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
    validation_path = args.index_dir / f"validation_{args.index_name}.json"

    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(saved_records, fh, indent=2, ensure_ascii=False)

    loaded_index = faiss.read_index(str(index_path))
    validation = validate_saved_index(
        model=model,
        index=loaded_index,
        records=saved_records,
        samples=args.validate_samples,
        top_k=args.validate_top_k,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    with validation_path.open("w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2, ensure_ascii=False)

    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(args.data_dir.resolve()),
        "index": {
            "name": args.index_name,
            "metric": "ip",
            "vectors": loaded_index.ntotal,
            "dim": loaded_index.d,
        },
        "embedding": {
            "model_id": args.model_id,
            "backend": "sentence-transformers",
            "encoding_api": "docapi",
            "normalize": True,
            "dim": loaded_index.d,
        },
        "validation": {
            "self_recall_at_1": validation["self_recall_at_1"],
            "self_recall_at_k": validation["self_recall_at_k"],
            "stored_vs_fresh_mean": validation["stored_vs_fresh_mean"],
            "samples": validation["samples"],
        },
        "build": {
            "max_pages": args.max_pages,
            "source_pages": len(records),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Index saved: {index_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Validation saved: {validation_path}")
    print(f"Manifest saved: {manifest_path}")

    if validation["self_recall_at_1"] < 0.99 or validation["self_recall_at_k"] < 0.99:
        raise SystemExit("Index validation failed.")


if __name__ == "__main__":
    main()
