from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean multi-vector visual page index for ColPali/ColQwen models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_clean")
    parser.add_argument("--model-id", default="vidore/colpali-v1.3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--validate-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=173)
    return parser.parse_args()


def parse_page_number(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("page_"):
        stem = stem.removeprefix("page_")
    return int(stem) if stem.isdigit() else None


def iter_page_records(data_dir: Path) -> list[dict]:
    records = []
    for folder in sorted(
        [path for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    ):
        pages_dir = folder / "extracted" / "pages"
        if not pages_dir.exists():
            continue
        for image_path in sorted(
            [path for path in pages_dir.glob("*.png") if parse_page_number(path) is not None],
            key=lambda path: parse_page_number(path) or 0,
        ):
            records.append(
                {
                    "folder": folder.name,
                    "page": parse_page_number(image_path),
                    "path": str(image_path.resolve()),
                    "index": len(records),
                }
            )
    return records


def load_images(records: list[dict]) -> list[Image.Image]:
    images = []
    for record in records:
        with Image.open(record["path"]) as img:
            images.append(img.convert("RGB").copy())
    return images


def load_model_and_processor(model_id: str, device: str) -> tuple[Any, Any, str]:
    import torch

    lower = model_id.lower()
    if "colqwen2.5" in lower or "colqwen2_5" in lower:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        model_cls = ColQwen2_5
        processor_cls = ColQwen2_5_Processor
        family = "colqwen2.5"
    elif "colqwen2" in lower:
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        model_cls = ColQwen2
        processor_cls = ColQwen2Processor
        family = "colqwen2"
    elif "colpali" in lower:
        from colpali_engine.models import ColPali, ColPaliProcessor

        model_cls = ColPali
        processor_cls = ColPaliProcessor
        family = "colpali"
    else:
        raise ValueError(
            "Unsupported model family. Use a ColPali/ColQwen model, e.g. vidore/colpali-v1.3."
        )

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
    ).eval()
    processor = processor_cls.from_pretrained(model_id)
    return model, processor, family


def split_embeddings(output: Any) -> list[Any]:
    import torch

    if isinstance(output, list):
        return [item.detach().cpu().to(torch.float16) for item in output]
    if hasattr(output, "detach"):
        tensor = output.detach().cpu().to(torch.float16)
        if tensor.ndim == 3:
            return [tensor[i] for i in range(tensor.shape[0])]
        if tensor.ndim == 2:
            return [tensor]
    raise TypeError(f"Unsupported embedding output type/shape: {type(output)}")


def encode_images(model: Any, processor: Any, images: list[Image.Image], device: str) -> list[Any]:
    import torch

    batch = processor.process_images(images).to(device)
    with torch.no_grad():
        output = model(**batch)
    return split_embeddings(output)


def save_shard(shard_dir: Path, shard_id: int, embeddings: list[Any]) -> str:
    import torch

    shard_dir.mkdir(parents=True, exist_ok=True)
    filename = f"shard_{shard_id:05d}.pt"
    torch.save(embeddings, shard_dir / filename)
    return filename


def load_stored_embedding(index_dir: Path, record: dict) -> Any:
    import torch

    shard = torch.load(index_dir / "shards" / record["shard"], map_location="cpu")
    return shard[int(record["shard_offset"])]


def mean_aligned_cosine(left: Any, right: Any) -> float | None:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        return None
    left_f = left.float()
    right_f = right.float()
    sims = torch.nn.functional.cosine_similarity(left_f, right_f, dim=-1)
    return float(sims.mean().item())


def validate_index(
    *,
    model: Any,
    processor: Any,
    index_dir: Path,
    records: list[dict],
    samples: int,
    seed: int,
    device: str,
) -> dict:
    rng = random.Random(seed)
    sample_records = rng.sample(records, min(samples, len(records)))
    rows = []
    cosines = []
    shape_matches = 0
    for record in sample_records:
        fresh = encode_images(model, processor, load_images([record]), device)[0]
        stored = load_stored_embedding(index_dir, record)
        cosine = mean_aligned_cosine(stored, fresh)
        if cosine is not None:
            cosines.append(cosine)
            shape_matches += 1
        row = {
            "index": record["index"],
            "folder": record["folder"],
            "page": record["page"],
            "stored_shape": list(stored.shape),
            "fresh_shape": list(fresh.shape),
            "shape_match": cosine is not None,
            "mean_aligned_cosine": cosine,
        }
        rows.append(row)
        print(
            f"[validate] idx={record['index']} folder={record['folder']} page={record['page']} "
            f"shape_match={row['shape_match']} cosine={cosine}"
        )

    return {
        "samples": len(rows),
        "shape_match_rate": shape_matches / max(len(rows), 1),
        "mean_aligned_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "mean_aligned_cosine_min": float(np.min(cosines)) if cosines else None,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    import torch
    from tqdm import tqdm

    args.index_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.index_dir / "shards"
    records = iter_page_records(args.data_dir)
    if args.max_pages > 0:
        records = records[: args.max_pages]
    if not records:
        raise RuntimeError(f"No page PNG files found under {args.data_dir}")

    model, processor, family = load_model_and_processor(args.model_id, args.device)
    start = time.time()

    metadata = []
    pending_embeddings = []
    pending_records = []
    shard_id = 0
    token_counts = []
    dim = None

    def flush() -> None:
        nonlocal shard_id, pending_embeddings, pending_records, dim
        if not pending_embeddings:
            return
        shard_name = save_shard(shard_dir, shard_id, pending_embeddings)
        for offset, (record, embedding) in enumerate(zip(pending_records, pending_embeddings)):
            if dim is None:
                dim = int(embedding.shape[-1])
            token_counts.append(int(embedding.shape[0]))
            metadata.append(
                {
                    **record,
                    "shard": shard_name,
                    "shard_offset": offset,
                    "embedding_shape": list(embedding.shape),
                }
            )
        shard_id += 1
        pending_embeddings = []
        pending_records = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for offset in tqdm(range(0, len(records), args.batch_size), desc="Embedding ColVision pages"):
        batch_records = records[offset : offset + args.batch_size]
        embeddings = encode_images(model, processor, load_images(batch_records), args.device)
        for record, embedding in zip(batch_records, embeddings):
            pending_records.append(record)
            pending_embeddings.append(embedding)
            if len(pending_embeddings) >= args.shard_size:
                flush()
    flush()

    metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
    validation_path = args.index_dir / f"validation_{args.index_name}.json"

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    validation = validate_index(
        model=model,
        processor=processor,
        index_dir=args.index_dir,
        records=metadata,
        samples=args.validate_samples,
        seed=args.seed,
        device=args.device,
    )
    with validation_path.open("w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2, ensure_ascii=False)

    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(args.data_dir.resolve()),
        "index": {
            "name": args.index_name,
            "type": "multi_vector_sharded",
            "pages": len(metadata),
            "shards": shard_id,
            "dim": dim,
            "token_count_min": min(token_counts) if token_counts else None,
            "token_count_mean": float(np.mean(token_counts)) if token_counts else None,
            "token_count_max": max(token_counts) if token_counts else None,
        },
        "embedding": {
            "model_id": args.model_id,
            "family": family,
            "backend": "colpali-engine",
            "encoding_api": "process_images/process_queries + score_multi_vector",
            "dtype": "float16_storage",
        },
        "build": {
            "max_pages": args.max_pages,
            "source_pages": len(records),
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
        },
        "validation": {
            "samples": validation["samples"],
            "shape_match_rate": validation["shape_match_rate"],
            "mean_aligned_cosine_mean": validation["mean_aligned_cosine_mean"],
            "mean_aligned_cosine_min": validation["mean_aligned_cosine_min"],
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Metadata saved: {metadata_path}")
    print(f"Validation saved: {validation_path}")
    print(f"Manifest saved: {manifest_path}")

    if (
        validation["shape_match_rate"] < 1.0
        or (validation["mean_aligned_cosine_mean"] or 0.0) < 0.99
    ):
        raise SystemExit("ColVision index validation failed.")


if __name__ == "__main__":
    main()
