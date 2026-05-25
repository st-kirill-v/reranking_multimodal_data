from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a page index and fail if self-retrieval validation fails."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3_docapi_v2")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validate-samples", type=int, default=30)
    parser.add_argument("--validate-top-k", type=int, default=5)
    parser.add_argument("--validation-output", type=Path, default=None)
    return parser.parse_args()


def load_record_image(record) -> Image.Image:
    with Image.open(record.path) as img:
        image = img.convert("RGB")
        if record.crop_box:
            image = image.crop(record.crop_box)
        return image.copy()


def validate_page_index_self_retrieval(
    config: PipelineConfig, *, samples: int, top_k: int, seed: int = 53
) -> dict:
    from src.mmrag.embeddings import Qwen3PageEmbedder
    from src.mmrag.indexing import FaissPageIndex

    page_index = FaissPageIndex(config).load()
    embedder = Qwen3PageEmbedder(config.embedder)
    rng = random.Random(seed)
    sample_indices = rng.sample(
        range(page_index.index.ntotal), min(samples, page_index.index.ntotal)
    )

    rows = []
    hits_at_1 = 0
    hits_at_k = 0
    for idx in sample_indices:
        record = page_index.records[idx]
        embedding = embedder.encode_images([load_record_image(record)], batch_size=1)
        scores, indices = page_index.index.search(embedding, top_k)
        retrieved = [int(value) for value in indices[0] if value >= 0]
        rank = retrieved.index(idx) + 1 if idx in retrieved else None
        hits_at_1 += 1 if rank == 1 else 0
        hits_at_k += 1 if rank is not None else 0

        stored = np.empty((page_index.index.d,), dtype=np.float32)
        page_index.index.reconstruct(idx, stored)
        rows.append(
            {
                "index": idx,
                "folder": record.folder,
                "page": record.page,
                "self_rank": rank,
                "stored_vs_fresh": float(np.dot(stored, embedding[0])),
                "top_indices": retrieved,
                "top_scores": [float(score) for score in scores[0]],
            }
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
    from src.mmrag.indexing import build_page_index

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    manifest = build_page_index(config)
    print(
        "Built page index: "
        f"{manifest['index']['vectors']} vectors, dim={manifest['index']['dim']}, "
        f"model={manifest['embedding']['model_id']}, "
        f"encoding_api={manifest['embedding'].get('encoding_api')}"
    )

    validation = validate_page_index_self_retrieval(
        config,
        samples=args.validate_samples,
        top_k=args.validate_top_k,
    )
    print(
        json.dumps(
            {k: v for k, v in validation.items() if k != "rows"}, indent=2, ensure_ascii=False
        )
    )

    output = args.validation_output or (
        config.paths.index_dir / f"validation_{args.index_name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2, ensure_ascii=False)
    print(f"Validation saved: {output}")

    if validation["self_recall_at_1"] < 0.99 or validation["self_recall_at_k"] < 0.99:
        raise SystemExit(
            "Index validation failed: fresh page embeddings do not retrieve their own stored vectors."
        )


if __name__ == "__main__":
    main()
