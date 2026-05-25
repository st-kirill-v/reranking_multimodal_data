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
from src.mmrag.indexing import FaissPageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FAISS vector and metadata alignment.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-reencode", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/audit_index_alignment.json"))
    return parser.parse_args()


def load_record_image(record) -> Image.Image:
    with Image.open(record.path) as img:
        image = img.convert("RGB")
        if record.crop_box:
            image = image.crop(record.crop_box)
        return image.copy()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    page_index = FaissPageIndex(config).load()
    rng = random.Random(args.seed)
    sample_indices = rng.sample(
        range(page_index.index.ntotal), min(args.samples, page_index.index.ntotal)
    )

    embedder = None
    if not args.no_reencode:
        from src.mmrag.embeddings import Qwen3PageEmbedder

        embedder = Qwen3PageEmbedder(config.embedder)

    rows = []
    for idx in sample_indices:
        record = page_index.records[idx]
        stored = np.empty((1, page_index.index.d), dtype=np.float32)
        page_index.index.reconstruct(idx, stored[0])
        row = {
            "index": idx,
            "folder": record.folder,
            "page": record.page,
            "path": str(record.path),
            "tile_id": record.tile_id,
            "crop_box": list(record.crop_box) if record.crop_box else None,
            "path_exists": record.path.exists(),
            "stored_norm": float(np.linalg.norm(stored[0])),
            "fresh_similarity": None,
        }
        if embedder is not None and record.path.exists():
            fresh = embedder.encode_images([load_record_image(record)], batch_size=1)
            row["fresh_similarity"] = float(np.dot(stored[0], fresh[0]))
        rows.append(row)
        print(
            f"idx={idx} folder={record.folder} page={record.page} tile={record.tile_id} "
            f"exists={row['path_exists']} stored_norm={row['stored_norm']:.4f} "
            f"fresh_sim={row['fresh_similarity']}"
        )

    sims = [row["fresh_similarity"] for row in rows if row["fresh_similarity"] is not None]
    summary = {
        "samples": len(rows),
        "missing_paths": sum(1 for row in rows if not row["path_exists"]),
        "fresh_similarity_min": min(sims) if sims else None,
        "fresh_similarity_mean": float(np.mean(sims)) if sims else None,
        "fresh_similarity_max": max(sims) if sims else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
