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
    parser = argparse.ArgumentParser(
        description="Check whether FAISS vectors are shifted vs metadata order."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoding-api", choices=["legacy_encode", "docapi"], default="legacy_encode"
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path, default=Path("data/audit_metadata_shift.json"))
    return parser.parse_args()


def load_image(record) -> Image.Image:
    with Image.open(record.path) as img:
        image = img.convert("RGB")
        if record.crop_box:
            image = image.crop(record.crop_box)
        return image.copy()


def main() -> None:
    args = parse_args()
    from src.mmrag.embeddings import Qwen3PageEmbedder

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
            batch_size=1,
        ),
        index=IndexConfig(name=args.index_name),
    )
    page_index = FaissPageIndex(config).load()
    embedder = Qwen3PageEmbedder(config.embedder)
    rng = random.Random(args.seed)
    sample_indices = rng.sample(
        range(page_index.index.ntotal), min(args.samples, page_index.index.ntotal)
    )

    rows = []
    for idx in sample_indices:
        stored = np.empty((1, page_index.index.d), dtype=np.float32)
        page_index.index.reconstruct(idx, stored[0])
        start = max(0, idx - args.window)
        end = min(page_index.index.ntotal, idx + args.window + 1)
        candidate_indices = list(range(start, end))
        images = [load_image(page_index.records[j]) for j in candidate_indices]
        fresh = embedder.encode_images(images, batch_size=1)
        sims = fresh @ stored[0]
        best_pos = int(np.argmax(sims))
        best_idx = candidate_indices[best_pos]
        own_pos = candidate_indices.index(idx)
        row = {
            "index": idx,
            "folder": page_index.records[idx].folder,
            "page": page_index.records[idx].page,
            "best_nearby_index": best_idx,
            "best_nearby_folder": page_index.records[best_idx].folder,
            "best_nearby_page": page_index.records[best_idx].page,
            "best_similarity": float(sims[best_pos]),
            "own_similarity": float(sims[own_pos]),
            "offset": best_idx - idx,
        }
        rows.append(row)
        print(
            f"idx={idx} own={row['own_similarity']:.5f} best={best_idx} "
            f"offset={row['offset']} best_sim={row['best_similarity']:.5f}"
        )

    offsets = [row["offset"] for row in rows]
    summary = {
        "samples": len(rows),
        "window": args.window,
        "encoding_api": args.encoding_api,
        "exact_offset_zero": sum(1 for offset in offsets if offset == 0) / max(len(offsets), 1),
        "offsets": offsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
