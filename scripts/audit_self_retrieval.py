from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.indexing import FaissPageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether indexed images retrieve themselves."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("data/audit_self_retrieval.json"))
    return parser.parse_args()


def load_record_image(record) -> Image.Image:
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
    hits_at_1 = 0
    hits_at_k = 0
    for i, idx in enumerate(sample_indices, start=1):
        record = page_index.records[idx]
        embedding = embedder.encode_images([load_record_image(record)], batch_size=1)
        scores, indices = page_index.index.search(embedding, args.top_k)
        retrieved = [int(value) for value in indices[0] if value >= 0]
        rank = retrieved.index(idx) + 1 if idx in retrieved else None
        hits_at_1 += 1 if rank == 1 else 0
        hits_at_k += 1 if rank is not None else 0
        row = {
            "index": idx,
            "folder": record.folder,
            "page": record.page,
            "tile_id": record.tile_id,
            "self_rank": rank,
            "top_indices": retrieved,
            "top_scores": [float(value) for value in scores[0]],
        }
        rows.append(row)
        print(
            f"[{i}/{len(sample_indices)}] idx={idx} folder={record.folder} page={record.page} "
            f"tile={record.tile_id} self_rank={rank} top1={retrieved[0] if retrieved else None} "
            f"score1={row['top_scores'][0] if row['top_scores'] else None}"
        )

    total = max(len(rows), 1)
    summary = {
        "samples": len(rows),
        "top_k": args.top_k,
        "self_recall_at_1": hits_at_1 / total,
        "self_recall_at_k": hits_at_k / total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
