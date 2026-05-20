from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.indexing import FaissPageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forensic checks for an existing FAISS page index."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3_docapi_v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hub-index", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path, default=Path("data/audit_index_forensics.json"))
    return parser.parse_args()


def load_image(record) -> Image.Image:
    with Image.open(record.path) as img:
        image = img.convert("RGB")
        if record.crop_box:
            image = image.crop(record.crop_box)
        return image.copy()


def reconstruct(index: faiss.Index, idx: int) -> np.ndarray:
    vector = np.empty((index.d,), dtype=np.float32)
    index.reconstruct(idx, vector)
    return vector


def search_vector(
    index: faiss.Index, vector: np.ndarray, top_k: int = 5
) -> tuple[list[int], list[float]]:
    scores, indices = index.search(vector.reshape(1, -1).astype(np.float32), top_k)
    return [int(idx) for idx in indices[0] if idx >= 0], [float(score) for score in scores[0]]


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
    if 0 <= args.hub_index < page_index.index.ntotal and args.hub_index not in sample_indices:
        sample_indices.append(args.hub_index)

    rows = []
    internal_hits_at_1 = 0
    fresh_hits_at_1 = 0
    for idx in sample_indices:
        record = page_index.records[idx]
        stored = reconstruct(page_index.index, idx)
        stored_top, stored_scores = search_vector(page_index.index, stored, top_k=5)
        internal_rank = stored_top.index(idx) + 1 if idx in stored_top else None
        internal_hits_at_1 += 1 if internal_rank == 1 else 0

        fresh = embedder.encode_images([load_image(record)], batch_size=1)[0]
        fresh_top, fresh_scores = search_vector(page_index.index, fresh, top_k=5)
        fresh_rank = fresh_top.index(idx) + 1 if idx in fresh_top else None
        fresh_hits_at_1 += 1 if fresh_rank == 1 else 0

        row = {
            "index": idx,
            "folder": record.folder,
            "page": record.page,
            "path": str(record.path),
            "stored_norm": float(np.linalg.norm(stored)),
            "fresh_norm": float(np.linalg.norm(fresh)),
            "stored_vs_fresh": float(np.dot(stored, fresh)),
            "stored_internal_rank": internal_rank,
            "stored_internal_top_indices": stored_top,
            "stored_internal_top_scores": stored_scores,
            "fresh_self_rank": fresh_rank,
            "fresh_top_indices": fresh_top,
            "fresh_top_scores": fresh_scores,
        }
        rows.append(row)
        print(
            f"idx={idx} folder={record.folder} page={record.page} "
            f"stored_rank={internal_rank} fresh_rank={fresh_rank} "
            f"stored_vs_fresh={row['stored_vs_fresh']:.6f} "
            f"fresh_top1={fresh_top[0] if fresh_top else None}"
        )

    hub = None
    if 0 <= args.hub_index < page_index.index.ntotal:
        hub_record = page_index.records[args.hub_index]
        hub_vector = reconstruct(page_index.index, args.hub_index)
        hub_fresh = embedder.encode_images([load_image(hub_record)], batch_size=1)[0]
        hub_top, hub_scores = search_vector(page_index.index, hub_vector, top_k=10)
        hub = {
            "index": args.hub_index,
            "folder": hub_record.folder,
            "page": hub_record.page,
            "path": str(hub_record.path),
            "stored_norm": float(np.linalg.norm(hub_vector)),
            "fresh_norm": float(np.linalg.norm(hub_fresh)),
            "stored_vs_fresh": float(np.dot(hub_vector, hub_fresh)),
            "stored_search_top_indices": hub_top,
            "stored_search_top_scores": hub_scores,
        }

    total = max(len(sample_indices), 1)
    summary = {
        "index_name": args.index_name,
        "index_vectors": page_index.index.ntotal,
        "index_dim": page_index.index.d,
        "samples": len(sample_indices),
        "encoding_api": args.encoding_api,
        "internal_stored_self_recall_at_1": internal_hits_at_1 / total,
        "fresh_self_recall_at_1": fresh_hits_at_1 / total,
        "stored_vs_fresh_mean": float(np.mean([row["stored_vs_fresh"] for row in rows])),
        "hub": hub,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
