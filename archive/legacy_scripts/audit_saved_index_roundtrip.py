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

from src.mmrag.config import EmbedderConfig, ProjectPaths
from src.mmrag.dataset import iter_docbench_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tiny saved FAISS index, reload it, and test self-retrieval."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--tmp-index", type=Path, default=Path("index/_audit_saved_roundtrip.index")
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit_saved_index_roundtrip.json")
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def main() -> None:
    args = parse_args()
    from src.mmrag.embeddings import Qwen3PageEmbedder

    paths = ProjectPaths(data_dir=args.data_dir)
    records = list(iter_docbench_pages(paths.data_dir))
    if not records:
        raise RuntimeError(f"No PNG pages found under {paths.data_dir}")

    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.samples, len(records)))
    images = [load_image(record.path) for record in selected]

    embedder = Qwen3PageEmbedder(
        EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
            batch_size=args.batch_size,
        )
    )
    embeddings = embedder.encode_images(images, batch_size=args.batch_size).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    args.tmp_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.tmp_index))

    loaded = faiss.read_index(str(args.tmp_index))
    reconstructed = np.empty_like(embeddings)
    loaded.reconstruct_n(0, loaded.ntotal, reconstructed)

    reconstruction_sims = np.sum(embeddings * reconstructed, axis=1)
    query_embeddings = embedder.encode_images(images, batch_size=args.batch_size).astype(np.float32)
    scores, indices = loaded.search(query_embeddings, min(5, loaded.ntotal))

    rows = []
    hits_at_1 = 0
    hits_at_k = 0
    for i, record in enumerate(selected):
        retrieved = [int(idx) for idx in indices[i] if idx >= 0]
        rank = retrieved.index(i) + 1 if i in retrieved else None
        hits_at_1 += 1 if rank == 1 else 0
        hits_at_k += 1 if rank is not None else 0
        row = {
            "sample": i,
            "folder": record.folder,
            "page": record.page,
            "path": str(record.path),
            "reconstruct_similarity": float(reconstruction_sims[i]),
            "self_rank": rank,
            "top_indices": retrieved,
            "top_scores": [float(score) for score in scores[i]],
        }
        rows.append(row)
        print(
            f"[{i + 1}/{len(selected)}] folder={record.folder} page={record.page} "
            f"reconstruct_sim={row['reconstruct_similarity']:.6f} "
            f"self_rank={rank} top1={retrieved[0] if retrieved else None}"
        )

    total = max(len(selected), 1)
    summary = {
        "samples": len(selected),
        "encoding_api": args.encoding_api,
        "model_id": args.model_id,
        "tmp_index": str(args.tmp_index),
        "reconstruct_similarity_min": (
            float(reconstruction_sims.min()) if len(reconstruction_sims) else None
        ),
        "reconstruct_similarity_mean": (
            float(reconstruction_sims.mean()) if len(reconstruction_sims) else None
        ),
        "reconstruct_similarity_max": (
            float(reconstruction_sims.max()) if len(reconstruction_sims) else None
        ),
        "saved_self_recall_at_1": hits_at_1 / total,
        "saved_self_recall_at_k": hits_at_k / total,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
