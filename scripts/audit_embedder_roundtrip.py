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
        description="Encode the same images twice and test in-memory FAISS self-retrieval."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path("data/audit_embedder_roundtrip.json"))
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def summarize_matrix(name: str, matrix: np.ndarray) -> dict:
    norms = np.linalg.norm(matrix, axis=1)
    return {
        f"{name}_shape": list(matrix.shape),
        f"{name}_norm_min": float(norms.min()) if len(norms) else None,
        f"{name}_norm_mean": float(norms.mean()) if len(norms) else None,
        f"{name}_norm_max": float(norms.max()) if len(norms) else None,
    }


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

    first = embedder.encode_images(images, batch_size=args.batch_size)
    second = embedder.encode_images(images, batch_size=args.batch_size)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: first={first.shape}, second={second.shape}")

    pairwise = first @ second.T
    same_scores = np.diag(pairwise)
    offdiag = pairwise.copy()
    np.fill_diagonal(offdiag, -np.inf)
    max_offdiag = np.max(offdiag, axis=1) if len(selected) > 1 else np.full(len(selected), -np.inf)

    index = faiss.IndexFlatIP(first.shape[1])
    index.add(first.astype(np.float32))
    scores, indices = index.search(second.astype(np.float32), min(5, len(selected)))

    rows = []
    hits_at_1 = 0
    hits_at_k = 0
    for i, record in enumerate(selected):
        retrieved = [int(idx) for idx in indices[i] if idx >= 0]
        if retrieved and retrieved[0] == i:
            hits_at_1 += 1
        if i in retrieved:
            hits_at_k += 1
        row = {
            "sample": i,
            "folder": record.folder,
            "page": record.page,
            "path": str(record.path),
            "same_encode_similarity": float(same_scores[i]),
            "max_other_similarity": float(max_offdiag[i]) if np.isfinite(max_offdiag[i]) else None,
            "self_rank": retrieved.index(i) + 1 if i in retrieved else None,
            "top1_sample": retrieved[0] if retrieved else None,
            "top1_score": float(scores[i][0]) if len(scores[i]) else None,
        }
        rows.append(row)
        print(
            f"[{i + 1}/{len(selected)}] folder={record.folder} page={record.page} "
            f"same={row['same_encode_similarity']:.6f} "
            f"max_other={row['max_other_similarity']} "
            f"self_rank={row['self_rank']} top1={row['top1_sample']}"
        )

    total = max(len(selected), 1)
    summary = {
        "samples": len(selected),
        "encoding_api": args.encoding_api,
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "same_similarity_min": float(same_scores.min()) if len(same_scores) else None,
        "same_similarity_mean": float(same_scores.mean()) if len(same_scores) else None,
        "same_similarity_max": float(same_scores.max()) if len(same_scores) else None,
        "in_memory_self_recall_at_1": hits_at_1 / total,
        "in_memory_self_recall_at_k": hits_at_k / total,
        **summarize_matrix("first", first),
        **summarize_matrix("second", second),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
