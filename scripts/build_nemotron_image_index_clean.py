from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mmrag.dataset import iter_docbench_pages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Nemotron VL image embeddings index.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="nemotron")
    parser.add_argument("--model-id", default="models/nemotron/embed-vl-1b-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model_and_processor(model_id: str, device: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(model, "processor"):
        model.processor = processor
    return model, processor


def encode_images(model, paths: list[Path]) -> np.ndarray:
    import torch

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    with torch.no_grad():
        embeddings = model.encode_documents(images=images)
    embeddings = embeddings.detach().float().cpu().numpy().astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def main() -> None:
    args = parse_args()
    records = list(iter_docbench_pages(args.data_dir))
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No page images found under {args.data_dir}")

    print(f"[Nemotron index] pages: {len(records)}")
    print(f"[Nemotron index] model: {args.model_id}")
    print(f"[Nemotron index] device: {args.device}")
    model, _processor = load_model_and_processor(args.model_id, args.device)

    all_embeddings: list[np.ndarray] = []
    for offset in tqdm(range(0, len(records), args.batch_size), desc="Embedding pages"):
        batch = records[offset : offset + args.batch_size]
        all_embeddings.append(encode_images(model, [record.path for record in batch]))
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrix = np.concatenate(all_embeddings, axis=0).astype("float32")
    import faiss

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_dir / f"pages_{args.index_name}.index"
    metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
    faiss.write_index(index, str(index_path))
    metadata = []
    for idx, record in enumerate(records):
        metadata.append(
            {
                "folder": record.folder,
                "page": record.page,
                "path": str(record.path),
                "index": idx,
                "source": "nemotron_image",
            }
        )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"[Nemotron index] saved index: {index_path}")
    print(f"[Nemotron index] saved metadata: {metadata_path}")
    print(f"[Nemotron index] vectors: {index.ntotal}, dim: {matrix.shape[1]}")


if __name__ == "__main__":
    main()
