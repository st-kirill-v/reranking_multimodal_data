from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from traceback import format_exception_only

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.indexing import FaissPageIndex


ST_VARIANTS = ["st_legacy_encode", "st_docapi"]
AUTOMODEL_VARIANTS = [
    "auto_mean_empty_text",
    "auto_mean_image_pad",
    "auto_mean_vision_tokens",
    "auto_first_empty_text",
    "auto_last_empty_text",
]
ALL_VARIANTS = ST_VARIANTS + AUTOMODEL_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stored FAISS vectors against multiple Qwen image-encoding variants."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    parser.add_argument("--output", type=Path, default=Path("data/audit_encoder_variants.json"))
    return parser.parse_args()


def load_record_image(record) -> Image.Image:
    with Image.open(record.path) as img:
        image = img.convert("RGB")
        if record.crop_box:
            image = image.crop(record.crop_box)
        return image.copy()


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_st_variant(
    model_id: str, device: str, variant: str, images: list[Image.Image]
) -> np.ndarray:
    from src.mmrag.embeddings import Qwen3PageEmbedder

    encoding_api = "legacy_encode" if variant == "st_legacy_encode" else "docapi"
    embedder = Qwen3PageEmbedder(
        EmbedderConfig(model_id=model_id, device=device, encoding_api=encoding_api, batch_size=1)
    )
    embeddings = embedder.encode_images(images, batch_size=1)
    del embedder
    clear_memory()
    return embeddings


def encode_automodel_variants(
    model_id: str, device: str, variants: list[str], images: list[Image.Image]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from transformers import AutoModel, AutoProcessor

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    text_by_variant = {
        "auto_mean_empty_text": "",
        "auto_first_empty_text": "",
        "auto_last_empty_text": "",
        "auto_mean_image_pad": "<|image_pad|>",
        "auto_mean_vision_tokens": "<|vision_start|><|image_pad|><|vision_end|>",
    }
    outputs_by_variant: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    errors_by_variant: dict[str, str] = {}

    for image in images:
        for text in sorted(set(text_by_variant[v] for v in variants)):
            selected = [v for v in variants if text_by_variant[v] == text]
            try:
                inputs = processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                ).to(device)
                with torch.no_grad():
                    output = model(**inputs)
                    hidden = output.last_hidden_state
                    pooled = {
                        "mean": normalize(hidden.mean(dim=1)),
                        "first": normalize(hidden[:, 0, :]),
                        "last": normalize(hidden[:, -1, :]),
                    }
            except Exception as exc:
                message = "".join(format_exception_only(type(exc), exc)).strip()
                for variant in selected:
                    errors_by_variant.setdefault(variant, message)
                clear_memory()
                continue
            for variant in selected:
                if variant.startswith("auto_mean"):
                    vec = pooled["mean"]
                elif variant.startswith("auto_first"):
                    vec = pooled["first"]
                elif variant.startswith("auto_last"):
                    vec = pooled["last"]
                else:
                    continue
                outputs_by_variant[variant].append(vec.cpu().float().numpy()[0])
            del inputs, output, hidden

    del model, processor
    clear_memory()
    return (
        {
            variant: np.asarray(rows, dtype=np.float32)
            for variant, rows in outputs_by_variant.items()
        },
        errors_by_variant,
    )


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(model_id=args.model_id, device=args.device),
        index=IndexConfig(name=args.index_name),
    )
    page_index = FaissPageIndex(config).load()
    rng = random.Random(args.seed)
    sample_indices = rng.sample(
        range(page_index.index.ntotal), min(args.samples, page_index.index.ntotal)
    )
    records = [page_index.records[idx] for idx in sample_indices]
    images = [load_record_image(record) for record in records]
    stored = np.empty((len(sample_indices), page_index.index.d), dtype=np.float32)
    for row, idx in enumerate(sample_indices):
        page_index.index.reconstruct(idx, stored[row])

    rows = []
    summary = {}
    requested = [variant for variant in args.variants if variant in ALL_VARIANTS]

    for variant in requested:
        if variant in ST_VARIANTS:
            embeddings = encode_st_variant(args.model_id, args.device, variant, images)
            if embeddings.shape[1] != page_index.index.d:
                sims = []
                dim_match = False
            else:
                sims = [float(np.dot(stored[i], embeddings[i])) for i in range(len(sample_indices))]
                dim_match = True
            summary[variant] = {
                "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
                "dim_match": dim_match,
                "similarity_min": min(sims) if sims else None,
                "similarity_mean": float(np.mean(sims)) if sims else None,
                "similarity_max": max(sims) if sims else None,
            }
            print(f"{variant}: {summary[variant]}")

    auto_requested = [variant for variant in requested if variant in AUTOMODEL_VARIANTS]
    if auto_requested:
        auto_embeddings, auto_errors = encode_automodel_variants(
            args.model_id, args.device, auto_requested, images
        )
        for variant, embeddings in auto_embeddings.items():
            if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_indices):
                sims = []
                dim_match = False
            elif embeddings.shape[1] != page_index.index.d:
                sims = []
                dim_match = False
            else:
                sims = [float(np.dot(stored[i], embeddings[i])) for i in range(len(sample_indices))]
                dim_match = True
            summary[variant] = {
                "dim": (
                    int(embeddings.shape[1])
                    if embeddings.ndim == 2 and embeddings.shape[0]
                    else None
                ),
                "dim_match": dim_match,
                "similarity_min": min(sims) if sims else None,
                "similarity_mean": float(np.mean(sims)) if sims else None,
                "similarity_max": max(sims) if sims else None,
                "error": auto_errors.get(variant),
            }
            print(f"{variant}: {summary[variant]}")

    for i, idx in enumerate(sample_indices):
        rows.append(
            {
                "index": idx,
                "folder": records[i].folder,
                "page": records[i].page,
                "path": str(records[i].path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "samples": rows}, fh, indent=2, ensure_ascii=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
