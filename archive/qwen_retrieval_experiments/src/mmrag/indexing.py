from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.mmrag.config import PipelineConfig
from src.mmrag.dataset import iter_docbench_pages, load_page_records, save_page_records
from src.mmrag.diagnostics import VectorStats, compute_vector_stats
from src.mmrag.schema import PageRecord, RetrievalCandidate
from src.mmrag.tiling import iter_page_tiles


class FaissPageIndex:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.index_path = config.paths.index_dir / config.index.index_filename
        self.metadata_path = config.paths.index_dir / config.index.metadata_filename
        self.manifest_path = config.paths.index_dir / config.index.manifest_filename
        self.index: faiss.Index | None = None
        self.records: list[PageRecord] = []
        self.manifest: dict = {}

    def load(self) -> "FaissPageIndex":
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {self.metadata_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.records = load_page_records(self.metadata_path)
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                self.manifest = json.load(fh)
        else:
            self.manifest = {}
        self._validate_loaded()
        return self

    def _validate_loaded(self) -> None:
        assert self.index is not None
        if self.index.ntotal != len(self.records):
            raise ValueError(
                f"Index/metadata mismatch: ntotal={self.index.ntotal}, metadata={len(self.records)}"
            )
        expected_dim = self.manifest.get("embedding", {}).get("dim")
        if expected_dim is not None and int(expected_dim) != self.index.d:
            raise ValueError(f"Manifest dim={expected_dim} but index dim={self.index.d}")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalCandidate]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim != 2 or query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding shape {query_embedding.shape} does not match index dim {self.index.d}"
            )
        scores, indices = self.index.search(query_embedding, top_k)
        candidates: list[RetrievalCandidate] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            record = self.records[int(idx)]
            candidates.append(
                RetrievalCandidate.from_page_record(
                    record, score=float(score), rank=rank, index=int(idx)
                )
            )
        return candidates

    def vectors(self) -> np.ndarray:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        vectors = np.empty((self.index.ntotal, self.index.d), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, vectors)
        return vectors

    def vector_stats(self) -> VectorStats:
        return compute_vector_stats(self.vectors())


def open_page_images(records: Iterable[PageRecord]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        with Image.open(record.path) as img:
            images.append(img.convert("RGB").copy())
    return images


def build_page_index(config: PipelineConfig) -> dict:
    from src.mmrag.embeddings import Qwen3PageEmbedder

    config.paths.index_dir.mkdir(parents=True, exist_ok=True)
    embedder = Qwen3PageEmbedder(config.embedder)
    records = list(iter_docbench_pages(config.paths.data_dir))
    if not records:
        raise RuntimeError(f"No PNG pages found under {config.paths.data_dir}")

    index: faiss.Index | None = None
    saved_records: list[PageRecord] = []
    start = time.time()

    for offset in tqdm(range(0, len(records), config.embedder.batch_size), desc="Embedding pages"):
        batch = records[offset : offset + config.embedder.batch_size]
        try:
            images = open_page_images(batch)
            embeddings = embedder.encode_images(images, batch_size=config.embedder.batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        if index is None:
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        if embeddings.shape[1] != index.d:
            raise ValueError(f"Embedding dim changed from {index.d} to {embeddings.shape[1]}")
        index.add(embeddings)
        saved_records.extend(
            PageRecord(folder=r.folder, page=r.page, path=r.path, index=len(saved_records) + i)
            for i, r in enumerate(batch)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if index is None:
        raise RuntimeError("No embeddings were produced")

    page_index = FaissPageIndex(config)
    faiss.write_index(index, str(page_index.index_path))
    save_page_records(page_index.metadata_path, saved_records)

    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(config.paths.data_dir),
        "index": {
            "name": config.index.name,
            "metric": config.index.metric,
            "vectors": index.ntotal,
            "dim": index.d,
        },
        "embedding": embedder.manifest(index.d),
    }
    with page_index.manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


def validate_page_index_self_retrieval(
    config: PipelineConfig, *, samples: int = 20, top_k: int = 5, seed: int = 53
) -> dict:
    from src.mmrag.embeddings import Qwen3PageEmbedder

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
        images = open_page_images([record])
        embedding = embedder.encode_images(images, batch_size=1)
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


def build_tile_index(config: PipelineConfig, *, tile_mode: str) -> dict:
    from src.mmrag.embeddings import Qwen3PageEmbedder

    config.paths.index_dir.mkdir(parents=True, exist_ok=True)
    embedder = Qwen3PageEmbedder(config.embedder)
    page_records = list(iter_docbench_pages(config.paths.data_dir))
    if not page_records:
        raise RuntimeError(f"No PNG pages found under {config.paths.data_dir}")

    dim = embedder.embedding_dim()
    index = faiss.IndexFlatIP(dim)
    saved_records: list[PageRecord] = []
    start = time.time()

    batch_images: list[Image.Image] = []
    batch_records: list[PageRecord] = []

    def flush_batch() -> None:
        nonlocal batch_images, batch_records
        if not batch_images:
            return
        embeddings = embedder.encode_images(batch_images, batch_size=config.embedder.batch_size)
        if embeddings.shape[1] != dim:
            raise ValueError(f"Embedding dim changed from {dim} to {embeddings.shape[1]}")
        index.add(embeddings)
        base = len(saved_records)
        for i, record in enumerate(batch_records):
            saved_records.append(
                PageRecord(
                    folder=record.folder,
                    page=record.page,
                    path=record.path,
                    index=base + i,
                    tile_id=record.tile_id,
                    crop_box=record.crop_box,
                    source_page_width=record.source_page_width,
                    source_page_height=record.source_page_height,
                )
            )
        batch_images = []
        batch_records = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_tiles = 0
    for record, tile in tqdm(
        iter_page_tiles(page_records, tile_mode), desc=f"Embedding {tile_mode} tiles"
    ):
        batch_records.append(record)
        batch_images.append(tile)
        total_tiles += 1
        if len(batch_images) >= config.embedder.batch_size:
            flush_batch()
    flush_batch()

    page_index = FaissPageIndex(config)
    faiss.write_index(index, str(page_index.index_path))
    save_page_records(page_index.metadata_path, saved_records)

    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(config.paths.data_dir),
        "index": {
            "name": config.index.name,
            "metric": config.index.metric,
            "vectors": index.ntotal,
            "dim": index.d,
        },
        "embedding": embedder.manifest(dim),
        "tiling": {
            "mode": tile_mode,
            "source_pages": len(page_records),
            "tiles": total_tiles,
        },
    }
    with page_index.manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest
