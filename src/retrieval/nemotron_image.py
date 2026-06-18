from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.mmrag.schema import RetrievalCandidate


class NemotronImageRetriever:
    """First-stage visual retriever based on Nemotron VL bi-encoder embeddings."""

    def __init__(
        self,
        *,
        index_dir: Path,
        index_name: str,
        model_id: str,
        device: str = "cuda",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.model_id = model_id
        self.device = device
        self.index_path = self._resolve_index_path()
        self.metadata_path = self._resolve_metadata_path()
        self.index = self._load_index()
        self.metadata = self._load_metadata()
        self.model, self.processor = self._load_model_and_processor()

    def _resolve_index_path(self) -> Path:
        candidates = [
            self.index_dir / f"pages_{self.index_name}.index",
            self.index_dir / f"{self.index_name}.index",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Missing Nemotron image FAISS index. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _resolve_metadata_path(self) -> Path:
        candidates = [
            self.index_dir / f"metadata_{self.index_name}.json",
            self.index_dir / f"{self.index_name}_metadata.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Missing Nemotron image metadata. Tried: " + ", ".join(str(path) for path in candidates)
        )

    def _load_index(self) -> Any:
        import faiss

        return faiss.read_index(str(self.index_path))

    def _load_metadata(self) -> list[dict[str, Any]]:
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, list):
            raise ValueError(f"Nemotron metadata must be a list: {self.metadata_path}")
        if self.index.ntotal != len(metadata):
            raise ValueError(
                f"Nemotron index/metadata size mismatch: index={self.index.ntotal}, "
                f"metadata={len(metadata)}"
            )
        return metadata

    def _load_model_and_processor(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        if hasattr(model, "processor"):
            model.processor = processor
        return model, processor

    def _encode_query(self, query: str) -> np.ndarray:
        import torch

        with torch.no_grad():
            embedding = self.model.encode_queries([query])
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().float().cpu().numpy()
        embedding = np.asarray(embedding, dtype=np.float32)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norms, 1e-12)
        return embedding

    def search(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        query_embedding = self._encode_query(query)
        scores, indices = self.index.search(query_embedding, top_k)
        candidates: list[RetrievalCandidate] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if int(idx) < 0:
                continue
            record = self.metadata[int(idx)]
            path = Path(record.get("path") or record.get("image_path") or "")
            if path and not path.is_absolute():
                path = Path.cwd() / path
            candidates.append(
                RetrievalCandidate(
                    folder=str(record.get("folder", record.get("doc_id", ""))),
                    page=int(record.get("page", record.get("page_number", 0))),
                    path=path,
                    score=float(score),
                    rank=rank,
                    index=int(record.get("index", idx)),
                    source="nemotron_image",
                )
            )
        return candidates
