from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.text_page_retriever import TextPageCandidate


class TextEncoderRetriever:
    """FAISS retriever over page-level text embeddings."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.load_latency = 0.0
        self.index: Any = None
        self.corpus: list[dict[str, Any]] = []
        self.model: Any = None
        self._load()

    def _load(self) -> None:
        start = time.time()
        index_path = self.index_dir / "pages.faiss"
        corpus_path = self.index_dir / "corpus.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing text encoder FAISS index: {index_path}")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing text encoder corpus: {corpus_path}")
        self.index = faiss.read_index(str(index_path))
        with corpus_path.open("r", encoding="utf-8") as handle:
            self.corpus = [json.loads(line) for line in handle if line.strip()]
        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                f"Index/corpus size mismatch: index={self.index.ntotal}, corpus={len(self.corpus)}"
            )
        self.load_latency = time.time() - start

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=True,
        )
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.model_max_length = self.max_length

    def _encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        vector = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def search(self, query: str, top_k: int = 30) -> list[TextPageCandidate]:
        if not query.strip():
            return []
        query_vector = self._encode_query(query)
        scores, indices = self.index.search(query_vector, top_k)
        candidates: list[TextPageCandidate] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            record = self.corpus[int(idx)]
            candidates.append(
                TextPageCandidate(
                    doc_id=str(record["doc_id"]),
                    page=int(record["page"]),
                    text=str(record["text"]),
                    score=float(score),
                    rank=rank,
                    source_path=str(record.get("source_path", "")),
                    source="text_encoder",
                    evidence_fields=list(record.get("available_source_fields") or []),
                )
            )
        return candidates

    def stats(self) -> dict[str, Any]:
        return {
            "num_pages": len(self.corpus),
            "index_dir": self.index_dir.as_posix(),
            "model_id": self.model_id,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "load_latency": self.load_latency,
        }
