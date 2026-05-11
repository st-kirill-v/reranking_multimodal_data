from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.mmrag.config import EmbedderConfig


class Qwen3PageEmbedder:
    """Single encoder contract for both page images and text queries."""

    def __init__(self, config: EmbedderConfig):
        self.config = config
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else None
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = SentenceTransformer(
            config.model_id,
            trust_remote_code=True,
            device=config.device,
            model_kwargs=model_kwargs,
        )

    def encode_images(
        self, images: Iterable[Image.Image], batch_size: int | None = None
    ) -> np.ndarray:
        image_list = list(images)
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                image_list,
                batch_size=batch_size or self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            embeddings = self.model.encode(
                image_list,
                batch_size=batch_size or self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            embedding = self.model.encode_query(
                query,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            embedding = self.model.encode(
                query,
                prompt=self.config.query_prompt,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray([embedding], dtype=np.float32)

    def embedding_dim(self) -> int:
        probe = Image.new("RGB", (224, 224), color="white")
        return int(self.encode_images([probe], batch_size=1).shape[1])

    def manifest(self, dim: int) -> dict:
        return {
            "model_id": self.config.model_id,
            "backend": self.config.backend,
            "encoding_api": "encode_query/encode_document",
            "normalize": self.config.normalize,
            "query_prompt": self.config.query_prompt,
            "dim": dim,
        }
