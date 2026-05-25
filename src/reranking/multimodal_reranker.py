from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_nemotron_rerank_from_candidates_clean import (  # noqa: E402
    load_nemotron_reranker,
    rerank_candidates,
)


class MultimodalReranker:
    name = "multimodal_reranking"

    def __init__(self, device: str = "cuda", batch_size: int = 1):
        class Args:
            reranker_model_id = "nvidia/llama-nemotron-rerank-vl-1b-v2"
            rerank_max_input_tiles = 6
            rerank_max_length = 2048
            no_thumbnail = False

        self.device = device
        self.batch_size = batch_size
        Args.device = device
        self.model, self.processor = load_nemotron_reranker(Args)

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rerank_candidates(
            query=question,
            candidates=candidates,
            model=self.model,
            processor=self.processor,
            device=self.device,
            batch_size=self.batch_size,
        )
