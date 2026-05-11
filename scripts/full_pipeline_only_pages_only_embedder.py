"""Compatibility wrapper for the clean multimodal page-RAG pipeline."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import PipelineConfig

_PIPELINE = None


def normalize_answer(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_answer(raw_output: str) -> str:
    if not raw_output:
        return "NOT FOUND"
    text = re.sub(r"^assistant\s*\n", "", raw_output, flags=re.IGNORECASE).strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text or "NOT FOUND"


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from src.mmrag.pipeline import PageRAGPipeline

        _PIPELINE = PageRAGPipeline(PipelineConfig())
    return _PIPELINE


def full_pipeline_only_pages_only_embedder(query: str, timeout_seconds: int = 180):
    del timeout_seconds
    start = time.time()
    result = get_pipeline().answer(query)
    answer = parse_answer(result.get("answer", "NOT FOUND"))
    return answer, normalize_answer(answer), time.time() - start
