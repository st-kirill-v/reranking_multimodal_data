from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.mmrag.schema import RetrievalCandidate


@dataclass(frozen=True)
class VectorStats:
    count: int
    dim: int
    norm_min: float
    norm_mean: float
    norm_max: float
    component_std_mean: float
    duplicate_rows: int


def compute_vector_stats(vectors: np.ndarray, duplicate_decimals: int = 6) -> VectorStats:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vector matrix, got shape={vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    rounded = np.round(vectors, duplicate_decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    return VectorStats(
        count=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        norm_min=float(np.min(norms)),
        norm_mean=float(np.mean(norms)),
        norm_max=float(np.max(norms)),
        component_std_mean=float(np.mean(np.std(vectors, axis=0))),
        duplicate_rows=int(np.sum(counts[counts > 1] - 1)),
    )


def summarize_scores(candidates: list[RetrievalCandidate]) -> dict[str, float]:
    if not candidates:
        return {}
    scores = np.array([candidate.score for candidate in candidates], dtype=np.float32)
    return {
        "min": float(np.min(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores)),
        "top1_top5_gap": float(scores[0] - scores[min(4, len(scores) - 1)]),
    }


def summarize_domains(
    candidates: list[RetrievalCandidate], domains: dict[str, str]
) -> dict[str, int]:
    counts = Counter(domains.get(candidate.folder, "unknown") for candidate in candidates)
    return dict(counts.most_common())
