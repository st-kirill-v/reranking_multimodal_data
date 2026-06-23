from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def _candidate_score(candidate: Any) -> float:
    if isinstance(candidate, dict):
        return float(candidate.get("score") or 0.0)
    return float(getattr(candidate, "score", 0.0) or 0.0)


def compute_retrieval_confidence(candidates: list[Any]) -> dict[str, float | int | None]:
    """Compute confidence statistics from first-stage retrieval scores only."""

    candidates_count = len(candidates)
    top1_score = _candidate_score(candidates[0]) if candidates_count >= 1 else None
    top2_score = _candidate_score(candidates[1]) if candidates_count >= 2 else None
    gap = None
    relative_gap = None
    if top1_score is not None and top2_score is not None:
        gap = top1_score - top2_score
        if abs(top1_score) > 1e-12:
            relative_gap = gap / abs(top1_score)
    return {
        "top1_score": top1_score,
        "top2_score": top2_score,
        "gap": gap,
        "relative_gap": relative_gap,
        "candidates_count": candidates_count,
    }


def should_skip_reranker(
    candidates: list[Any],
    threshold_top1: float,
    threshold_gap: float,
) -> tuple[bool, str, dict[str, float | int | None]]:
    """Decide whether to skip the fallback reranker using retrieval confidence."""

    stats = compute_retrieval_confidence(candidates)
    top1 = stats["top1_score"]
    gap = stats["gap"]
    if top1 is None:
        return False, "no_candidates", stats
    if gap is None:
        return False, "not_enough_candidates_for_gap", stats
    if top1 >= threshold_top1 and gap >= threshold_gap:
        return True, "top1_and_gap_passed", stats
    if top1 < threshold_top1 and gap < threshold_gap:
        return False, "top1_and_gap_below_thresholds", stats
    if top1 < threshold_top1:
        return False, "top1_below_threshold", stats
    return False, "gap_below_threshold", stats


@dataclass
class ThresholdSkipDecision:
    skip_reranker: bool
    route_used: str
    reason: str
    threshold_top1: float
    threshold_gap: float
    latency: float
    confidence_stats: dict[str, float | int | None]


@dataclass
class ThresholdSkipReranker:
    """Skip an expensive reranker when retrieval is already confident."""

    fallback_reranker: Any
    threshold_top1: float
    threshold_gap: float
    last_decision: ThresholdSkipDecision | None = None

    def rerank(
        self,
        question: str,
        candidates: list[Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        del metadata
        start = time.time()
        skip, reason, stats = should_skip_reranker(
            candidates,
            threshold_top1=self.threshold_top1,
            threshold_gap=self.threshold_gap,
        )
        if skip:
            route_used = "skipped"
            reranked = list(candidates)
        else:
            route_used = "vl_reranker"
            reranked = self.fallback_reranker.rerank(question, candidates)
        self.last_decision = ThresholdSkipDecision(
            skip_reranker=skip,
            route_used=route_used,
            reason=reason,
            threshold_top1=self.threshold_top1,
            threshold_gap=self.threshold_gap,
            latency=time.time() - start,
            confidence_stats=stats,
        )
        return reranked
