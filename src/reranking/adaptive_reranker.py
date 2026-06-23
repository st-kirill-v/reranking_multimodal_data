from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.cropping.layout_aware import question_crop_intent


AdaptiveRoute = Literal["high_confidence", "table_or_text", "visual", "unknown"]


TABLE_TERMS = {
    "table",
    "tabular",
    "row",
    "column",
    "cell",
    "total",
    "amount",
    "value",
    "percentage",
    "percent",
}

VISUAL_TERMS = {
    "figure",
    "fig.",
    "fig",
    "chart",
    "graph",
    "plot",
    "diagram",
    "image",
    "picture",
    "visual",
    "map",
}


def _candidate_score(candidate: Any) -> float:
    if isinstance(candidate, dict):
        return float(candidate.get("score") or 0.0)
    return float(getattr(candidate, "score", 0.0) or 0.0)


def _metadata_value(metadata: dict[str, Any] | None, *keys: str) -> str:
    if not metadata:
        return ""
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return ""


def should_skip_reranker(
    candidates: list[Any],
    threshold_top1: float | None = None,
    threshold_gap: float | None = None,
) -> bool:
    """Return True when retrieval is confident enough to skip an expensive reranker.

    The check is disabled when both thresholds are omitted. When both thresholds
    are provided, both must pass. This keeps the default behavior conservative.
    """

    if not candidates or (threshold_top1 is None and threshold_gap is None):
        return False

    scores = [_candidate_score(candidate) for candidate in candidates[:2]]
    top1 = scores[0]
    if threshold_top1 is not None and top1 < threshold_top1:
        return False
    if threshold_gap is not None:
        if len(scores) < 2:
            return False
        if top1 - scores[1] < threshold_gap:
            return False
    return True


def detect_question_route(
    question: str,
    metadata: dict[str, Any] | None = None,
    candidates: list[Any] | None = None,
    *,
    threshold_top1: float | None = None,
    threshold_gap: float | None = None,
) -> AdaptiveRoute:
    """Choose an adaptive reranking route for one DocBench question."""

    if should_skip_reranker(
        candidates or [],
        threshold_top1=threshold_top1,
        threshold_gap=threshold_gap,
    ):
        return "high_confidence"

    question_type = _metadata_value(metadata, "type", "original_type").lower()
    intent = question_crop_intent(question)
    lowered = (question or "").lower()
    tokens = set(lowered.replace("/", " ").replace("-", " ").split())

    if question_type == "multimodal-f" or intent == "visual" or tokens & VISUAL_TERMS:
        return "visual"
    if question_type == "multimodal-t" or intent == "table" or tokens & TABLE_TERMS:
        return "table_or_text"
    return "unknown"


@dataclass
class AdaptiveRouteInfo:
    route: AdaptiveRoute
    strategy: str
    skipped_reranker: bool
    latency: float
    reason: str
    retrieval_top1_score: float | None = None
    retrieval_top1_gap: float | None = None


@dataclass
class AdaptiveReranker:
    """Route each question to an existing reranking strategy.

    This class intentionally delegates scoring to already implemented rerankers.
    It adds only question-level routing and trace metadata for experiments.
    """

    image_reranker: Any | None = None
    text_image_reranker: Any | None = None
    no_reranker: Any | None = None
    threshold_top1: float | None = None
    threshold_gap: float | None = None
    high_confidence_strategy: str = "no_reranker"
    route_strategy_map: dict[AdaptiveRoute, str] = field(default_factory=dict)
    last_route_info: AdaptiveRouteInfo | None = None

    def __post_init__(self) -> None:
        defaults: dict[AdaptiveRoute, str] = {
            "high_confidence": self.high_confidence_strategy,
            "table_or_text": "text_image_reranker",
            "visual": "image_reranker",
            "unknown": "text_image_reranker",
        }
        defaults.update(self.route_strategy_map)
        self.route_strategy_map = defaults

    def _scores_debug(self, candidates: list[Any]) -> tuple[float | None, float | None]:
        if not candidates:
            return None, None
        top1 = _candidate_score(candidates[0])
        if len(candidates) < 2:
            return top1, None
        return top1, top1 - _candidate_score(candidates[1])

    def _run_strategy(
        self,
        strategy: str,
        question: str,
        candidates: list[Any],
    ) -> tuple[list[Any], bool]:
        if strategy in {"none", "no_reranker", "skip"}:
            if self.no_reranker is not None:
                return self.no_reranker.rerank(question, candidates), True
            return list(candidates), True
        if strategy == "image_reranker":
            if self.image_reranker is None:
                return list(candidates), True
            return self.image_reranker.rerank(question, candidates), False
        if strategy == "text_image_reranker":
            if self.text_image_reranker is None:
                return list(candidates), True
            return self.text_image_reranker.rerank(question, candidates), False
        raise ValueError(f"Unsupported adaptive reranking strategy: {strategy}")

    def rerank(
        self,
        question: str,
        candidates: list[Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        start = time.time()
        route = detect_question_route(
            question,
            metadata=metadata,
            candidates=candidates,
            threshold_top1=self.threshold_top1,
            threshold_gap=self.threshold_gap,
        )
        strategy = self.route_strategy_map.get(route, "text_image_reranker")
        reranked, skipped = self._run_strategy(strategy, question, candidates)
        top1, gap = self._scores_debug(candidates)
        reason = (
            f"route={route}; strategy={strategy}; "
            f"type={_metadata_value(metadata, 'type', 'original_type') or 'unknown'}; "
            f"intent={question_crop_intent(question)}"
        )
        self.last_route_info = AdaptiveRouteInfo(
            route=route,
            strategy=strategy,
            skipped_reranker=skipped,
            latency=time.time() - start,
            reason=reason,
            retrieval_top1_score=top1,
            retrieval_top1_gap=gap,
        )
        return reranked


class LazyReranker:
    """Load a heavy reranker only when its route is used for the first time."""

    def __init__(self, factory: Any, name: str) -> None:
        self.factory = factory
        self.name = name
        self._instance: Any | None = None

    def _get(self) -> Any:
        if self._instance is None:
            print(f"[Adaptive Reranker] Loading {self.name}")
            self._instance = self.factory()
        return self._instance

    def rerank(self, question: str, candidates: list[Any]) -> list[Any]:
        return self._get().rerank(question, candidates)
