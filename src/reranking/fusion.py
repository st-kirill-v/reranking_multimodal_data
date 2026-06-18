from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.mmrag.schema import RetrievalCandidate
from src.retrieval.text_page_retriever import collect_text_evidence_records


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[._+\-][a-zA-Z0-9]+)*")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
EXPLICIT_REF_RE = re.compile(
    r"\b(?P<label>table|figure|fig\.?|chart|graph)\s*(?P<number>\d+[a-z]?)\b",
    re.IGNORECASE,
)


@dataclass
class FusionWeights:
    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 1.0
    lambda_number: float = 0.05
    lambda_keyword: float = 0.05
    lambda_exact_phrase: float = 0.05
    lambda_table_header: float = 0.20


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 2]


def normalize_values(values: list[float]) -> list[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return [0.0 for _ in values]
    low = min(finite_values)
    high = max(finite_values)
    if abs(high - low) <= 1e-12:
        return [1.0 if math.isfinite(value) else 0.0 for value in values]
    return [(value - low) / (high - low) if math.isfinite(value) else 0.0 for value in values]


def load_text_evidence_map(
    data_dir: str | Path,
    *,
    source_fields: list[str] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    records = collect_text_evidence_records(data_dir, source_fields=source_fields)
    evidence: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        try:
            page = int(record.get("page"))
        except (TypeError, ValueError):
            continue
        evidence[(str(record.get("doc_id")), page)] = {
            "text": str(record.get("text") or ""),
            "available_source_fields": list(record.get("available_source_fields") or []),
            "requested_source_fields": list(record.get("requested_source_fields") or []),
        }
    return evidence


def attach_text_evidence(
    candidates: list[RetrievalCandidate],
    evidence_map: dict[tuple[str, int], dict[str, Any]],
) -> None:
    for candidate in candidates:
        evidence = evidence_map.get((str(candidate.folder), int(candidate.page)), {})
        candidate.text = str(evidence.get("text") or "")  # type: ignore[attr-defined]
        candidate.evidence_fields = list(evidence.get("available_source_fields") or [])  # type: ignore[attr-defined]
        candidate.text_rerank_score = None  # type: ignore[attr-defined]
        candidate.text_rerank_rank = None  # type: ignore[attr-defined]
        candidate.fusion_features = {}  # type: ignore[attr-defined]
        candidate.fusion_score = None  # type: ignore[attr-defined]
        candidate.fusion_rank = None  # type: ignore[attr-defined]


def number_match_score(question: str, evidence_text: str) -> float:
    question_numbers = {
        match.group(0).replace(",", "") for match in NUMBER_RE.finditer(question or "")
    }
    if not question_numbers:
        return 0.0
    evidence_numbers = {
        match.group(0).replace(",", "") for match in NUMBER_RE.finditer(evidence_text or "")
    }
    if not evidence_numbers:
        return 0.0
    return len(question_numbers & evidence_numbers) / max(len(question_numbers), 1)


def keyword_match_score(question: str, evidence_text: str) -> float:
    stopwords = {
        "what",
        "which",
        "when",
        "where",
        "does",
        "did",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "according",
        "table",
        "figure",
    }
    q_tokens = {token for token in tokenize(question) if token not in stopwords}
    if not q_tokens:
        return 0.0
    e_tokens = set(tokenize(evidence_text))
    return len(q_tokens & e_tokens) / len(q_tokens)


def exact_phrase_match_score(question: str, evidence_text: str) -> float:
    q_tokens = tokenize(question)
    if len(q_tokens) < 3:
        return 0.0
    evidence_norm = " ".join(tokenize(evidence_text))
    if not evidence_norm:
        return 0.0
    max_score = 0.0
    for ngram_size in (5, 4, 3):
        if len(q_tokens) < ngram_size:
            continue
        total = len(q_tokens) - ngram_size + 1
        hits = 0
        for idx in range(total):
            phrase = " ".join(q_tokens[idx : idx + ngram_size])
            if phrase in evidence_norm:
                hits += 1
        max_score = max(max_score, hits / max(total, 1))
    return max_score


def table_header_match_score(question: str, evidence_text: str) -> float:
    question_norm = (question or "").lower()
    evidence_norm = (evidence_text or "").lower()
    score = 0.0
    refs = list(EXPLICIT_REF_RE.finditer(question_norm))
    if refs:
        for ref in refs:
            label = ref.group("label").replace(".", "")
            if label == "fig":
                label = "figure"
            needle = f"{label} {ref.group('number')}".lower()
            if needle in evidence_norm:
                return 1.0
        return 0.0
    table_terms = {"table", "row", "column", "accuracy", "score", "f1", "ppl", "bleu"}
    figure_terms = {"figure", "fig", "chart", "graph", "plot", "diagram"}
    q_tokens = set(tokenize(question_norm))
    if q_tokens & table_terms and ("[table_text]" in evidence_norm or "table " in evidence_norm):
        score += 0.5
    if q_tokens & figure_terms and (
        "[caption]" in evidence_norm
        or "figure " in evidence_norm
        or "fig " in evidence_norm
        or "chart " in evidence_norm
        or "graph " in evidence_norm
    ):
        score += 0.5
    metric_terms = {"accuracy", "score", "f1", "ppl", "bleu", "wer", "auc"}
    if q_tokens & metric_terms and any(term in evidence_norm for term in metric_terms):
        score += 0.25
    return min(score, 1.0)


def compute_fusion_features(question: str, evidence_text: str) -> dict[str, float]:
    return {
        "number_match": number_match_score(question, evidence_text),
        "keyword_match": keyword_match_score(question, evidence_text),
        "exact_phrase_match": exact_phrase_match_score(question, evidence_text),
        "table_header_match": table_header_match_score(question, evidence_text),
    }


def fuse_candidates(
    question: str,
    candidates: list[RetrievalCandidate],
    *,
    weights: FusionWeights,
) -> list[RetrievalCandidate]:
    retrieval_scores = normalize_values([float(candidate.score) for candidate in candidates])
    text_scores = normalize_values(
        [float(getattr(candidate, "text_rerank_score", 0.0) or 0.0) for candidate in candidates]
    )
    image_scores = normalize_values(
        [float(candidate.rerank_score or 0.0) for candidate in candidates]
    )

    ranked: list[tuple[float, int, RetrievalCandidate]] = []
    for idx, candidate in enumerate(candidates):
        evidence_text = str(getattr(candidate, "text", "") or "")
        features = compute_fusion_features(question, evidence_text)
        base_score = (
            weights.alpha * retrieval_scores[idx]
            + weights.beta * text_scores[idx]
            + weights.gamma * image_scores[idx]
        )
        boost_score = (
            weights.lambda_number * features["number_match"]
            + weights.lambda_keyword * features["keyword_match"]
            + weights.lambda_exact_phrase * features["exact_phrase_match"]
            + weights.lambda_table_header * features["table_header_match"]
        )
        final_score = base_score + boost_score
        candidate.fusion_features = {  # type: ignore[attr-defined]
            **features,
            "retrieval_score_norm": retrieval_scores[idx],
            "text_rerank_score_norm": text_scores[idx],
            "image_rerank_score_norm": image_scores[idx],
            "base_score": base_score,
            "boost_score": boost_score,
        }
        candidate.fusion_score = final_score  # type: ignore[attr-defined]
        ranked.append((final_score, -idx, candidate))

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    fused: list[RetrievalCandidate] = []
    for rank, (_score, _idx, candidate) in enumerate(ranked, start=1):
        candidate.fusion_rank = rank  # type: ignore[attr-defined]
        fused.append(candidate)
    return fused
