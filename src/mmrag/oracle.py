from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageTextMatch:
    page: int
    exact_answer: bool
    number_recall: float
    keyword_recall: float
    matched_numbers: list[str]
    matched_keywords: list[str]


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "has",
    "have",
    "had",
    "which",
    "what",
    "when",
    "where",
    "who",
    "how",
    "according",
    "table",
    "figure",
    "score",
    "accuracy",
    "using",
    "resulting",
    "answer",
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numbers(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"\d+(?:\.\d+)?%?", normalized)


def extract_keywords(text: str) -> list[str]:
    normalized = normalize_text(text)
    words = re.findall(r"[a-z][a-z0-9_+\-]*", normalized)
    return [word for word in words if len(word) >= 3 and word not in STOP_WORDS]


def load_pages_text(folder: Path) -> list[dict]:
    path = folder / "extracted" / "pages_text.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_answer_pages(folder: Path, answer: str, evidence: str = "") -> list[PageTextMatch]:
    pages = load_pages_text(folder)
    answer_norm = normalize_text(answer)
    numbers = extract_numbers(answer)
    keywords = extract_keywords(answer)
    evidence_keywords = extract_keywords(evidence)
    if evidence_keywords:
        keywords = list(dict.fromkeys([*keywords, *evidence_keywords]))

    matches: list[PageTextMatch] = []
    for row in pages:
        page_text = normalize_text(row.get("text", ""))
        page_numbers = set(extract_numbers(page_text))
        page_words = set(extract_keywords(page_text))
        matched_numbers = [
            num for num in numbers if num in page_numbers or num.rstrip("%") in page_numbers
        ]
        matched_keywords = [word for word in keywords if word in page_words]
        exact = bool(answer_norm and answer_norm in page_text)
        number_recall = len(set(matched_numbers)) / len(set(numbers)) if numbers else 0.0
        keyword_recall = len(set(matched_keywords)) / len(set(keywords)) if keywords else 0.0
        if exact or number_recall > 0 or keyword_recall >= 0.25:
            matches.append(
                PageTextMatch(
                    page=int(row.get("page", 0)),
                    exact_answer=exact,
                    number_recall=number_recall,
                    keyword_recall=keyword_recall,
                    matched_numbers=matched_numbers,
                    matched_keywords=matched_keywords[:20],
                )
            )

    matches.sort(
        key=lambda item: (item.exact_answer, item.number_recall, item.keyword_recall),
        reverse=True,
    )
    return matches
