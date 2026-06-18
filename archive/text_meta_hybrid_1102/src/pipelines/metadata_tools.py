from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.text_bm25_retriever import TextBM25Result


PAGE_TEXT_FILENAMES = ("page_text.json", "pages_text.json")


@dataclass(frozen=True)
class MetadataToolResult:
    answer: str | None
    tool_used: str | None
    route_subtype: str
    pages: list[TextBM25Result]
    context: str


def find_page_text_path(data_dir: Path, doc_id: str) -> Path | None:
    extracted = data_dir / str(doc_id) / "extracted"
    for filename in PAGE_TEXT_FILENAMES:
        path = extracted / filename
        if path.exists():
            return path
    return None


def load_document_pages(data_dir: Path, doc_id: str) -> list[TextBM25Result]:
    path = find_page_text_path(data_dir, doc_id)
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("pages", "records", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []

    pages: list[TextBM25Result] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        pages.append(
            TextBM25Result(
                doc_id=str(doc_id),
                page=page_number,
                score=1.0,
                text=str(item.get("text") or ""),
                source_path=path.as_posix(),
            )
        )
    return sorted(pages, key=lambda page: page.page)


def detect_page_count(question: str) -> bool:
    q = question.lower()
    page_terms = (
        "how many pages",
        "number of pages",
        "total pages",
        "pages does the document have",
        "pages does the report consist",
    )
    return any(term in q for term in page_terms)


def detect_page_reference(question: str) -> int | None:
    match = re.search(r"\bpage\s+(\d+)\b", question, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def detect_mention_count_phrase(question: str) -> str | None:
    q = question.strip()
    quoted = re.search(r"['\"]([^'\"]{1,120})['\"]", q)
    if quoted and re.search(r"\bmention", q, flags=re.IGNORECASE):
        return quoted.group(1).strip()

    match = re.search(
        r"\bmention(?:s|ed)?\s+(?P<phrase>.+?)(?:\?|$)",
        q,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    phrase = match.group("phrase").strip()
    phrase = re.sub(r"^(the\s+)?(term|phrase|word)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip(" \"'“”‘’.")
    if not phrase or len(phrase.split()) > 8:
        return None
    return phrase


def count_exact_mentions(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    return len(re.findall(re.escape(phrase), text or "", flags=re.IGNORECASE))


def build_context_from_pages(pages: list[TextBM25Result], max_chars: int) -> str:
    chunks: list[str] = []
    total = 0
    for page in pages:
        chunk = (
            f"[doc_id={page.doc_id} page={page.page} score={page.score:.4f}]\n{page.text.strip()}"
        )
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


def run_metadata_tool(
    *,
    question: str,
    question_type: str,
    data_dir: Path,
    doc_id: str,
    context_max_chars: int,
) -> MetadataToolResult | None:
    pages = load_document_pages(data_dir, doc_id)
    if not pages:
        return None

    page_number = detect_page_reference(question)
    if page_number is not None:
        selected = [page for page in pages if page.page == page_number]
        return MetadataToolResult(
            answer=None,
            tool_used="page_direct",
            route_subtype="page_direct",
            pages=selected,
            context=build_context_from_pages(selected, context_max_chars),
        )

    if question_type == "meta-data" and detect_page_count(question):
        page_count = max((page.page for page in pages), default=len(pages))
        return MetadataToolResult(
            answer=str(page_count),
            tool_used="page_count",
            route_subtype="metadata_tool",
            pages=[],
            context="",
        )

    if question_type == "meta-data":
        phrase = detect_mention_count_phrase(question)
        if phrase:
            full_text = "\n".join(page.text for page in pages)
            count = count_exact_mentions(full_text, phrase)
            return MetadataToolResult(
                answer=str(count),
                tool_used="mention_count",
                route_subtype="metadata_tool",
                pages=[],
                context="",
            )

    return None


def normalize_text_route_answer(answer: str) -> str:
    value = (answer or "").strip()
    if not value:
        return "NOT FOUND"
    norm = re.sub(r"\s+", " ", value.lower()).strip(" .")
    if re.fullmatch(r"yes[\s.!]*", norm) or norm.startswith("yes,") or norm.startswith("yes "):
        return "Yes."
    if re.fullmatch(r"no[\s.!]*", norm) or norm.startswith("no,") or norm.startswith("no "):
        return "No."
    not_found_markers = (
        "not found",
        "not mentioned",
        "not available",
        "does not mention",
        "doesn't mention",
        "does not contain",
        "doesn't contain",
        "not explicitly present",
        "cannot be found",
        "no information",
    )
    if any(marker in norm for marker in not_found_markers):
        return "NOT FOUND"
    return value
