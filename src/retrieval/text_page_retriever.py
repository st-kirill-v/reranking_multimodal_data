from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[._+\-][a-zA-Z0-9]+)*")
PAGE_TEXT_FILENAMES = ("pages_text.json", "page_text.json")
TEXT_EVIDENCE_FIELDS = ("ocr", "page_text", "caption", "table_text")
FIELD_KEYS = {
    "ocr": ("ocr", "ocr_text", "page_ocr"),
    "page_text": ("text", "page_text"),
    "caption": ("caption", "captions", "figure_caption", "table_caption"),
    "table_text": ("table_text", "tables_text", "table", "tables"),
}
SIDE_CAR_FILENAMES = {
    "ocr": ("pages_ocr.json", "page_ocr.json", "ocr_text.json", "ocr.json"),
    "caption": ("captions.json", "page_captions.json", "pages_captions.json"),
    "table_text": ("tables_text.json", "page_tables_text.json", "pages_tables_text.json"),
}


@dataclass
class TextPageCandidate:
    doc_id: str
    page: int
    text: str
    score: float
    rank: int
    source_path: str
    source: str = "text_page_bm25"
    evidence_fields: list[str] | None = None
    text_rerank_score: float | None = None
    text_rerank_rank: int | None = None

    @property
    def folder(self) -> str:
        return self.doc_id

    def to_json(self) -> dict[str, Any]:
        return {
            "folder": self.doc_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "source_path": self.source_path,
            "source": self.source,
            "evidence_fields": self.evidence_fields or [],
            "text_rerank_score": self.text_rerank_score,
            "text_rerank_rank": self.text_rerank_rank,
        }


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def find_page_text_path(doc_folder: Path) -> Path | None:
    extracted = doc_folder / "extracted"
    for filename in PAGE_TEXT_FILENAMES:
        path = extracted / filename
        if path.exists():
            return path
    return None


def load_page_text_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("pages", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported page text JSON shape: {path}")


def normalize_source_fields(source_fields: list[str] | None) -> list[str]:
    if not source_fields:
        return ["page_text"]
    normalized = []
    aliases = {
        "ocr_text": "ocr",
        "page": "page_text",
        "text": "page_text",
        "captions": "caption",
        "tables": "table_text",
        "tables_text": "table_text",
    }
    for field in source_fields:
        item = aliases.get(str(field).strip(), str(field).strip())
        if item in TEXT_EVIDENCE_FIELDS and item not in normalized:
            normalized.append(item)
    return normalized or ["page_text"]


def _stringify_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_stringify_field(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key in ("text", "content", "caption", "table_text", "ocr_text"):
            if key in value:
                parts.append(_stringify_field(value[key]))
        if not parts:
            parts = [f"{key}: {_stringify_field(val)}" for key, val in value.items()]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def _load_side_car_field_map(extracted_dir: Path, field: str) -> dict[int, str]:
    for filename in SIDE_CAR_FILENAMES.get(field, ()):
        path = extracted_dir / filename
        if not path.exists():
            continue
        try:
            rows = load_page_text_rows(path)
        except Exception:
            continue
        mapping: dict[int, str] = {}
        for row in rows:
            try:
                page = int(row.get("page"))
            except (TypeError, ValueError):
                continue
            text = ""
            for key in FIELD_KEYS[field]:
                if key in row:
                    text = _stringify_field(row[key])
                    if text:
                        break
            if not text:
                text = _stringify_field(row)
            if text:
                mapping[page] = text
        return mapping
    return {}


def build_page_evidence_text(
    page_row: dict[str, Any],
    *,
    page: int,
    source_fields: list[str],
    side_car_maps: dict[str, dict[int, str]] | None = None,
) -> tuple[str, list[str]]:
    side_car_maps = side_car_maps or {}
    chunks: list[str] = []
    available: list[str] = []
    for field in normalize_source_fields(source_fields):
        text = ""
        for key in FIELD_KEYS[field]:
            if key in page_row:
                text = _stringify_field(page_row[key])
                if text:
                    break
        if not text:
            text = side_car_maps.get(field, {}).get(page, "")
        if not text:
            continue
        available.append(field)
        label = field.upper()
        chunks.append(f"[{label}]\n{text.strip()}")
    return "\n\n".join(chunks).strip(), available


def collect_text_evidence_records(
    data_dir: str | Path,
    *,
    source_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(data_dir)
    fields = normalize_source_fields(source_fields)
    records: list[dict[str, Any]] = []
    for doc_folder in sorted(
        root.iterdir(),
        key=lambda path: int(path.name) if path.name.isdigit() else 10**9,
    ):
        if not doc_folder.is_dir() or not doc_folder.name.isdigit():
            continue
        page_text_path = find_page_text_path(doc_folder)
        if page_text_path is None:
            continue
        try:
            rows = load_page_text_rows(page_text_path)
        except Exception:
            continue
        extracted_dir = doc_folder / "extracted"
        side_car_maps = {
            field: _load_side_car_field_map(extracted_dir, field)
            for field in fields
            if field != "page_text"
        }
        for row in rows:
            try:
                page = int(row.get("page"))
            except (TypeError, ValueError):
                continue
            text, available = build_page_evidence_text(
                row,
                page=page,
                source_fields=fields,
                side_car_maps=side_car_maps,
            )
            if not text:
                continue
            records.append(
                {
                    "doc_id": doc_folder.name,
                    "page": page,
                    "text": text,
                    "source_path": page_text_path.as_posix(),
                    "requested_source_fields": fields,
                    "available_source_fields": available,
                }
            )
    return records


class TextPageBM25Retriever:
    """In-memory BM25 retriever over DocBench extracted page text.

    This intentionally uses only extracted page-level text and does not read answers,
    evidence, gold pages, or visual artifacts.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets/docbench",
        *,
        source_fields: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.source_fields = normalize_source_fields(source_fields)
        self.records: list[dict[str, Any]] = []
        self.doc_lengths: list[int] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avgdl = 0.0
        self.load_latency = 0.0
        self._load()

    def _load(self) -> None:
        start = time.time()
        for record in collect_text_evidence_records(
            self.data_dir, source_fields=self.source_fields
        ):
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            page = int(record["page"])
            tokens = tokenize(text)
            if not tokens:
                continue
            term_freq = Counter(tokens)
            self.records.append(record)
            self.term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.update(term_freq.keys())
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.load_latency = time.time() - start

    def search(self, query: str, top_k: int = 30) -> list[TextPageCandidate]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        scored: list[tuple[float, int]] = []
        total_docs = max(len(self.records), 1)
        k1 = 1.5
        b = 0.75
        for idx, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            for term, qf in query_counts.items():
                tf = term_freq.get(term, 0)
                if tf <= 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf * (k1 + 1) / denom) * qf
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            record = self.records[idx]
            candidates.append(
                TextPageCandidate(
                    doc_id=str(record["doc_id"]),
                    page=int(record["page"]),
                    text=str(record["text"]),
                    score=float(score),
                    rank=rank,
                    source_path=str(record["source_path"]),
                    evidence_fields=list(record.get("available_source_fields") or []),
                )
            )
        return candidates

    def stats(self) -> dict[str, Any]:
        by_doc: dict[str, int] = defaultdict(int)
        for record in self.records:
            by_doc[str(record["doc_id"])] += 1
        return {
            "num_pages": len(self.records),
            "num_docs": len(by_doc),
            "avg_doc_length_tokens": self.avgdl,
            "load_latency": self.load_latency,
            "source_fields": self.source_fields,
        }
