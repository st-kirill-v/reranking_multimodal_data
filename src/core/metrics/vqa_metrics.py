from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


STOP_WORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "by",
    "company",
    "for",
    "from",
    "has",
    "have",
    "in",
    "inc",
    "is",
    "it",
    "of",
    "on",
    "or",
    "question",
    "reported",
    "the",
    "to",
    "was",
    "were",
    "which",
    "with",
}

ENTITY_SUFFIXES = {
    "ag",
    "bank",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "pty",
    "sa",
}

UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "pp": "percentage_point",
    "percentage point": "percentage_point",
    "percentage points": "percentage_point",
    "point": "point",
    "points": "point",
    "million": "million",
    "millions": "million",
    "m": "million",
    "billion": "billion",
    "billions": "billion",
    "bn": "billion",
    "thousand": "thousand",
    "thousands": "thousand",
    "crore": "crore",
    "crores": "crore",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "$": "usd",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "dkk": "dkk",
    "chf": "chf",
    "rmb": "rmb",
    "cny": "rmb",
    "rupee": "inr",
    "rupees": "inr",
    "inr": "inr",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lbs",
    "lbs": "lbs",
    "pound": "lbs",
    "pounds": "lbs",
    "mtco2e": "mtco2e",
    "mtco2-e": "mtco2e",
    "mtco2": "mtco2",
    "co2e": "co2e",
    "tonne": "tonne",
    "tonnes": "tonne",
    "ton": "ton",
    "tons": "ton",
    "barrel": "barrel",
    "barrels": "barrel",
    "share": "share",
    "shares": "share",
    "store": "store",
    "stores": "store",
    "employee": "employee",
    "employees": "employee",
    "dialogue": "dialogue",
    "dialogues": "dialogue",
    "sentence": "sentence",
    "sentences": "sentence",
    "token": "token",
    "tokens": "token",
    "acre": "acre",
    "acres": "acre",
    "mw": "mw",
    "f1": "f1",
    "bleu": "bleu",
    "auprc": "auprc",
    "wer": "wer",
}

TABLE_HEAVY_TERMS = {
    "according to table",
    "table",
    "row",
    "column",
    "score",
    "accuracy",
    "f1",
    "bleu",
    "revenue",
    "income",
    "sales",
    "assets",
    "cost",
    "costs",
    "expenses",
    "shares",
    "employees",
    "stores",
    "percentage",
    "increase",
    "decrease",
    "difference",
    "average",
    "total",
}


@dataclass(frozen=True)
class NumericValue:
    raw: str
    value: float
    normalized: str
    is_integer: bool
    is_percentage: bool


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("co2-e", "co2e").replace("co2 e", "co2e")
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer(text: str) -> str:
    text = normalize_text(text)
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_number_token(token: str) -> str:
    token = normalize_text(token)
    token = token.replace(",", "")
    token = token.replace("$", "").replace("₹", "").replace("€", "").replace("£", "")
    token = token.rstrip(".")
    if token.endswith("%"):
        token = token[:-1]
    try:
        value = float(token)
    except ValueError:
        return token
    if value.is_integer():
        return str(int(value))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def extract_numeric_values(text: str) -> list[NumericValue]:
    values = []
    for match in re.finditer(r"[$₹€£]?\s*-?\d[\d,]*(?:\.\d+)?\s*%?", text or ""):
        raw = match.group(0).strip()
        normalized = normalize_number_token(raw)
        try:
            value = float(normalized)
        except ValueError:
            continue
        values.append(
            NumericValue(
                raw=raw,
                value=value,
                normalized=normalized,
                is_integer=float(value).is_integer() and "." not in raw,
                is_percentage="%" in raw,
            )
        )
    return values


def extract_numbers(text: str) -> list[str]:
    return [value.normalized for value in extract_numeric_values(text)]


def _numeric_multiset_exact(
    generated: list[NumericValue], expected: list[NumericValue]
) -> float | None:
    if not expected:
        return None
    return float(
        Counter(item.normalized for item in generated)
        == Counter(item.normalized for item in expected)
    )


def _numbers_relaxed_equal(generated: NumericValue, expected: NumericValue) -> bool:
    if expected.is_percentage or generated.is_percentage:
        return abs(generated.value - expected.value) <= 0.1
    if expected.is_integer:
        return generated.normalized == expected.normalized
    denominator = max(abs(expected.value), 1e-12)
    return abs(generated.value - expected.value) / denominator <= 0.01


def _numeric_relaxed_match(
    generated: list[NumericValue], expected: list[NumericValue]
) -> float | None:
    if not expected:
        return None
    unused = list(generated)
    for expected_value in expected:
        match_index = next(
            (
                idx
                for idx, generated_value in enumerate(unused)
                if _numbers_relaxed_equal(generated_value, expected_value)
            ),
            None,
        )
        if match_index is None:
            return 0.0
        unused.pop(match_index)
    return 1.0


def normalize_unit_token(token: str) -> str | None:
    token = normalize_text(token)
    token = token.strip(" .,:;()[]{}")
    token = token.replace("co₂", "co2").replace("co2-e", "co2e")
    return UNIT_ALIASES.get(token)


def extract_units(text: str) -> set[str]:
    normalized = normalize_text(text)
    units = set()
    if "%" in normalized:
        units.add("%")
    if "$" in text:
        units.add("usd")
    if "€" in text:
        units.add("eur")
    if "₹" in text:
        units.add("inr")

    for raw in re.findall(r"[a-z%$€₹][a-z0-9%$€₹\-]*", normalized):
        unit = normalize_unit_token(raw)
        if unit:
            units.add(unit)

    phrase_units = {
        "percentage point": "percentage_point",
        "percentage points": "percentage_point",
        "million dollars": "usd",
        "billion dollars": "usd",
        "thousand shares": "share",
    }
    for phrase, unit in phrase_units.items():
        if phrase in normalized:
            units.add(unit)
    return units


def compute_unit_match(generated: str, expected: str) -> float | None:
    expected_units = extract_units(expected)
    if not expected_units:
        return None
    generated_units = extract_units(generated)
    return float(expected_units.issubset(generated_units))


def normalize_entity_text(text: str) -> set[str]:
    text = normalize_text(text)
    text = re.sub(r"[$₹€£]?\s*-?\d[\d,]*(?:\.\d+)?\s*%?", " ", text)
    words = re.findall(r"[a-z][a-z0-9&+\-]*", text)
    tokens = set()
    for word in words:
        word = word.strip("-")
        if not word or word in STOP_WORDS or word in ENTITY_SUFFIXES:
            continue
        unit = normalize_unit_token(word)
        if unit:
            continue
        tokens.add(word)
    return tokens


def compute_entity_match(generated: str, expected: str) -> float | None:
    expected_tokens = normalize_entity_text(expected)
    if not expected_tokens:
        return None
    generated_tokens = normalize_entity_text(generated)
    if not generated_tokens:
        return 0.0
    if expected_tokens.issubset(generated_tokens):
        return 1.0
    intersection = expected_tokens & generated_tokens
    jaccard = len(intersection) / len(expected_tokens | generated_tokens)
    recall = len(intersection) / len(expected_tokens)
    return float(jaccard >= 0.8 or recall >= 0.9)


def compute_similarity(generated: str, expected: str) -> tuple[float, float]:
    if generated in {"NOT FOUND", "ERROR", "TIMEOUT"}:
        return 0.0, 0.0

    gen = normalize_answer(generated)
    exp = normalize_answer(expected)
    exact = (
        1.0
        if gen.replace(" ", "").replace("%", "") == exp.replace(" ", "").replace("%", "")
        else 0.0
    )

    gen_numbers = re.findall(r"\d+(?:\.\d+)?%?", gen)
    exp_numbers = re.findall(r"\d+(?:\.\d+)?%?", exp)

    stop_words = {
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
        "been",
        "has",
        "have",
        "had",
        "does",
        "do",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "can",
        "answer",
        "question",
    }
    gen_words = set(
        word
        for word in re.findall(r"[a-z][a-z0-9_\-]*", gen)
        if word not in stop_words and len(word) > 2
    )
    exp_words = set(
        word
        for word in re.findall(r"[a-z][a-z0-9_\-]*", exp)
        if word not in stop_words and len(word) > 2
    )

    word_intersection = gen_words & exp_words
    word_precision = len(word_intersection) / len(gen_words) if gen_words else 0.0
    word_recall = len(word_intersection) / len(exp_words) if exp_words else 0.0
    word_f1 = (
        2 * word_precision * word_recall / (word_precision + word_recall)
        if word_precision + word_recall > 0
        else 0.0
    )

    num_intersection = set(gen_numbers) & set(exp_numbers)
    num_precision = len(num_intersection) / len(gen_numbers) if gen_numbers else 0.0
    num_recall = len(num_intersection) / len(exp_numbers) if exp_numbers else 0.0
    num_f1 = (
        2 * num_precision * num_recall / (num_precision + num_recall)
        if num_precision + num_recall > 0
        else 0.0
    )

    f1 = 0.4 * word_f1 + 0.6 * num_f1 if exp_numbers else word_f1
    return exact, f1


def compute_extended_metrics(generated: str, expected: str) -> dict[str, float | None]:
    gen_norm = normalize_answer(generated)
    exp_norm = normalize_answer(expected)
    gen_compact = re.sub(r"[^a-z0-9]+", "", gen_norm)
    exp_compact = re.sub(r"[^a-z0-9]+", "", exp_norm)
    gen_numbers = extract_numbers(generated)
    exp_numbers = extract_numbers(expected)
    generated_numeric = extract_numeric_values(generated)
    expected_numeric = extract_numeric_values(expected)

    expected_number_counts = Counter(exp_numbers)
    generated_number_counts = Counter(gen_numbers)
    matched_numbers = sum(
        min(count, generated_number_counts[number])
        for number, count in expected_number_counts.items()
    )
    numeric_all_recall = (
        float(matched_numbers == sum(expected_number_counts.values()))
        if expected_number_counts
        else float(not generated_number_counts)
    )
    numeric_any_match = float(
        bool(expected_number_counts)
        and any(number in generated_number_counts for number in expected_number_counts)
    )
    numeric_precision = (
        matched_numbers / sum(generated_number_counts.values())
        if generated_number_counts
        else float(not expected_number_counts)
    )
    numeric_recall = (
        matched_numbers / sum(expected_number_counts.values())
        if expected_number_counts
        else float(not generated_number_counts)
    )

    return {
        "relaxed_exact": float(gen_compact == exp_compact),
        "answer_contains_expected": float(bool(exp_compact) and exp_compact in gen_compact),
        "expected_contains_answer": float(bool(gen_compact) and gen_compact in exp_compact),
        "numeric_any_match": numeric_any_match,
        "numeric_all_recall": numeric_all_recall,
        "numeric_precision": float(numeric_precision),
        "numeric_recall": float(numeric_recall),
        "numeric_exact_match": _numeric_multiset_exact(generated_numeric, expected_numeric),
        "numeric_relaxed_match": _numeric_relaxed_match(generated_numeric, expected_numeric),
        "unit_match": compute_unit_match(generated, expected),
        "entity_match": compute_entity_match(generated, expected),
    }


def oracle_pages_from_row(row: dict[str, Any]) -> list[int]:
    pages = []
    for item in row.get("oracle_pages", []) or []:
        if isinstance(item, int):
            pages.append(item)
        elif isinstance(item, dict) and item.get("page") is not None:
            pages.append(int(item["page"]))
    for key in ("oracle_page", "expected_page", "table_page"):
        if row.get(key) is not None:
            pages.append(int(row[key]))
    return sorted(set(pages))


def expected_table_pages_from_row(row: dict[str, Any]) -> list[int]:
    pages = []
    for key in ("expected_table_pages", "table_pages"):
        value = row.get(key)
        if isinstance(value, list):
            pages.extend(int(page) for page in value)
        elif value is not None:
            pages.append(int(value))
    for key in ("expected_table_page", "table_page"):
        if row.get(key) is not None:
            pages.append(int(row[key]))
    return sorted(set(pages))


def is_table_heavy_question(question: str) -> bool:
    normalized = normalize_text(question)
    return any(term in normalized for term in TABLE_HEAVY_TERMS)


def page_key(folder: str | int | None, page: str | int | None) -> str:
    return f"{folder}/{page}"


def compute_retrieval_metrics(
    row: dict[str, Any], pages: list[dict[str, Any]]
) -> dict[str, float | None]:
    expected_folder = row.get("expected_folder")
    if expected_folder is None and row.get("folder") is not None:
        expected_folder = row.get("folder")
    expected_folder = str(expected_folder) if expected_folder is not None else ""

    selected_folders = {str(page.get("folder")) for page in pages if page.get("folder") is not None}
    doc_hit = float(expected_folder in selected_folders) if expected_folder else None

    oracle_pages = oracle_pages_from_row(row)
    selected_page_keys = {
        page_key(page.get("folder"), page.get("page"))
        for page in pages
        if page.get("folder") is not None and page.get("page") is not None
    }
    oracle_keys = {page_key(expected_folder, page) for page in oracle_pages}
    page_hit = float(bool(selected_page_keys & oracle_keys)) if oracle_keys else None

    table_pages = expected_table_pages_from_row(row)
    if table_pages:
        table_keys = {page_key(expected_folder, page) for page in table_pages}
        table_hit = float(bool(selected_page_keys & table_keys))
    elif is_table_heavy_question(row.get("question", "")) and page_hit is not None:
        table_hit = page_hit
    elif is_table_heavy_question(row.get("question", "")):
        table_hit = None
    else:
        table_hit = None

    return {
        "doc_hit_at_k": doc_hit,
        "page_hit_at_k": page_hit,
        "table_hit_at_k": table_hit,
    }


METRIC_KEYS = [
    "exact",
    "f1",
    "relaxed_exact",
    "answer_contains_expected",
    "expected_contains_answer",
    "numeric_any_match",
    "numeric_all_recall",
    "numeric_precision",
    "numeric_recall",
    "numeric_exact_match",
    "numeric_relaxed_match",
    "unit_match",
    "entity_match",
    "doc_hit_at_k",
    "page_hit_at_k",
    "table_hit_at_k",
]


def mean_available(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and not math.isnan(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def metric_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1_values = [float(row.get("f1", 0.0)) for row in rows]
    output: dict[str, Any] = {
        "count": len(rows),
        "exact_match": mean_available(rows, "exact"),
        "mean_f1": float(np.mean(f1_values)) if f1_values else None,
        "accuracy_f1_gt_0_5": (
            float(np.mean([score > 0.5 for score in f1_values])) if f1_values else None
        ),
    }
    for key in METRIC_KEYS:
        if key in {"exact", "f1"}:
            continue
        output[key] = mean_available(rows, key)
        output[f"{key}_count"] = sum(1 for row in rows if row.get(key) is not None)
    return output


def summarize_answer_results(
    results: list[dict[str, Any]], latencies: list[float] | None = None
) -> dict[str, Any]:
    latencies = latencies or [
        float(row["latency"]) for row in results if row.get("latency") is not None
    ]
    groups: dict[str, list[dict[str, Any]]] = {
        "overall": results,
        "f1_lt_0_5": [row for row in results if float(row.get("f1", 0.0)) < 0.5],
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_type[row.get("type", "")].append(row)
    for key, rows in by_type.items():
        groups[key or "unknown_type"] = rows

    overall = metric_group(results)
    return {
        "total": len(results),
        "exact_match": overall["exact_match"],
        "mean_f1": overall["mean_f1"],
        "accuracy_f1_gt_0_5": overall["accuracy_f1_gt_0_5"],
        "latency_seconds": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
        "extended_metrics": {
            key: mean_available(results, key)
            for key in [
                "relaxed_exact",
                "answer_contains_expected",
                "expected_contains_answer",
                "numeric_any_match",
                "numeric_all_recall",
                "numeric_precision",
                "numeric_recall",
            ]
        },
        "new_metrics": {
            key: mean_available(results, key)
            for key in [
                "numeric_exact_match",
                "numeric_relaxed_match",
                "unit_match",
                "entity_match",
                "doc_hit_at_k",
                "page_hit_at_k",
                "table_hit_at_k",
            ]
        },
        "by_type": {key: metric_group(rows) for key, rows in sorted(by_type.items())},
        "metric_groups": {key: metric_group(rows) for key, rows in groups.items()},
    }
