from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_vlm_from_page_candidates_clean import (  # noqa: E402
    compute_extended_metrics,
    compute_similarity,
    crop_2x2,
    crop_largest_nonwhite_regions,
    expand_with_neighbors,
    is_visual_context,
    is_visual_question,
    load_images,
    select_candidates,
    summarize,
)


TABLE_TERMS = {
    "table",
    "row",
    "column",
    "cell",
    "accuracy",
    "f1",
    "score",
    "percentage",
    "percent",
    "revenue",
    "sales",
    "profit",
    "income",
    "cost",
    "expense",
    "expenses",
    "assets",
    "debt",
    "employees",
    "tokens",
    "ppl",
    "bleu",
    "auc",
    "wer",
}

VISUAL_TERMS = {
    "figure",
    "fig",
    "chart",
    "graph",
    "diagram",
    "architecture",
    "pipeline",
    "component",
    "arrow",
    "axis",
    "legend",
    "plot",
    "photo",
    "image",
    "picture",
    "shown",
}

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "which",
    "where",
    "when",
    "how",
    "does",
    "did",
    "was",
    "were",
    "are",
    "is",
    "according",
    "based",
}


@dataclass
class LayoutCrop:
    crop_type: str
    bbox: tuple[int, int, int, int]
    score: float
    path: str | None
    page_label: str
    page_rank: int
    ocr_text: str = ""
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLM answers from cached page candidates with an optional "
            "layout-aware crop policy. This script is separate from the baseline evaluator."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=["colpali", "reranked"], default="reranked")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-pages", type=int, default=5)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "text_top3_visual_top5"],
        default="none",
    )
    parser.add_argument("--text-top-pages", type=int, default=3)
    parser.add_argument("--visual-top-pages", type=int, default=5)
    parser.add_argument(
        "--context-policy",
        choices=["raw", "top_folder_vote", "top_folder_vote3"],
        default="raw",
    )
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument("--max-image-long-edge", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-images", type=int, default=5)
    parser.add_argument(
        "--crop-policy",
        choices=[
            "none",
            "top_2x2",
            "visual_2x2",
            "visual_main",
            "layout_aware",
            "layout_aware_v2",
        ],
        default="none",
    )
    parser.add_argument("--crop-top-n", type=int, default=1)
    parser.add_argument(
        "--visual-crop-policy",
        choices=[
            "none",
            "top_2x2",
            "visual_2x2",
            "visual_main",
            "layout_aware",
            "layout_aware_v2",
        ],
        default="visual_main",
    )
    parser.add_argument("--debug-crop-dir", type=Path, default=Path("data/debug_crops"))
    parser.add_argument(
        "--layout-context-mode",
        choices=["crop_only", "full_page_plus_crop"],
        default="crop_only",
        help=(
            "Only used with layout_aware_v2. crop_only sends the selected crop; "
            "full_page_plus_crop sends the selected full page first, then the crop."
        ),
    )
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["concise", "legacy", "think_answer", "smart_universal"],
        default="concise",
    )
    parser.add_argument("--answer-refine", choices=["none", "text"], default="none")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_vlm_layout_aware_from_candidates.json"),
    )
    return parser.parse_args()


def normalize_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def text_similarity(a: str, b: str) -> float:
    left = normalize_tokens(a)
    right = normalize_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def question_target_type(question: str, row: dict[str, Any]) -> str:
    question_lower = (question or "").lower()
    tokens = normalize_tokens(question_lower)
    if "table" in tokens or re.search(r"\btable\s*\d+\b", question_lower):
        return "table"
    if tokens & VISUAL_TERMS or is_visual_context(row):
        return "figure"
    if tokens & TABLE_TERMS:
        return "table"
    return "text"


def question_crop_intent(question: str) -> str:
    question_lower = (question or "").lower()
    explicit = explicit_reference(question)
    if explicit:
        return "table" if explicit["kind"] == "table" else "visual"

    has_table = bool(re.search(r"\btable\b", question_lower))
    has_visual = bool(
        re.search(
            r"\b(figure|fig\.?|chart|graph|plot|diagram|architecture|component|image|picture|photo)\b",
            question_lower,
        )
    )
    if has_table and not has_visual:
        return "table"
    if has_visual and not has_table:
        return "visual"
    return "unknown"


def explicit_reference(question: str) -> dict[str, str] | None:
    match = re.search(r"\b(table|figure|fig\.?)\s*([0-9]+[a-z]?)\b", question or "", re.I)
    if not match:
        return None
    raw_kind = match.group(1).lower().replace(".", "")
    kind = "figure" if raw_kind.startswith("fig") else raw_kind
    number = match.group(2).lower()
    return {"kind": kind, "number": number, "normalized": f"{kind} {number}"}


def explicit_caption_target(question: str) -> str | None:
    reference = explicit_reference(question)
    return reference["normalized"] if reference else None


def allowed_crop_types(intent: str) -> set[str]:
    if intent == "visual":
        return {"figure"}
    if intent == "table":
        return {"table"}
    return {"table", "figure", "text"}


def is_numeric_table_heavy(question: str) -> bool:
    q_tokens = normalize_tokens(question)
    return bool(q_tokens & TABLE_TERMS) or bool(
        re.search(
            r"\b(total|difference|ratio|increase|decrease|percentage|score|accuracy|f1)\b",
            question or "",
            re.I,
        )
    )


def extract_crop_caption(ocr_text: str, crop_type: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (ocr_text or "").splitlines()]
    caption_terms = {
        "table": r"\btable\s*[0-9]+[a-z]?\b",
        "figure": r"\b(fig\.?|figure|chart|graph|diagram)\s*[0-9]+[a-z]?\b",
        "text": r"\b(table|fig\.?|figure|chart|graph|diagram)\s*[0-9]+[a-z]?\b",
    }
    pattern = caption_terms.get(crop_type, caption_terms["text"])
    for line in lines:
        if re.search(pattern, line, re.I):
            return line
    for line in lines:
        if re.search(r"\b(table|fig\.?|figure|chart|graph|diagram)\b", line, re.I):
            return line
    return ""


def caption_reference(caption: str) -> str | None:
    match = re.search(r"\b(table|figure|fig\.?)\s*([0-9]+[a-z]?)\b", caption or "", re.I)
    if not match:
        return None
    raw_kind = match.group(1).lower().replace(".", "")
    kind = "figure" if raw_kind.startswith("fig") else raw_kind
    return f"{kind} {match.group(2).lower()}"


def caption_matches_reference(caption: str, reference: dict[str, str] | None) -> bool:
    if not caption or not reference:
        return False
    return caption_reference(caption) == reference["normalized"]


def caption_conflicts_reference(caption: str, reference: dict[str, str] | None) -> bool:
    if not caption or not reference:
        return False
    found = caption_reference(caption)
    return found is not None and found != reference["normalized"]


def crop_type_mismatch(intent: str, crop_type: str | None) -> bool:
    if not crop_type or intent == "unknown":
        return False
    return crop_type not in allowed_crop_types(intent)


def fallback_policy_for_intent(intent: str, question: str) -> str:
    if intent == "visual" or is_visual_question(question):
        return "visual_main"
    return "none"


def fallback_images_for_intent(
    candidates: list[dict[str, Any]], crop_top_n: int, question: str, intent: str
) -> list[Image.Image]:
    return load_images(
        candidates, fallback_policy_for_intent(intent, question), crop_top_n, question
    )


def safe_ocr(image: Image.Image) -> str:
    if importlib.util.find_spec("pytesseract") is None:
        return ""
    try:
        import pytesseract

        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int, pad: int = 0
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def block_components(mask: np.ndarray, block: int = 10) -> list[tuple[int, int, int, int, int]]:
    grid_h = int(np.ceil(mask.shape[0] / block))
    grid_w = int(np.ceil(mask.shape[1] / block))
    grid = np.zeros((grid_h, grid_w), dtype=bool)
    for gy in range(grid_h):
        for gx in range(grid_w):
            patch = mask[gy * block : (gy + 1) * block, gx * block : (gx + 1) * block]
            if patch.size and float(np.mean(patch)) >= 0.08:
                grid[gy, gx] = True

    visited = np.zeros(grid.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for y in range(grid_h):
        for x in range(grid_w):
            if visited[y, x] or not grid[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx == cx and ny == cy:
                            continue
                        if nx < 0 or ny < 0 or nx >= grid_w or ny >= grid_h:
                            continue
                        if visited[ny, nx] or not grid[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            components.append((area, min_x, min_y, max_x + 1, max_y + 1))
    return components


def table_lineness(gray: np.ndarray) -> float:
    dark = gray < 120
    if dark.size == 0:
        return 0.0
    row_density = np.mean(dark, axis=1)
    col_density = np.mean(dark, axis=0)
    horizontal = float(np.mean(row_density > 0.18))
    vertical = float(np.mean(col_density > 0.12))
    return horizontal + vertical


def classify_region(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[str, float]:
    crop = image.crop(box).convert("RGB")
    if crop.width < 80 or crop.height < 80:
        return "text", 0.0
    small = crop.resize((min(320, crop.width), min(320, crop.height)), Image.Resampling.BILINEAR)
    rgb = np.asarray(small).astype(np.float32)
    gray = np.mean(rgb, axis=2)
    colorfulness = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
    dark_density = float(np.mean(gray < 180))
    line_score = table_lineness(gray)
    aspect = crop.width / max(crop.height, 1)

    if line_score > 0.08 and dark_density > 0.03:
        return "table", min(1.0, line_score * 3.0 + dark_density)
    if colorfulness > 12 or (dark_density > 0.12 and 0.5 <= aspect <= 2.8):
        return "figure", min(1.0, colorfulness / 45.0 + dark_density)
    return "text", min(1.0, dark_density * 2.0)


def detect_layout_regions(image: Image.Image) -> list[tuple[str, tuple[int, int, int, int], float]]:
    width, height = image.size
    scale = max(width, height) / 640
    small_size = (max(1, round(width / scale)), max(1, round(height / scale)))
    small = image.convert("RGB").resize(small_size, Image.Resampling.BILINEAR)
    rgb = np.asarray(small).astype(np.float32)
    gray = np.mean(rgb, axis=2)
    nonwhite = gray < 245

    regions: list[tuple[str, tuple[int, int, int, int], float]] = []
    for area, min_x, min_y, max_x, max_y in block_components(nonwhite, block=10):
        if area < 8:
            continue
        left = int(min_x * 10 * scale)
        top = int(min_y * 10 * scale)
        right = int(max_x * 10 * scale)
        bottom = int(max_y * 10 * scale)
        box = clamp_box((left, top, right, bottom), width, height, pad=24)
        if box[2] - box[0] < 96 or box[3] - box[1] < 96:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > 0.92 * width * height:
            continue
        crop_type, confidence = classify_region(image, box)
        regions.append((crop_type, box, confidence))

    if not regions:
        content_y, content_x = np.where(nonwhite)
        if len(content_x) and len(content_y):
            box = clamp_box(
                (
                    int(content_x.min() * scale),
                    int(content_y.min() * scale),
                    int((content_x.max() + 1) * scale),
                    int((content_y.max() + 1) * scale),
                ),
                width,
                height,
                pad=24,
            )
            crop_type, confidence = classify_region(image, box)
            regions.append((crop_type, box, confidence))

    return regions


def score_layout_crop(
    *,
    question: str,
    target_type: str,
    crop_type: str,
    confidence: float,
    ocr_text: str,
) -> tuple[float, str]:
    score = confidence
    reasons = [f"confidence={confidence:.3f}"]
    if target_type == crop_type:
        score += 1.0
        reasons.append("target_type_match")
    if target_type == "figure" and crop_type in {"figure", "text"}:
        score += 0.25
    if target_type == "table" and crop_type == "text":
        score += 0.15

    sim = text_similarity(question, ocr_text)
    if sim:
        score += 2.0 * sim
        reasons.append(f"ocr_similarity={sim:.3f}")

    caption = explicit_caption_target(question)
    if caption and caption in re.sub(r"\s+", " ", ocr_text.lower()):
        score += 2.5
        reasons.append(f"caption_match={caption}")

    q_tokens = normalize_tokens(question)
    if q_tokens & TABLE_TERMS and crop_type == "table":
        score += 0.5
        reasons.append("table_keyword_bonus")
    if q_tokens & VISUAL_TERMS and crop_type == "figure":
        score += 0.5
        reasons.append("visual_keyword_bonus")
    return score, ";".join(reasons)


def score_layout_crop_v2(
    *,
    question: str,
    intent: str,
    explicit_ref: dict[str, str] | None,
    crop_type: str,
    confidence: float,
    ocr_text: str,
    caption: str,
    page_rank: int,
) -> tuple[float | None, str, bool, bool]:
    if crop_type not in allowed_crop_types(intent):
        return None, "rejected_type_not_allowed", False, True

    caption_match = caption_matches_reference(caption, explicit_ref)
    caption_conflict = caption_conflicts_reference(caption, explicit_ref)
    if caption_conflict:
        return None, "rejected_caption_conflict", False, crop_type_mismatch(intent, crop_type)

    score = confidence
    reasons = [f"confidence={confidence:.3f}"]

    if intent != "unknown":
        score += 1.5
        reasons.append("intent_type_match")
    elif is_numeric_table_heavy(question) and crop_type == "table":
        score += 0.7
        reasons.append("unknown_numeric_table_bonus")

    page_penalty = 0.18 * max(0, page_rank - 1)
    score -= page_penalty
    reasons.append(f"page_rank_penalty={page_penalty:.3f}")

    sim = text_similarity(question, ocr_text)
    if sim:
        score += 1.5 * sim
        reasons.append(f"ocr_similarity={sim:.3f}")

    if caption_match:
        score += 5.0
        reasons.append(f"caption_match={explicit_ref['normalized']}")
    elif explicit_ref and caption:
        score -= 2.0
        reasons.append("explicit_ref_without_match_penalty")
    elif explicit_ref and not caption:
        score -= 0.75
        reasons.append("missing_caption_penalty")

    if explicit_ref and page_rank == 1 and caption_match:
        score += 2.0
        reasons.append("top1_caption_match_bonus")

    q_tokens = normalize_tokens(question)
    if q_tokens & TABLE_TERMS and crop_type == "table":
        score += 0.4
        reasons.append("table_keyword_bonus")
    if q_tokens & VISUAL_TERMS and crop_type == "figure":
        score += 0.4
        reasons.append("visual_keyword_bonus")

    mismatch = crop_type_mismatch(intent, crop_type)
    return score, ";".join(reasons), caption_match, mismatch


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    output_dir: Path,
    question_index: int,
    page_label: str,
    crop_type: str,
) -> tuple[Image.Image, str]:
    crop = image.crop(box).convert("RGB")
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", page_label)
    path = output_dir / f"q{question_index:04d}_{safe_label}_{crop_type}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)
    return crop, str(path)


def layout_aware_images(
    *,
    candidates: list[dict[str, Any]],
    question: str,
    row: dict[str, Any],
    crop_top_n: int,
    debug_crop_dir: Path,
    question_index: int,
) -> tuple[list[Image.Image], list[dict[str, Any]], dict[str, Any] | None]:
    pages: list[tuple[dict[str, Any], Image.Image]] = []
    for candidate in candidates:
        try:
            with Image.open(candidate["path"]) as img:
                pages.append((candidate, img.convert("RGB").copy()))
        except OSError as exc:
            print(f"    [WARN] failed to load image {candidate.get('path')}: {exc}")

    if not pages:
        return [], [], None

    target_type = question_target_type(question, row)
    all_debug: list[dict[str, Any]] = []
    best: tuple[float, LayoutCrop, Image.Image] | None = None

    for page_rank, (candidate, page_image) in enumerate(pages, start=1):
        page_label = f"{candidate.get('folder')}_{candidate.get('page')}"
        for crop_type, box, confidence in detect_layout_regions(page_image):
            crop_preview = page_image.crop(box).convert("RGB")
            ocr_text = safe_ocr(crop_preview)
            score, reason = score_layout_crop(
                question=question,
                target_type=target_type,
                crop_type=crop_type,
                confidence=confidence,
                ocr_text=ocr_text,
            )
            debug_item = LayoutCrop(
                crop_type=crop_type,
                bbox=box,
                score=score,
                path=None,
                page_label=page_label,
                page_rank=page_rank,
                ocr_text=ocr_text[:500],
                reason=reason,
            )
            all_debug.append(
                {
                    "crop_type": debug_item.crop_type,
                    "bbox": list(debug_item.bbox),
                    "crop_score": debug_item.score,
                    "crop_path": debug_item.path,
                    "page_label": debug_item.page_label,
                    "page_rank": debug_item.page_rank,
                    "ocr_text": debug_item.ocr_text,
                    "reason": debug_item.reason,
                }
            )
            if best is None or score > best[0]:
                best = (score, debug_item, page_image)

    if best is None:
        fallback = load_images(candidates, "visual_main", crop_top_n, question)
        return fallback, all_debug, None

    _, selected, selected_page = best
    crop_image, crop_path = save_crop(
        selected_page,
        selected.bbox,
        output_dir=debug_crop_dir,
        question_index=question_index,
        page_label=selected.page_label,
        crop_type=selected.crop_type,
    )
    selected.path = crop_path

    selected_debug = {
        "crop_type": selected.crop_type,
        "bbox": list(selected.bbox),
        "crop_score": selected.score,
        "crop_path": selected.path,
        "page_label": selected.page_label,
        "page_rank": selected.page_rank,
        "ocr_text": selected.ocr_text,
        "reason": selected.reason,
        "target_type": target_type,
    }
    for item in all_debug:
        if (
            item["bbox"] == selected_debug["bbox"]
            and item["page_label"] == selected_debug["page_label"]
        ):
            item["crop_path"] = crop_path

    selected_page_idx = max(0, selected.page_rank - 1)
    ordered_full_pages = [pages[selected_page_idx][1]]
    ordered_full_pages.extend(
        page for idx, (_candidate, page) in enumerate(pages) if idx != selected_page_idx
    )

    # Put the selected full page and selected crop first so both survive max_images truncation.
    return [ordered_full_pages[0], crop_image, *ordered_full_pages[1:]], all_debug, selected_debug


def layout_aware_images_v2(
    *,
    candidates: list[dict[str, Any]],
    question: str,
    row: dict[str, Any],
    crop_top_n: int,
    debug_crop_dir: Path,
    question_index: int,
    layout_context_mode: str,
) -> tuple[list[Image.Image], list[dict[str, Any]], dict[str, Any] | None]:
    pages: list[tuple[dict[str, Any], Image.Image]] = []
    for candidate in candidates:
        try:
            with Image.open(candidate["path"]) as img:
                pages.append((candidate, img.convert("RGB").copy()))
        except OSError as exc:
            print(f"    [WARN] failed to load image {candidate.get('path')}: {exc}")

    intent = question_crop_intent(question)
    explicit_ref = explicit_reference(question)
    all_debug: list[dict[str, Any]] = []

    if not pages:
        return (
            [],
            all_debug,
            {
                "question_crop_intent": intent,
                "explicit_reference": explicit_ref["normalized"] if explicit_ref else None,
                "fallback_used": True,
                "fallback_reason": "no_pages",
                "crop_used": False,
                "full_page_plus_crop": False,
                "crop_type_mismatch": False,
                "caption_match": False,
            },
        )

    best: tuple[float, LayoutCrop, Image.Image, str, bool, bool] | None = None
    for page_rank, (candidate, page_image) in enumerate(pages, start=1):
        page_label = f"{candidate.get('folder')}_{candidate.get('page')}"
        for crop_type, box, confidence in detect_layout_regions(page_image):
            crop_preview = page_image.crop(box).convert("RGB")
            ocr_text = safe_ocr(crop_preview)
            caption = extract_crop_caption(ocr_text, crop_type)
            score, reason, caption_match, mismatch = score_layout_crop_v2(
                question=question,
                intent=intent,
                explicit_ref=explicit_ref,
                crop_type=crop_type,
                confidence=confidence,
                ocr_text=ocr_text,
                caption=caption,
                page_rank=page_rank,
            )
            debug_row = {
                "question_crop_intent": intent,
                "explicit_reference": explicit_ref["normalized"] if explicit_ref else None,
                "crop_type": crop_type,
                "bbox": list(box),
                "crop_score": score,
                "crop_path": None,
                "page_label": page_label,
                "page_rank": page_rank,
                "ocr_text": ocr_text[:500],
                "selected_crop_caption": caption,
                "caption_match": caption_match,
                "crop_type_mismatch": mismatch,
                "rejected": score is None,
                "reason": reason,
            }
            all_debug.append(debug_row)
            if score is None:
                continue

            debug_item = LayoutCrop(
                crop_type=crop_type,
                bbox=box,
                score=score,
                path=None,
                page_label=page_label,
                page_rank=page_rank,
                ocr_text=ocr_text[:500],
                reason=reason,
            )
            if best is None or score > best[0]:
                best = (score, debug_item, page_image, caption, caption_match, mismatch)

    min_score = 1.15
    if explicit_ref:
        min_score = 2.25
    if best is None or best[0] < min_score:
        fallback_images = fallback_images_for_intent(candidates, crop_top_n, question, intent)
        selected_debug = {
            "question_crop_intent": intent,
            "explicit_reference": explicit_ref["normalized"] if explicit_ref else None,
            "crop_type": None,
            "selected_crop_type": None,
            "selected_crop_caption": None,
            "crop_score": None,
            "selected_crop_score": None,
            "crop_path": None,
            "bbox": None,
            "page_label": None,
            "page_rank": None,
            "caption_match": False,
            "crop_type_mismatch": False,
            "fallback_used": True,
            "fallback_reason": "no_crop_above_threshold",
            "crop_used": False,
            "full_page_plus_crop": False,
        }
        return fallback_images, all_debug, selected_debug

    _score, selected, selected_page, caption, caption_match, mismatch = best
    crop_image, crop_path = save_crop(
        selected_page,
        selected.bbox,
        output_dir=debug_crop_dir,
        question_index=question_index,
        page_label=selected.page_label,
        crop_type=selected.crop_type,
    )
    selected.path = crop_path

    selected_debug = {
        "question_crop_intent": intent,
        "explicit_reference": explicit_ref["normalized"] if explicit_ref else None,
        "crop_type": selected.crop_type,
        "selected_crop_type": selected.crop_type,
        "selected_crop_caption": caption,
        "crop_score": selected.score,
        "selected_crop_score": selected.score,
        "crop_path": selected.path,
        "bbox": list(selected.bbox),
        "page_label": selected.page_label,
        "page_rank": selected.page_rank,
        "ocr_text": selected.ocr_text,
        "reason": selected.reason,
        "caption_match": caption_match,
        "crop_type_mismatch": mismatch,
        "fallback_used": False,
        "fallback_reason": None,
        "crop_used": True,
        "full_page_plus_crop": layout_context_mode == "full_page_plus_crop",
    }
    for item in all_debug:
        if (
            item["bbox"] == selected_debug["bbox"]
            and item["page_label"] == selected_debug["page_label"]
        ):
            item["crop_path"] = crop_path

    selected_page_idx = max(0, selected.page_rank - 1)
    ordered_full_pages = [pages[selected_page_idx][1]]
    ordered_full_pages.extend(
        page for idx, (_candidate, page) in enumerate(pages) if idx != selected_page_idx
    )
    if layout_context_mode == "full_page_plus_crop":
        return (
            [ordered_full_pages[0], crop_image, *ordered_full_pages[1:]],
            all_debug,
            selected_debug,
        )
    return [crop_image], all_debug, selected_debug


def effective_context_settings_layout(
    args: argparse.Namespace, row: dict[str, Any]
) -> tuple[int, str, bool]:
    visual = is_visual_context(row)
    if args.adaptive_policy == "text_top3_visual_top5":
        if visual:
            return args.visual_top_pages, args.visual_crop_policy, visual
        if args.visual_crop_policy in {"layout_aware", "layout_aware_v2"}:
            if args.visual_crop_policy == "layout_aware_v2":
                if question_crop_intent(row.get("question", "")) in {"table", "visual", "unknown"}:
                    return args.text_top_pages, "layout_aware_v2", visual
            elif question_target_type(row.get("question", ""), row) in {"table", "figure"}:
                return args.text_top_pages, "layout_aware", visual
        return args.text_top_pages, "none", visual
    return args.top_pages, args.crop_policy, visual


def load_context_images(
    *,
    candidates: list[dict[str, Any]],
    crop_policy: str,
    crop_top_n: int,
    question: str,
    row: dict[str, Any],
    debug_crop_dir: Path,
    question_index: int,
    layout_context_mode: str,
) -> tuple[list[Image.Image], list[dict[str, Any]], dict[str, Any] | None]:
    if crop_policy == "layout_aware_v2":
        return layout_aware_images_v2(
            candidates=candidates,
            question=question,
            row=row,
            crop_top_n=crop_top_n,
            debug_crop_dir=debug_crop_dir,
            question_index=question_index,
            layout_context_mode=layout_context_mode,
        )
    if crop_policy == "layout_aware":
        return layout_aware_images(
            candidates=candidates,
            question=question,
            row=row,
            crop_top_n=crop_top_n,
            debug_crop_dir=debug_crop_dir,
            question_index=question_index,
        )
    images = load_images(candidates, crop_policy, crop_top_n, question)
    return images, [], None


def create_generator(args: argparse.Namespace) -> Any:
    if args.prompt_style == "smart_universal":
        from src.core.generators.qwen_vl_generator_promt_style import create_table_generator
    else:
        from src.core.generators.qwen_vl_generator import create_table_generator

    return create_table_generator(
        device=args.device,
        max_image_long_edge=args.max_image_long_edge,
        load_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        do_sample=args.do_sample,
        max_images=args.max_context_images,
        answer_refine=args.answer_refine,
    )


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "type",
        "expected",
        "generated",
        "f1",
        "latency",
        "effective_crop_policy",
        "selected_crop_type",
        "selected_crop_score",
        "selected_crop_path",
        "selected_crop_bbox",
        "selected_crop_page",
        "question_crop_intent",
        "explicit_reference",
        "selected_crop_caption",
        "crop_type_mismatch",
        "caption_match",
        "fallback_used",
        "crop_used",
        "full_page_plus_crop",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    selected = row.get("layout_aware_selected_crop") or {}
    return {
        "question": row.get("question"),
        "type": row.get("type"),
        "expected": row.get("expected"),
        "generated": row.get("generated"),
        "f1": row.get("f1"),
        "latency": row.get("latency"),
        "effective_crop_policy": row.get("effective_crop_policy"),
        "selected_crop_type": selected.get("crop_type"),
        "selected_crop_score": selected.get("crop_score"),
        "selected_crop_path": selected.get("crop_path"),
        "selected_crop_bbox": selected.get("bbox"),
        "selected_crop_page": selected.get("page_label"),
        "question_crop_intent": selected.get("question_crop_intent"),
        "explicit_reference": selected.get("explicit_reference"),
        "selected_crop_caption": selected.get("selected_crop_caption"),
        "crop_type_mismatch": selected.get("crop_type_mismatch"),
        "caption_match": selected.get("caption_match"),
        "fallback_used": selected.get("fallback_used"),
        "crop_used": selected.get("crop_used"),
        "full_page_plus_crop": selected.get("full_page_plus_crop"),
    }


def summarize_with_crop_metrics(
    results: list[dict[str, Any]], latencies: list[float]
) -> dict[str, Any]:
    summary = summarize(results, latencies)
    total = max(len(results), 1)
    selected = [row.get("layout_aware_selected_crop") or {} for row in results]
    crop_used = [not item.get("fallback_used") and bool(item.get("crop_path")) for item in selected]
    mismatches = [bool(item.get("crop_type_mismatch")) for item in selected]
    caption_matches = [bool(item.get("caption_match")) for item in selected]
    fallbacks = [bool(item.get("fallback_used")) for item in selected]

    mismatch_rows = [row for row, flag in zip(results, mismatches) if flag]
    caption_match_rows = [row for row, flag in zip(results, caption_matches) if flag]
    summary["crop_debug_metrics"] = {
        "crop_used_rate": float(sum(crop_used) / total),
        "crop_type_mismatch_rate": float(sum(mismatches) / total),
        "caption_match_rate": float(sum(caption_matches) / total),
        "fallback_rate": float(sum(fallbacks) / total),
        "mean_f1_when_crop_type_mismatch_true": (
            float(np.mean([row["f1"] for row in mismatch_rows])) if mismatch_rows else None
        ),
        "mean_f1_when_caption_match_true": (
            float(np.mean([row["f1"] for row in caption_match_rows]))
            if caption_match_rows
            else None
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    generator = create_generator(args)

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") or payload.get("results") or []
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    debug_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1 + args.start):
        expected = row.get("answer") or row.get("expected") or ""
        effective_top_pages, effective_crop_policy, visual_context = (
            effective_context_settings_layout(args, row)
        )
        candidates = select_candidates(row, args.mode, effective_top_pages, args.context_policy)
        candidates = expand_with_neighbors(candidates, args.neighbor_radius)
        images, layout_debug, selected_crop = load_context_images(
            candidates=candidates,
            crop_policy=effective_crop_policy,
            crop_top_n=args.crop_top_n,
            question=row["question"],
            row=row,
            debug_crop_dir=args.debug_crop_dir,
            question_index=idx,
            layout_context_mode=args.layout_context_mode,
        )

        print(f"\n[{idx}/{args.start + len(rows)}] {row['question'][:100]}")
        page_labels = [
            f"{candidate.get('folder')}/{candidate.get('page')}" for candidate in candidates[:10]
        ]
        print(f"    pages={page_labels}")
        print(
            f"    context=top{effective_top_pages} crop={effective_crop_policy} visual={visual_context}"
        )
        if selected_crop:
            crop_score = selected_crop.get("crop_score")
            crop_score_text = f"{crop_score:.3f}" if isinstance(crop_score, int | float) else "None"
            print(
                "    layout_crop="
                f"{selected_crop.get('crop_type')} {selected_crop.get('page_label')} "
                f"score={crop_score_text} path={selected_crop.get('crop_path')} "
                f"fallback={selected_crop.get('fallback_used')}"
            )

        start_time = time.time()
        try:
            answer = generator.generate_answer(row["question"], images)
        except Exception as exc:
            print(f"    [ERROR] VLM failed: {exc}")
            generator.last_raw_output = ""
            generator.last_reasoning = ""
            generator.last_answer = "ERROR"
            answer = "ERROR"
        latency = time.time() - start_time
        latencies.append(latency)

        exact, f1 = compute_similarity(answer, expected)
        extended_metrics = compute_extended_metrics(answer, expected)
        result = {
            "question": row["question"],
            "expected": expected,
            "generated": answer,
            "raw_generated": getattr(generator, "last_raw_output", answer),
            "vlm_think": getattr(generator, "last_reasoning", ""),
            "exact": exact,
            "f1": f1,
            "latency": latency,
            "type": row.get("type", ""),
            "expected_folder": row.get("expected_folder"),
            "effective_top_pages": effective_top_pages,
            "effective_crop_policy": effective_crop_policy,
            "visual_context": visual_context,
            "oracle_pages": row.get("oracle_pages", []),
            "pages": candidates,
            "layout_aware_debug": layout_debug,
            "layout_aware_selected_crop": selected_crop,
            "question_crop_intent": (selected_crop or {}).get("question_crop_intent"),
            "explicit_reference": (selected_crop or {}).get("explicit_reference"),
            "selected_crop_type": (selected_crop or {}).get("selected_crop_type")
            or (selected_crop or {}).get("crop_type"),
            "selected_crop_caption": (selected_crop or {}).get("selected_crop_caption"),
            "selected_crop_score": (selected_crop or {}).get("selected_crop_score")
            or (selected_crop or {}).get("crop_score"),
            "crop_type_mismatch": (selected_crop or {}).get("crop_type_mismatch", False),
            "caption_match": (selected_crop or {}).get("caption_match", False),
            "fallback_used": (selected_crop or {}).get("fallback_used", selected_crop is None),
            "crop_used": (selected_crop or {}).get("crop_used", False),
            "full_page_plus_crop": (selected_crop or {}).get("full_page_plus_crop", False),
            "crop_path": (selected_crop or {}).get("crop_path"),
            **extended_metrics,
        }
        results.append(result)
        debug_rows.append(compact_case(result))

        if args.print_think:
            think = (result["vlm_think"] or result["raw_generated"] or "").replace("\n", " ")
            print(f"    think={think[:500]}")
        print(f"    generated={answer[:240]}")
        print(f"    expected={expected}")
        print(f"    exact={exact} f1={f1:.3f} latency={latency:.2f}s")

        partial = {
            "summary": summarize_with_crop_metrics(results, latencies),
            "config": vars(args),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(partial, handle, indent=2, ensure_ascii=False, default=str)
        write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)

    output = {
        "summary": summarize_with_crop_metrics(results, latencies),
        "config": vars(args),
        "source_summary": payload.get("summary"),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)
    write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Debug crops: {args.debug_crop_dir}")


if __name__ == "__main__":
    main()
