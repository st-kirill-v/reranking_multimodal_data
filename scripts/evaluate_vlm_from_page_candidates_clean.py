from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL answers from ColPali or reranked page candidates."
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
        choices=["none", "top_2x2", "visual_2x2", "visual_main"],
        default="none",
    )
    parser.add_argument("--crop-top-n", type=int, default=1)
    parser.add_argument(
        "--visual-crop-policy",
        choices=["none", "top_2x2", "visual_2x2", "visual_main"],
        default="visual_main",
    )
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["concise", "legacy", "think_answer"],
        default="concise",
    )
    parser.add_argument("--answer-refine", choices=["none", "text"], default="none")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/eval_vlm_from_candidates.json"))
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_number_token(token: str) -> str:
    token = token.lower().replace(",", "").strip()
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
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: str) -> list[str]:
    return [
        normalize_number_token(match.group(0))
        for match in re.finditer(r"[$₹€£]?\d[\d,]*(?:\.\d+)?%?", text or "")
    ]


def compute_extended_metrics(generated: str, expected: str) -> dict[str, float]:
    gen_norm = normalize_answer(generated)
    exp_norm = normalize_answer(expected)
    gen_compact = re.sub(r"[^a-z0-9]+", "", gen_norm)
    exp_compact = re.sub(r"[^a-z0-9]+", "", exp_norm)
    gen_numbers = extract_numbers(generated)
    exp_numbers = extract_numbers(expected)

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
    }


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
        w for w in re.findall(r"[a-z][a-z0-9_\-]*", gen) if w not in stop_words and len(w) > 2
    )
    exp_words = set(
        w for w in re.findall(r"[a-z][a-z0-9_\-]*", exp) if w not in stop_words and len(w) > 2
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

    if exp_numbers:
        f1 = 0.4 * word_f1 + 0.6 * num_f1
    else:
        f1 = word_f1
    return exact, f1


def select_candidates(
    row: dict[str, Any], mode: str, top_pages: int, context_policy: str
) -> list[dict[str, Any]]:
    if mode == "reranked":
        candidates = row.get("top10_reranked") or row.get("reranked") or []
    else:
        candidates = row.get("candidates") or row.get("top10_prerank") or []
    if context_policy == "raw":
        return candidates[:top_pages]

    seed = candidates[:top_pages]
    folder_counts = Counter(str(candidate.get("folder")) for candidate in seed)
    if not folder_counts:
        return seed
    first_rank = {
        str(candidate.get("folder")): idx for idx, candidate in reversed(list(enumerate(seed)))
    }
    winning_folder = min(
        folder_counts,
        key=lambda folder: (-folder_counts[folder], first_rank.get(folder, top_pages)),
    )
    if context_policy == "top_folder_vote3" and folder_counts[winning_folder] < 3:
        return seed

    selected = []
    seen = set()
    for candidate in candidates:
        if str(candidate.get("folder")) != winning_folder:
            continue
        key = candidate.get("path") or (candidate.get("folder"), candidate.get("page"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= top_pages:
            return selected

    for candidate in seed:
        key = candidate.get("path") or (candidate.get("folder"), candidate.get("page"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= top_pages:
            break
    return selected


def neighbor_path(path: Path, page: int, delta: int) -> Path:
    return path.with_name(f"page_{page + delta}.png")


def expand_with_neighbors(candidates: list[dict[str, Any]], radius: int) -> list[dict[str, Any]]:
    if radius <= 0:
        return candidates

    expanded = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate["path"])
        page = int(candidate["page"])
        for delta in range(-radius, radius + 1):
            page_path = neighbor_path(path, page, delta)
            if not page_path.exists():
                continue
            key = str(page_path)
            if key in seen:
                continue
            seen.add(key)
            item = dict(candidate)
            item["path"] = str(page_path)
            item["page"] = page + delta
            item["neighbor_delta"] = delta
            expanded.append(item)
    return expanded


def crop_2x2(img: Image.Image) -> list[Image.Image]:
    width, height = img.size
    mid_x = width // 2
    mid_y = height // 2
    boxes = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, width, mid_y),
        (0, mid_y, mid_x, height),
        (mid_x, mid_y, width, height),
    ]
    return [img.crop(box) for box in boxes]


def crop_largest_nonwhite_regions(img: Image.Image, max_regions: int = 2) -> list[Image.Image]:
    width, height = img.size
    scale = max(width, height) / 320
    small_size = (max(1, round(width / scale)), max(1, round(height / scale)))
    small = img.convert("RGB").resize(small_size, Image.Resampling.BILINEAR)
    rgb = np.asarray(small).astype(np.float32)
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    # Prefer photographic/visual regions over black text by looking for sustained color variation.
    nonwhite = ((max_channel - min_channel) > 10) & (max_channel < 248) & (max_channel > 20)

    block = 8
    grid_h = int(np.ceil(nonwhite.shape[0] / block))
    grid_w = int(np.ceil(nonwhite.shape[1] / block))
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for gy in range(grid_h):
        for gx in range(grid_w):
            patch = nonwhite[gy * block : (gy + 1) * block, gx * block : (gx + 1) * block]
            if patch.size and float(np.mean(patch)) >= 0.18:
                mask[gy, gx] = True

    visited = np.zeros(mask.shape, dtype=bool)
    regions = []
    grid_h, grid_w = mask.shape
    for y in range(grid_h):
        for x in range(grid_w):
            if visited[y, x] or not mask[y, x]:
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
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1
            if area < 6 or box_w < 3 or box_h < 3:
                continue
            if box_w * box_h > 0.85 * grid_w * grid_h:
                continue
            regions.append((area, min_x, min_y, max_x + 1, max_y + 1))

    crops = []
    for _, min_x, min_y, max_x, max_y in sorted(regions, reverse=True)[:max_regions]:
        pad = 2
        left = max(0, int((min_x - pad) * block * scale))
        top = max(0, int((min_y - pad) * block * scale))
        right = min(width, int((max_x + pad) * block * scale))
        bottom = min(height, int((max_y + pad) * block * scale))
        if right - left < 64 or bottom - top < 64:
            continue
        crops.append(img.crop((left, top, right, bottom)))
    return crops


def is_visual_question(question: str) -> bool:
    question = (question or "").lower()
    visual_terms = [
        "photo",
        "picture",
        "image",
        "shown",
        "standing",
        "soldiers",
        "visual",
        "figure",
        "chart",
        "graph",
        "diagram",
    ]
    return any(term in question for term in visual_terms)


def is_visual_context(row: dict[str, Any]) -> bool:
    return row.get("type") == "multimodal-f" or is_visual_question(row.get("question", ""))


def effective_context_settings(
    args: argparse.Namespace, row: dict[str, Any]
) -> tuple[int, str, bool]:
    visual = is_visual_context(row)
    if args.adaptive_policy == "text_top3_visual_top5":
        if visual:
            return args.visual_top_pages, args.visual_crop_policy, visual
        return args.text_top_pages, "none", visual
    return args.top_pages, args.crop_policy, visual


def load_images(
    candidates: list[dict[str, Any]], crop_policy: str, crop_top_n: int, question: str
) -> list[Image.Image]:
    full_pages = []
    crops = []
    for candidate in candidates:
        try:
            with Image.open(candidate["path"]) as img:
                full_pages.append(img.convert("RGB").copy())
        except OSError as exc:
            print(f"    [WARN] failed to load image {candidate.get('path')}: {exc}")
    should_crop = crop_policy == "top_2x2" or (
        crop_policy == "visual_2x2" and is_visual_question(question)
    )
    if should_crop:
        for img in full_pages[: max(0, crop_top_n)]:
            crops.extend(crop_2x2(img))
    if crop_policy == "visual_main" and is_visual_question(question):
        for img in full_pages[: max(0, crop_top_n)]:
            crops.extend(crop_largest_nonwhite_regions(img))
        return crops + full_pages
    return full_pages + crops


def summarize(results: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    exact = [row["exact"] for row in results]
    f1 = [row["f1"] for row in results]
    by_type: dict[str, list[float]] = defaultdict(list)
    metric_keys = [
        "relaxed_exact",
        "answer_contains_expected",
        "expected_contains_answer",
        "numeric_any_match",
        "numeric_all_recall",
        "numeric_precision",
        "numeric_recall",
    ]
    for row in results:
        by_type[row["type"]].append(row["f1"])

    return {
        "total": len(results),
        "exact_match": float(np.mean(exact)) if exact else None,
        "mean_f1": float(np.mean(f1)) if f1 else None,
        "accuracy_f1_gt_0_5": float(np.mean([score > 0.5 for score in f1])) if f1 else None,
        "latency_seconds": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
        "extended_metrics": {
            key: float(np.mean([row.get(key, 0.0) for row in results])) for key in metric_keys
        },
        "by_type": {
            key: {
                "count": len(scores),
                "mean_f1": float(np.mean(scores)) if scores else None,
                "accuracy_f1_gt_0_5": (
                    float(np.mean([score > 0.5 for score in scores])) if scores else None
                ),
                "extended_metrics": {
                    metric_key: float(
                        np.mean([row.get(metric_key, 0.0) for row in results if row["type"] == key])
                    )
                    for metric_key in metric_keys
                },
            }
            for key, scores in sorted(by_type.items())
        },
    }


def main() -> None:
    args = parse_args()
    from src.core.generators.qwen_vl_generator import create_table_generator

    with args.input.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload["rows"]
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]

    generator = create_table_generator(
        device=args.device,
        max_image_long_edge=args.max_image_long_edge,
        load_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        do_sample=args.do_sample,
        max_images=args.max_context_images,
        answer_refine=args.answer_refine,
    )

    results = []
    latencies = []
    for idx, row in enumerate(rows, start=1 + args.start):
        effective_top_pages, effective_crop_policy, visual_context = effective_context_settings(
            args, row
        )
        candidates = select_candidates(row, args.mode, effective_top_pages, args.context_policy)
        candidates = expand_with_neighbors(candidates, args.neighbor_radius)
        images = load_images(candidates, effective_crop_policy, args.crop_top_n, row["question"])
        print(f"\n[{idx}/{args.start + len(rows)}] {row['question'][:100]}")
        page_labels = [
            f"{candidate.get('folder')}/{candidate.get('page')}" for candidate in candidates[:10]
        ]
        print(f"    pages={page_labels}")
        if args.adaptive_policy != "none":
            print(
                f"    context=top{effective_top_pages} crop={effective_crop_policy} visual={visual_context}"
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

        exact, f1 = compute_similarity(answer, row["answer"])
        extended_metrics = compute_extended_metrics(answer, row["answer"])
        result = {
            "question": row["question"],
            "expected": row["answer"],
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
            "pages": candidates,
            **extended_metrics,
        }
        results.append(result)
        if args.print_think:
            think = (result["vlm_think"] or result["raw_generated"] or "").replace("\n", " ")
            print(f"    think={think[:500]}")
        print(f"    generated={answer[:240]}")
        print(f"    expected={row['answer']}")
        print(f"    exact={exact} f1={f1:.3f} latency={latency:.2f}s")

        partial = {
            "summary": summarize(results, latencies),
            "config": {
                "input": str(args.input),
                "mode": args.mode,
                "top_pages": args.top_pages,
                "adaptive_policy": args.adaptive_policy,
                "text_top_pages": args.text_top_pages,
                "visual_top_pages": args.visual_top_pages,
                "context_policy": args.context_policy,
                "neighbor_radius": args.neighbor_radius,
                "max_image_long_edge": args.max_image_long_edge,
                "max_new_tokens": args.max_new_tokens,
                "max_context_images": args.max_context_images,
                "crop_policy": args.crop_policy,
                "crop_top_n": args.crop_top_n,
                "visual_crop_policy": args.visual_crop_policy,
                "print_think": args.print_think,
                "prompt_style": args.prompt_style,
                "answer_refine": args.answer_refine,
                "do_sample": args.do_sample,
                "load_4bit": not args.no_4bit,
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(partial, fh, indent=2, ensure_ascii=False)

    output = {
        "summary": summarize(results, latencies),
        "config": {
            "input": str(args.input),
            "mode": args.mode,
            "top_pages": args.top_pages,
            "adaptive_policy": args.adaptive_policy,
            "text_top_pages": args.text_top_pages,
            "visual_top_pages": args.visual_top_pages,
            "context_policy": args.context_policy,
            "neighbor_radius": args.neighbor_radius,
            "max_image_long_edge": args.max_image_long_edge,
            "max_new_tokens": args.max_new_tokens,
            "max_context_images": args.max_context_images,
            "crop_policy": args.crop_policy,
            "crop_top_n": args.crop_top_n,
            "visual_crop_policy": args.visual_crop_policy,
            "print_think": args.print_think,
            "prompt_style": args.prompt_style,
            "answer_refine": args.answer_refine,
            "do_sample": args.do_sample,
            "load_4bit": not args.no_4bit,
        },
        "source_summary": payload.get("summary"),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
