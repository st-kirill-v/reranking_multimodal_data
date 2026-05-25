from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cropping.layout_aware_eval import (  # noqa: E402
    compact_case,
    compute_extended_metrics,
    compute_similarity,
    create_generator,
    effective_context_settings_layout,
    load_context_images,
    summarize_with_crop_metrics,
    write_case_csv,
)
from src.retrieval.colvision import (  # noqa: E402
    encode_query,
    find_answer_pages,
    load_model_and_processor,
    score_docs,
)
from src.mmrag.dataset import load_docbench_questions  # noqa: E402
from src.mmrag.schema import RetrievalCandidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the honest full DocBench pipeline: question -> page retrieval -> "
            "Nemotron reranking -> layout-aware context -> Qwen VLM answer -> metrics."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument(
        "--types",
        nargs="*",
        default=["multimodal-t", "multimodal-f"],
        help='Question types to evaluate, or "all" for the full dataset.',
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3_merged"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_merged_clean")
    parser.add_argument("--retriever-model-id", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--retrieval-device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=1)

    parser.add_argument("--first-stage-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--rerank-device", default="cuda")
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--neighbor-radius", type=int, default=0)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-long-edge", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-images", type=int, default=5)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["concise", "legacy", "think_answer", "smart_universal"],
        default="concise",
    )
    parser.add_argument("--answer-refine", choices=["none", "text"], default="none")

    parser.add_argument("--top-pages", type=int, default=5)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "text_top3_visual_top5"],
        default="text_top3_visual_top5",
    )
    parser.add_argument("--text-top-pages", type=int, default=3)
    parser.add_argument("--visual-top-pages", type=int, default=5)
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
        default="layout_aware_v2",
    )
    parser.add_argument(
        "--layout-context-mode",
        choices=["crop_only", "full_page_plus_crop"],
        default="full_page_plus_crop",
    )
    parser.add_argument(
        "--debug-crop-dir",
        type=Path,
        default=Path("data/debug_crops/full_pipeline_layout_aware_v2"),
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_full_pipeline_layout_aware_308.json"),
    )
    return parser.parse_args()


class ColVisionRetriever:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
        self.manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing ColVision metadata: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.manifest = {}
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        self.records_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.metadata:
            self.records_by_shard[record["shard"]].append(record)
        self.model, self.processor, self.family = load_model_and_processor(
            args.retriever_model_id, args.retrieval_device
        )

    def search(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        import torch

        query_embedding = encode_query(
            self.model, self.processor, query, self.args.retrieval_device
        )
        heap: list[tuple[float, int, dict[str, Any]]] = []
        for shard_name, shard_records in self.records_by_shard.items():
            shard_embeddings = torch.load(
                self.args.index_dir / "shards" / shard_name, map_location="cpu"
            )
            for offset in range(0, len(shard_records), self.args.score_batch_size):
                batch_records = shard_records[offset : offset + self.args.score_batch_size]
                docs = [shard_embeddings[int(record["shard_offset"])] for record in batch_records]
                scores = score_docs(
                    self.processor,
                    query_embedding,
                    docs,
                    self.args.retrieval_device,
                )
                for score, record in zip(scores, batch_records):
                    item = (float(score), int(record["index"]), record)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            del shard_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        candidates = []
        for rank, (score, _, record) in enumerate(
            sorted(heap, key=lambda item: item[0], reverse=True),
            start=1,
        ):
            candidates.append(
                RetrievalCandidate(
                    folder=str(record["folder"]),
                    page=int(record["page"]),
                    path=Path(record["path"]),
                    score=float(score),
                    rank=rank,
                    index=int(record["index"]),
                    source=self.family,
                )
            )
        return candidates


def candidate_to_json(candidate: RetrievalCandidate, rank: int | None = None) -> dict[str, Any]:
    row = candidate.to_json()
    if rank is not None:
        row["rank"] = rank
    return row


def expand_with_neighbors_local(
    candidates: list[RetrievalCandidate],
    radius: int,
    final_limit: int | None = None,
) -> list[RetrievalCandidate]:
    if radius <= 0:
        return candidates[:final_limit] if final_limit else candidates

    expanded: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        for delta in range(-radius, radius + 1):
            page = candidate.page + delta
            if page < 1:
                continue
            path = Path(candidate.path).parent / f"page_{page}.png"
            if not path.exists():
                continue
            key = f"{candidate.folder}_{page}"
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                RetrievalCandidate(
                    folder=candidate.folder,
                    page=page,
                    path=path,
                    score=candidate.score if delta == 0 else candidate.score * 0.8,
                    rank=candidate.rank,
                    index=candidate.index,
                    source="neighbor_expansion" if delta else candidate.source,
                    rerank_score=candidate.rerank_score if delta == 0 else None,
                )
            )

    expanded.sort(
        key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
        reverse=True,
    )
    return expanded[:final_limit] if final_limit else expanded


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    question_types = (
        None if "all" in {str(item).lower() for item in args.types} else set(args.types)
    )
    questions = load_docbench_questions(
        args.data_dir,
        question_types=question_types,
    )
    rows: list[dict[str, Any]] = []
    for question_id, question in enumerate(questions, start=1):
        expected_folder = str(question.get("folder"))
        rows.append(
            {
                "question_id": question_id,
                "folder": expected_folder,
                "expected_folder": expected_folder,
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
                "type": question.get("type", ""),
                "evidence": question.get("evidence", ""),
                "oracle_pages": [],
            }
        )
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def latency_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key, 0.0)) for row in rows]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def optional_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean([float(value) for value in values]))


def normalize_unit(unit: str) -> str:
    unit = (unit or "").lower().strip()
    unit = unit.replace("mtco₂-e", "mtco2e").replace("mtco2-e", "mtco2e")
    unit = unit.replace("$", "usd").replace("€", "eur").replace("£", "gbp").replace("₹", "inr")
    aliases = {
        "percent": "%",
        "percentage": "%",
        "token": "tokens",
        "point": "points",
        "lb": "lbs",
        "crore": "crores",
        "share": "shares",
        "employee": "employees",
        "document": "documents",
        "dialogue": "dialogues",
        "store": "stores",
        "restaurant": "restaurants",
        "sentence": "sentences",
        "page": "pages",
    }
    return aliases.get(unit, unit)


def extract_numeric_mentions(text: str) -> list[dict[str, Any]]:
    unit_pattern = (
        r"%|percent|percentage|million|billion|thousand|tokens?|points?|pp|kg|lbs?|"
        r"mtco2e|mtco₂-e|co2e|dkk|usd|eur|chf|rmb|crores?|shares?|employees?|"
        r"documents?|dialogues?|stores?|restaurants?|sentences?|pages?"
    )
    pattern = re.compile(
        rf"(?P<prefix>[$€£₹])?\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{unit_pattern})?",
        re.IGNORECASE,
    )
    mentions: list[dict[str, Any]] = []
    for match in pattern.finditer(text or ""):
        raw_num = match.group("num").replace(",", "")
        try:
            value = float(raw_num)
        except ValueError:
            continue
        unit = normalize_unit(match.group("unit") or match.group("prefix") or "")
        mentions.append(
            {
                "value": value,
                "unit": unit,
                "is_integer": "." not in raw_num,
                "is_percentage": unit == "%",
            }
        )
    return mentions


def exact_number_match(expected: dict[str, Any], generated: dict[str, Any]) -> bool:
    return abs(float(expected["value"]) - float(generated["value"])) <= 1e-9


def relaxed_number_match(expected: dict[str, Any], generated: dict[str, Any]) -> bool:
    left = float(expected["value"])
    right = float(generated["value"])
    if expected.get("is_percentage") or generated.get("is_percentage"):
        return abs(left - right) <= 0.1
    if expected.get("is_integer") and generated.get("is_integer"):
        return exact_number_match(expected, generated)
    return abs(left - right) / max(abs(left), 1e-9) <= 0.01


def all_expected_numbers_match(
    expected: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    relaxed: bool,
) -> bool | None:
    if not expected:
        return None
    matcher = relaxed_number_match if relaxed else exact_number_match
    return all(any(matcher(exp, gen) for gen in generated) for exp in expected)


def compute_unit_match(
    expected: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> bool | None:
    expected_units = {item["unit"] for item in expected if item.get("unit")}
    generated_units = {item["unit"] for item in generated if item.get("unit")}
    if not expected_units:
        return None
    return expected_units.issubset(generated_units)


def normalize_entity_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    suffixes = {
        "the",
        "ltd",
        "limited",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "group",
        "plc",
        "llc",
    }
    tokens = [
        token
        for token in text.split()
        if token not in suffixes and not token.isdigit() and len(token) > 1
    ]
    return " ".join(tokens)


def compute_entity_match(generated: str, expected: str) -> bool | None:
    expected_norm = normalize_entity_text(expected)
    generated_norm = normalize_entity_text(generated)
    if not expected_norm or re.fullmatch(r"[\d\s.%-]+", expected_norm):
        return None
    if expected_norm in generated_norm or generated_norm in expected_norm:
        return True
    expected_tokens = set(expected_norm.split())
    generated_tokens = set(generated_norm.split())
    if not expected_tokens:
        return None
    return len(expected_tokens & generated_tokens) / len(expected_tokens) >= 0.8


def doc_hit(
    candidates: list[RetrievalCandidate], expected_folder: str | None, top_k: int
) -> bool | None:
    if expected_folder is None:
        return None
    return any(str(candidate.folder) == str(expected_folder) for candidate in candidates[:top_k])


def page_hit(
    candidates: list[RetrievalCandidate], oracle_pages: list[Any], top_k: int
) -> bool | None:
    if not oracle_pages:
        return None
    normalized = set()
    for page in oracle_pages:
        if isinstance(page, dict) and "page" in page:
            normalized.add(str(page["page"]))
        else:
            normalized.add(str(page))
    return any(
        f"{candidate.folder}/{candidate.page}" in normalized
        or f"{candidate.folder}_{candidate.page}" in normalized
        or str(candidate.page) in normalized
        for candidate in candidates[:top_k]
    )


def extra_row_metrics(
    *,
    generated: str,
    expected: str,
    retrieved: list[RetrievalCandidate],
    reranked: list[RetrievalCandidate],
    expected_folder: str | None,
    oracle_pages: list[Any],
) -> dict[str, Any]:
    expected_numbers = extract_numeric_mentions(expected)
    generated_numbers = extract_numeric_mentions(generated)
    return {
        "numeric_exact_match": all_expected_numbers_match(
            expected_numbers, generated_numbers, relaxed=False
        ),
        "numeric_relaxed_match": all_expected_numbers_match(
            expected_numbers, generated_numbers, relaxed=True
        ),
        "unit_match": compute_unit_match(expected_numbers, generated_numbers),
        "entity_match": compute_entity_match(generated, expected),
        "doc_hit_at_1": doc_hit(retrieved, expected_folder, 1),
        "doc_hit_at_5": doc_hit(retrieved, expected_folder, 5),
        "doc_hit_at_10": doc_hit(retrieved, expected_folder, 10),
        "doc_hit_at_30": doc_hit(retrieved, expected_folder, 30),
        "reranked_doc_hit_at_1": doc_hit(reranked, expected_folder, 1),
        "reranked_doc_hit_at_5": doc_hit(reranked, expected_folder, 5),
        "reranked_doc_hit_at_10": doc_hit(reranked, expected_folder, 10),
        "page_hit_at_1": page_hit(retrieved, oracle_pages, 1),
        "page_hit_at_5": page_hit(retrieved, oracle_pages, 5),
        "page_hit_at_10": page_hit(retrieved, oracle_pages, 10),
        "table_hit_at_k": None,
    }


def summarize_full_pipeline(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_with_crop_metrics(results, [row["latency"] for row in results])
    summary["latency_note"] = (
        "latency_seconds is end-to-end per question: retrieval + rerank + crop/context + VLM."
    )
    summary["latency_breakdown_seconds"] = {
        "retrieval": latency_summary(results, "retrieval_latency"),
        "rerank": latency_summary(results, "rerank_latency"),
        "context": latency_summary(results, "context_latency"),
        "vlm": latency_summary(results, "vlm_latency"),
    }
    metric_keys = [
        "numeric_exact_match",
        "numeric_relaxed_match",
        "unit_match",
        "entity_match",
        "doc_hit_at_1",
        "doc_hit_at_5",
        "doc_hit_at_10",
        "doc_hit_at_30",
        "reranked_doc_hit_at_1",
        "reranked_doc_hit_at_5",
        "reranked_doc_hit_at_10",
        "page_hit_at_1",
        "page_hit_at_5",
        "page_hit_at_10",
        "table_hit_at_k",
    ]
    summary["additional_metrics"] = {key: optional_mean(results, key) for key in metric_keys}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row.get("type", "unknown"), []).append(row)
    summary["additional_metrics_by_type"] = {
        group_name: {key: optional_mean(group_rows, key) for key in metric_keys}
        for group_name, group_rows in grouped.items()
    }
    summary["additional_metrics_note"] = (
        "page_hit_at_k and table_hit_at_k are null unless the dataset provides oracle page/table annotations."
    )
    return summary


def write_partial(
    args: argparse.Namespace, results: list[dict[str, Any]], debug_rows: list[dict[str, Any]]
) -> None:
    output = {
        "summary": summarize_full_pipeline(results),
        "config": vars(args),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)
    write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)


def main() -> None:
    args = parse_args()
    rows = load_questions(args)

    print(
        json.dumps(
            {
                "selected_questions": len(rows),
                "data_dir": str(args.data_dir),
                "retrieval": "colvision_multi_vector",
                "index_name": args.index_name,
                "index_dir": str(args.index_dir),
                "retriever_model_id": args.retriever_model_id,
                "first_stage_top_k": args.first_stage_top_k,
                "rerank_top_k": args.rerank_top_k,
                "adaptive_policy": args.adaptive_policy,
                "visual_crop_policy": args.visual_crop_policy,
                "layout_context_mode": args.layout_context_mode,
                "prompt_style": args.prompt_style,
                "dry_run": args.dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    from src.mmrag.rerank import NemotronVLReranker
    from src.mmrag.config import RerankerConfig

    retriever = ColVisionRetriever(args)
    reranker = NemotronVLReranker(
        RerankerConfig(device=args.rerank_device, batch_size=args.rerank_batch_size)
    )
    generator = create_generator(args)

    results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for display_idx, row in enumerate(rows, start=1 + args.start):
        expected = row.get("answer") or ""
        question = row["question"]
        total_start = time.time()

        retrieval_start = time.time()
        retrieved = retriever.search(question, top_k=args.first_stage_top_k)
        retrieval_latency = time.time() - retrieval_start

        rerank_start = time.time()
        reranked = reranker.rerank(question, retrieved)[: args.rerank_top_k]
        rerank_latency = time.time() - rerank_start

        effective_top_pages, effective_crop_policy, visual_context = (
            effective_context_settings_layout(args, row)
        )
        context_candidates = expand_with_neighbors_local(
            reranked[:effective_top_pages],
            args.neighbor_radius,
            final_limit=effective_top_pages,
        )
        context_candidate_rows = [
            candidate_to_json(candidate, rank=rank)
            for rank, candidate in enumerate(context_candidates, start=1)
        ]

        context_start = time.time()
        images, layout_debug, selected_crop = load_context_images(
            candidates=context_candidate_rows,
            crop_policy=effective_crop_policy,
            crop_top_n=args.crop_top_n,
            question=question,
            row=row,
            debug_crop_dir=args.debug_crop_dir,
            question_index=display_idx,
            layout_context_mode=args.layout_context_mode,
        )
        context_latency = time.time() - context_start

        print(f"\n[{display_idx}/{args.start + len(rows)}] {question[:100]}")
        print(
            "    retrieved_top5="
            f"{[f'{candidate.folder}/{candidate.page}' for candidate in retrieved[:5]]}"
        )
        print(
            "    reranked_top5="
            f"{[f'{candidate.folder}/{candidate.page}' for candidate in reranked[:5]]}"
        )
        print(
            f"    context=top{effective_top_pages} crop={effective_crop_policy} "
            f"mode={args.layout_context_mode} visual={visual_context}"
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

        vlm_start = time.time()
        try:
            answer = generator.generate_answer(question, images)
        except Exception as exc:
            print(f"    [ERROR] VLM failed: {exc}")
            generator.last_raw_output = ""
            generator.last_reasoning = ""
            generator.last_answer = "ERROR"
            answer = "ERROR"
        vlm_latency = time.time() - vlm_start
        latency = time.time() - total_start

        exact, f1 = compute_similarity(answer, expected)
        extended_metrics = compute_extended_metrics(answer, expected)
        oracle_matches = find_answer_pages(
            args.data_dir / str(row.get("expected_folder")),
            expected,
            row.get("evidence", ""),
        )
        oracle_pages = [
            {
                "page": match.page,
                "exact_answer": match.exact_answer,
                "number_recall": match.number_recall,
                "keyword_recall": match.keyword_recall,
                "matched_numbers": match.matched_numbers,
                "matched_keywords": match.matched_keywords,
            }
            for match in oracle_matches[:10]
        ]
        extra_metrics = extra_row_metrics(
            generated=answer,
            expected=expected,
            retrieved=retrieved,
            reranked=reranked,
            expected_folder=row.get("expected_folder"),
            oracle_pages=oracle_pages,
        )
        result = {
            "question_id": row.get("question_id"),
            "question": question,
            "expected": expected,
            "generated": answer,
            "raw_generated": getattr(generator, "last_raw_output", answer),
            "vlm_think": getattr(generator, "last_reasoning", ""),
            "exact": exact,
            "f1": f1,
            "latency": latency,
            "retrieval_latency": retrieval_latency,
            "rerank_latency": rerank_latency,
            "context_latency": context_latency,
            "vlm_latency": vlm_latency,
            "type": row.get("type", ""),
            "expected_folder": row.get("expected_folder"),
            "evidence": row.get("evidence", ""),
            "oracle_pages": oracle_pages,
            "effective_top_pages": effective_top_pages,
            "effective_crop_policy": effective_crop_policy,
            "visual_context": visual_context,
            "retrieved_candidates": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(retrieved, start=1)
            ],
            "reranked_candidates": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(reranked, start=1)
            ],
            "pages": context_candidate_rows,
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
            **extra_metrics,
        }
        results.append(result)
        debug_rows.append(compact_case(result))

        if args.print_think:
            think = (result["vlm_think"] or result["raw_generated"] or "").replace("\n", " ")
            print(f"    think={think[:500]}")
        print(f"    generated={answer[:240]}")
        print(f"    expected={expected}")
        print(
            f"    exact={exact} f1={f1:.3f} "
            f"latency={latency:.2f}s "
            f"(retrieval={retrieval_latency:.2f}s rerank={rerank_latency:.2f}s "
            f"context={context_latency:.2f}s vlm={vlm_latency:.2f}s)"
        )

        write_partial(args, results, debug_rows)

    output = {
        "summary": summarize_full_pipeline(results),
        "config": vars(args),
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
