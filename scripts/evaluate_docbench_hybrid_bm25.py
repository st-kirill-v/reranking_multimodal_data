from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_full_pipeline_layout_aware_clean import (  # noqa: E402
    ColVisionRetriever,
    candidate_to_json,
    effective_context_settings_layout,
    expand_with_neighbors_local,
    extra_row_metrics,
    find_answer_pages,
    load_context_images,
    load_questions,
)
from src.cropping.layout_aware_eval import (  # noqa: E402
    compact_case,
    compute_extended_metrics,
    compute_similarity,
    summarize_with_crop_metrics,
    write_case_csv,
)
from src.pipelines.text_qa_pipeline import TextQAPipeline  # noqa: E402
from src.retrieval.text_bm25_retriever import DocBenchBM25Retriever  # noqa: E402


TEXT_TYPES = {"text-only", "meta-data", "unanswerable"}
MULTIMODAL_TYPES = {"multimodal-t", "multimodal-f"}


def create_hybrid_generator(args: argparse.Namespace) -> Any:
    if args.prompt_profile == "docbench_hybrid_v1":
        from src.generation.docbench_hybrid_generator import create_docbench_hybrid_generator

        return create_docbench_hybrid_generator(
            device=args.device,
            max_image_long_edge=args.max_image_long_edge,
            load_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
            do_sample=args.do_sample,
            max_images=args.max_context_images,
            answer_refine=args.answer_refine,
        )

    from src.cropping.layout_aware_eval import create_generator

    return create_generator(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid DocBench evaluation: BM25 text pipeline + current multimodal pipeline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--types", nargs="*", default=["all"])
    parser.add_argument("--routing-mode", choices=["by_official_type"], default="by_official_type")

    parser.add_argument("--text-index-dir", type=Path, default=Path("data/indexes/docbench_bm25"))
    parser.add_argument("--text-top-k", type=int, default=5)
    parser.add_argument("--text-context-max-chars", type=int, default=12000)

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
    parser.add_argument("--prompt-profile", default="legacy")

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
        default=Path("data/debug_crops/docbench_hybrid_bm25"),
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_docbench_hybrid_bm25.json"),
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1 - frac) + values[high] * frac


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency", 0.0)) for row in rows if row.get("latency") is not None]
    return {
        "count": len(rows),
        "exact_match": mean([float(row.get("exact", 0.0)) for row in rows]) if rows else 0.0,
        "mean_f1": mean([float(row.get("f1", 0.0)) for row in rows]) if rows else 0.0,
        "accuracy_f1_gt_0_5": (
            mean([float(row.get("f1", 0.0)) > 0.5 for row in rows]) if rows else 0.0
        ),
        "latency_seconds": {
            "mean": mean(latencies) if latencies else 0.0,
            "p50": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
        },
    }


def summarize_hybrid(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency", 0.0)) for row in results]
    summary = summarize_with_crop_metrics(results, latencies)
    by_original_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pipeline_used: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_original_type[row.get("original_type") or row.get("type") or "unknown"].append(row)
        by_pipeline_used[row.get("pipeline_used") or "unknown"].append(row)
    summary["by_original_type"] = {
        key: summarize_group(value) for key, value in sorted(by_original_type.items())
    }
    summary["by_pipeline_used"] = {
        key: summarize_group(value) for key, value in sorted(by_pipeline_used.items())
    }
    return summary


def text_doc_hit(
    results: list[dict[str, Any]], expected_folder: str | None, top_k: int
) -> bool | None:
    if expected_folder is None:
        return None
    return any(str(row.get("doc_id")) == str(expected_folder) for row in results[:top_k])


def text_page_labels(results: list[dict[str, Any]]) -> list[str]:
    return [f"{row.get('doc_id')}/{row.get('page')}" for row in results]


def run_text_question(
    args: argparse.Namespace,
    text_pipeline: TextQAPipeline,
    row: dict[str, Any],
) -> dict[str, Any]:
    question = row["question"]
    expected = row.get("answer") or ""
    total_start = time.time()
    expected_folder = row.get("expected_folder")
    qa_result = text_pipeline.answer(
        question,
        doc_id=expected_folder,
        question_type=row.get("type", "text-only"),
        context_mode="text",
    )
    latency = time.time() - total_start

    answer = qa_result.answer
    exact, f1 = compute_similarity(answer, expected)
    extended_metrics = compute_extended_metrics(answer, expected)
    retrieved_pages = qa_result.retrieved_text_pages
    return {
        "question_id": row.get("question_id"),
        "question": question,
        "expected": expected,
        "generated": answer,
        "raw_generated": getattr(text_pipeline.generator, "last_raw_output", answer),
        "vlm_think": getattr(text_pipeline.generator, "last_reasoning", ""),
        "exact": exact,
        "f1": f1,
        "latency": latency,
        "retrieval_latency": qa_result.latency_retrieval,
        "rerank_latency": 0.0,
        "context_latency": 0.0,
        "vlm_latency": qa_result.latency_generation,
        "latency_generation": qa_result.latency_generation,
        "type": row.get("type", ""),
        "original_type": row.get("type", ""),
        "pipeline_used": "text_bm25",
        "prompt_profile": qa_result.prompt_profile,
        "prompt_name": qa_result.prompt_name,
        "context_mode": "text",
        "expected_folder": row.get("expected_folder"),
        "text_doc_scope": expected_folder,
        "retrieved_text_pages": retrieved_pages,
        "selected_pages": text_page_labels(retrieved_pages),
        "retrieved_pages": text_page_labels(retrieved_pages),
        "text_context": qa_result.context,
        "crop_used": False,
        "fallback_used": False,
        "doc_hit_at_1": text_doc_hit(retrieved_pages, row.get("expected_folder"), 1),
        "doc_hit_at_5": text_doc_hit(retrieved_pages, row.get("expected_folder"), 5),
        "page_hit_at_1": None,
        "page_hit_at_5": None,
        **extended_metrics,
    }


def run_multimodal_question(
    args: argparse.Namespace,
    retriever: ColVisionRetriever,
    reranker: Any,
    generator: Any,
    row: dict[str, Any],
    display_idx: int,
) -> dict[str, Any]:
    expected = row.get("answer") or ""
    question = row["question"]
    total_start = time.time()

    retrieval_start = time.time()
    retrieved = retriever.search(question, top_k=args.first_stage_top_k)
    retrieval_latency = time.time() - retrieval_start

    rerank_start = time.time()
    reranked = reranker.rerank(question, retrieved)[: args.rerank_top_k]
    rerank_latency = time.time() - rerank_start

    effective_top_pages, effective_crop_policy, visual_context = effective_context_settings_layout(
        args, row
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

    vlm_start = time.time()
    try:
        if hasattr(generator, "generate_answer_for_type"):
            answer = generator.generate_answer_for_type(
                query=question,
                question_type=row.get("type", "multimodal-t"),
                context_mode=args.layout_context_mode,
                context_images=images,
                context_text=None,
            )
        else:
            answer = generator.generate_answer(question, images)
    except Exception as exc:  # noqa: BLE001 - keep one failed sample from killing the run.
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
    return {
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
        "latency_generation": vlm_latency,
        "type": row.get("type", ""),
        "original_type": row.get("type", ""),
        "pipeline_used": "multimodal",
        "prompt_profile": getattr(generator, "last_prompt_profile", None),
        "prompt_name": getattr(generator, "last_prompt_name", None),
        "context_mode": args.layout_context_mode,
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


def write_partial(
    args: argparse.Namespace, results: list[dict[str, Any]], debug_rows: list[dict[str, Any]]
) -> None:
    output = {
        "summary": summarize_hybrid(results),
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
    text_rows = [row for row in rows if row.get("type") in TEXT_TYPES]
    multimodal_rows = [row for row in rows if row.get("type") in MULTIMODAL_TYPES]

    print(
        json.dumps(
            {
                "selected_questions": len(rows),
                "text_questions": len(text_rows),
                "multimodal_questions": len(multimodal_rows),
                "routing_mode": args.routing_mode,
                "text_retriever": "bm25",
                "text_index_dir": str(args.text_index_dir),
                "text_top_k": args.text_top_k,
                "text_context_max_chars": args.text_context_max_chars,
                "multimodal_retrieval": "colvision_multi_vector",
                "visual_crop_policy": args.visual_crop_policy,
                "layout_context_mode": args.layout_context_mode,
                "prompt_style": args.prompt_style,
                "prompt_profile": args.prompt_profile,
                "dry_run": args.dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    generator = create_hybrid_generator(args)
    text_retriever = DocBenchBM25Retriever(args.text_index_dir)
    text_pipeline = TextQAPipeline(
        retriever=text_retriever,
        generator=generator,
        top_k=args.text_top_k,
        context_max_chars=args.text_context_max_chars,
    )

    multimodal_retriever = None
    reranker = None
    if multimodal_rows:
        from src.mmrag.config import RerankerConfig
        from src.mmrag.rerank import NemotronVLReranker

        multimodal_retriever = ColVisionRetriever(args)
        reranker = NemotronVLReranker(
            RerankerConfig(device=args.rerank_device, batch_size=args.rerank_batch_size)
        )

    results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for display_idx, row in enumerate(rows, start=1 + args.start):
        question = row["question"]
        question_type = row.get("type", "")
        print(f"\n[{display_idx}/{args.start + len(rows)}] {question[:100]}")
        print(f"    original_type={question_type}")

        if question_type in TEXT_TYPES:
            result = run_text_question(args, text_pipeline, row)
            text_hits = result.get("retrieved_text_pages", [])
            print(
                "    pipeline=text_bm25 "
                f"doc_scope={result.get('text_doc_scope')} "
                f"selected_pages={result.get('selected_pages', [])[:5]}"
            )
            print(
                f"    prompt_profile={result.get('prompt_profile')} "
                f"prompt_name={result.get('prompt_name')}"
            )
            print(
                "    bm25_top5="
                f"{[f'{hit.get('doc_id')}/{hit.get('page')}:{float(hit.get('score', 0.0)):.3f}' for hit in text_hits[:5]]}"
            )
        elif question_type in MULTIMODAL_TYPES:
            if multimodal_retriever is None or reranker is None:
                raise RuntimeError("Multimodal components were not initialized")
            result = run_multimodal_question(
                args,
                multimodal_retriever,
                reranker,
                generator,
                row,
                display_idx,
            )
            print(
                "    pipeline=multimodal "
                f"retrieved_top5={[f'{c.get('folder')}/{c.get('page')}' for c in result.get('retrieved_candidates', [])[:5]]}"
            )
            print(
                "    reranked_top5="
                f"{[f'{c.get('folder')}/{c.get('page')}' for c in result.get('reranked_candidates', [])[:5]]}"
            )
            print(
                "    selected_pages="
                f"{[f'{c.get('folder')}/{c.get('page')}' for c in result.get('pages', [])]}"
            )
            print(
                "    layout_crop="
                f"{result.get('selected_crop_type')} score={result.get('selected_crop_score')} "
                f"path={result.get('crop_path')} fallback={result.get('fallback_used')}"
            )
            print(
                f"    prompt_profile={result.get('prompt_profile')} "
                f"prompt_name={result.get('prompt_name')}"
            )
        else:
            result = {
                "question_id": row.get("question_id"),
                "question": question,
                "expected": row.get("answer") or "",
                "generated": "ERROR",
                "exact": 0.0,
                "f1": 0.0,
                "latency": 0.0,
                "retrieval_latency": 0.0,
                "rerank_latency": 0.0,
                "context_latency": 0.0,
                "vlm_latency": 0.0,
                "latency_generation": 0.0,
                "type": question_type,
                "original_type": question_type,
                "pipeline_used": "unsupported",
                "runtime_error": f"Unsupported question type: {question_type}",
            }

        results.append(result)
        if result.get("pipeline_used") == "multimodal":
            debug_rows.append(compact_case(result))
        print(f"    generated={str(result.get('generated', ''))[:240]}")
        print(f"    expected={result.get('expected', '')}")
        print(
            f"    exact={result.get('exact')} f1={float(result.get('f1', 0.0)):.3f} "
            f"latency={float(result.get('latency', 0.0)):.2f}s"
        )
        write_partial(args, results, debug_rows)

    output = {
        "summary": summarize_hybrid(results),
        "config": vars(args),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)
    write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
