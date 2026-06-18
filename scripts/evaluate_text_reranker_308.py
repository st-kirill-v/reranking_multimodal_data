#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_full_pipeline_layout_aware_clean import (  # noqa: E402
    extra_row_metrics,
    find_answer_pages,
)
from src.cropping.layout_aware_eval import (  # noqa: E402
    compute_extended_metrics,
    compute_similarity,
)
from src.generation.openai_compatible_vlm import create_openai_compatible_vlm  # noqa: E402
from src.mmrag.dataset import load_docbench_questions  # noqa: E402
from src.reranking.text_reranker import create_text_reranker  # noqa: E402
from src.retrieval.text_encoder_retriever import TextEncoderRetriever  # noqa: E402
from src.retrieval.text_page_retriever import TextPageBM25Retriever, TextPageCandidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DocBench 308 text-only reranking experiment: page_text retrieval -> "
            "neural text reranker -> Qwen3-VL-30B text generation -> metrics."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--first-stage-top-k", type=int, default=30)
    parser.add_argument(
        "--text-retriever-backend",
        choices=["bm25", "text_encoder"],
        default="bm25",
    )
    parser.add_argument(
        "--text-encoder-index-dir",
        type=Path,
        default=Path("data/indexes/docbench_text_encoder_bge_base_en_v1_5"),
    )
    parser.add_argument("--text-encoder-model-id", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--text-encoder-device", default="cuda")
    parser.add_argument("--text-encoder-batch-size", type=int, default=32)
    parser.add_argument("--text-encoder-max-length", type=int, default=512)
    parser.add_argument("--no-text-encoder-normalize", action="store_true")
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--context-top-pages", type=int, default=3)
    parser.add_argument("--text-context-max-chars", type=int, default=12000)
    parser.add_argument(
        "--text-source-fields",
        nargs="*",
        default=["page_text"],
        help="Recorded text evidence fields. Current active corpus uses page_text.",
    )

    parser.add_argument("--text-reranker-model-id", required=True)
    parser.add_argument("--text-reranker-device", default="cuda")
    parser.add_argument("--text-reranker-batch-size", type=int, default=8)
    parser.add_argument("--text-reranker-max-length", type=int, default=4096)
    parser.add_argument(
        "--text-reranker-backend",
        choices=["cross_encoder", "lexical", "none"],
        default="cross_encoder",
    )
    parser.add_argument("--no-trust-remote-code", action="store_true")

    parser.add_argument("--openai-vlm-base-url", default="http://10.32.15.88:4000/v1")
    parser.add_argument("--openai-vlm-model", default="openai/qwen3-vl-30b")
    parser.add_argument("--openai-vlm-api-key-env", default="OPENAI_COMPAT_API_KEY")
    parser.add_argument("--openai-vlm-api-key", default=None)
    parser.add_argument("--openai-vlm-temperature", type=float, default=0.0)
    parser.add_argument("--openai-vlm-max-tokens", type=int, default=192)
    parser.add_argument("--openai-vlm-timeout", type=float, default=180.0)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/eval_text_reranker_308.json"))
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    question_types = (
        None if "all" in {str(item).lower() for item in args.types} else set(args.types)
    )
    questions = load_docbench_questions(args.data_dir, question_types=question_types)
    rows = []
    for question_id, question in enumerate(questions, start=1):
        folder = str(question.get("folder"))
        rows.append(
            {
                "question_id": question_id,
                "folder": folder,
                "expected_folder": folder,
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
                "type": question.get("type", ""),
                "evidence": question.get("evidence", ""),
            }
        )
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def candidate_label(candidate: TextPageCandidate) -> str:
    return f"{candidate.doc_id}/{candidate.page}"


def build_text_context(
    candidates: list[TextPageCandidate], max_chars: int
) -> tuple[str, bool, int]:
    chunks: list[str] = []
    total = 0
    truncated = False
    for candidate in candidates:
        header = (
            f"[doc_id={candidate.doc_id} page={candidate.page} "
            f"retrieval_score={candidate.score:.4f} "
            f"source={candidate.source} "
            f"evidence_fields={','.join(candidate.evidence_fields or [])} "
            f"text_rerank_score={candidate.text_rerank_score}]\n"
        )
        chunk = f"{header}{candidate.text.strip()}"
        remaining = max_chars - total
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
            truncated = True
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks), truncated, total


def latency_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key, 0.0)) for row in rows if row.get(key) is not None]
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


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row.get("type", "unknown"), []).append(row)
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
    ]
    return {
        "total": len(results),
        "exact_match": float(np.mean([row["exact"] for row in results])) if results else 0.0,
        "mean_f1": float(np.mean([row["f1"] for row in results])) if results else 0.0,
        "accuracy_f1_gt_0_5": (
            float(np.mean([row["f1"] > 0.5 for row in results])) if results else 0.0
        ),
        "latency_seconds": latency_summary(results, "latency"),
        "latency_breakdown_seconds": {
            "retrieval": latency_summary(results, "retrieval_latency"),
            "rerank": latency_summary(results, "rerank_latency"),
            "context": latency_summary(results, "context_latency"),
            "vlm": latency_summary(results, "vlm_latency"),
        },
        "by_type": {
            name: {
                "count": len(rows),
                "mean_f1": float(np.mean([row["f1"] for row in rows])) if rows else 0.0,
                "accuracy_f1_gt_0_5": (
                    float(np.mean([row["f1"] > 0.5 for row in rows])) if rows else 0.0
                ),
            }
            for name, rows in grouped.items()
        },
        "additional_metrics": {key: optional_mean(results, key) for key in metric_keys},
        "additional_metrics_by_type": {
            name: {key: optional_mean(rows, key) for key in metric_keys}
            for name, rows in grouped.items()
        },
        "context_truncation_rate": (
            float(np.mean([bool(row.get("context_truncated")) for row in results]))
            if results
            else 0.0
        ),
    }


def write_partial(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarize(results), "config": vars(args), "results": results}
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows = load_questions(args)
    print(
        json.dumps(
            {
                "selected_questions": len(rows),
                "data_dir": str(args.data_dir),
                "pipeline": "text_evidence_retriever_to_text_reranker_to_qwen3vl30b",
                "text_retriever_backend": args.text_retriever_backend,
                "text_encoder_index_dir": str(args.text_encoder_index_dir),
                "text_encoder_model_id": args.text_encoder_model_id,
                "text_source_fields": args.text_source_fields,
                "first_stage_top_k": args.first_stage_top_k,
                "text_reranker_model_id": args.text_reranker_model_id,
                "text_reranker_batch_size": args.text_reranker_batch_size,
                "text_reranker_max_length": args.text_reranker_max_length,
                "context_top_pages": args.context_top_pages,
                "text_context_max_chars": args.text_context_max_chars,
                "openai_vlm_model": args.openai_vlm_model,
                "dry_run": args.dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    if args.text_retriever_backend == "text_encoder":
        retriever = TextEncoderRetriever(
            args.text_encoder_index_dir,
            model_id=args.text_encoder_model_id,
            device=args.text_encoder_device,
            batch_size=args.text_encoder_batch_size,
            max_length=args.text_encoder_max_length,
            normalize=not args.no_text_encoder_normalize,
        )
    else:
        retriever = TextPageBM25Retriever(args.data_dir, source_fields=args.text_source_fields)
    print(f"[TextRetriever] {json.dumps(retriever.stats(), ensure_ascii=False)}")
    reranker = None
    if args.text_reranker_backend != "none":
        reranker = create_text_reranker(
            args.text_reranker_model_id,
            device=args.text_reranker_device,
            batch_size=args.text_reranker_batch_size,
            max_length=args.text_reranker_max_length,
            trust_remote_code=not args.no_trust_remote_code,
            backend=args.text_reranker_backend,
        )
    generator = create_openai_compatible_vlm(
        base_url=args.openai_vlm_base_url,
        model=args.openai_vlm_model,
        api_key_env=args.openai_vlm_api_key_env,
        api_key=args.openai_vlm_api_key,
        temperature=args.openai_vlm_temperature,
        max_tokens=args.openai_vlm_max_tokens,
        timeout=args.openai_vlm_timeout,
    )

    results: list[dict[str, Any]] = []
    total_count = args.start + len(rows)
    for display_idx, row in enumerate(rows, start=1 + args.start):
        question = row["question"]
        expected = row.get("answer") or ""
        total_start = time.time()

        retrieval_start = time.time()
        retrieved = retriever.search(question, top_k=args.first_stage_top_k)
        retrieval_latency = time.time() - retrieval_start

        if reranker is None:
            reranked = retrieved[: args.rerank_top_k]
            rerank_latency = 0.0
            rerank_backend = "none"
            rerank_model_id = "none"
        else:
            rerank_output = reranker.rerank(question, retrieved)
            reranked = rerank_output.candidates[: args.rerank_top_k]
            rerank_latency = rerank_output.latency
            rerank_backend = rerank_output.backend
            rerank_model_id = rerank_output.model_id

        context_start = time.time()
        selected = reranked[: args.context_top_pages]
        context_text, context_truncated, context_chars = build_text_context(
            selected, args.text_context_max_chars
        )
        context_latency = time.time() - context_start

        print(f"\n[{display_idx}/{total_count}] {question[:100]}")
        print(f"    original_type={row.get('type')}")
        print(f"    retrieved_top5={[candidate_label(candidate) for candidate in retrieved[:5]]}")
        print(
            "    text_reranked_top5="
            f"{[candidate_label(candidate) for candidate in reranked[:5]]}"
        )
        print(
            f"    context=top{args.context_top_pages} chars={context_chars} "
            f"truncated={context_truncated}"
        )

        vlm_start = time.time()
        answer = generator.generate_answer_for_type(
            query=question,
            question_type="text-only",
            context_mode="text",
            context_images=None,
            context_text=context_text,
        )
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
            "raw_generated_answer": getattr(generator, "last_raw_output", answer),
            "postprocessed_answer": answer,
            "exact": exact,
            "f1": f1,
            "latency": latency,
            "retrieval_latency": retrieval_latency,
            "rerank_latency": rerank_latency,
            "context_latency": context_latency,
            "vlm_latency": vlm_latency,
            "latency_generation": getattr(generator, "last_latency_generation", vlm_latency),
            "generation_backend": getattr(generator, "generation_backend", "openai_compatible_vlm"),
            "model_name": getattr(generator, "model", args.openai_vlm_model),
            "generation_error": getattr(generator, "last_error", None),
            "type": row.get("type", ""),
            "expected_folder": row.get("expected_folder"),
            "evidence": row.get("evidence", ""),
            "oracle_pages": oracle_pages,
            "pipeline_used": "text_page_reranker",
            "retriever_backend": args.text_retriever_backend,
            "text_encoder_index_dir": (
                str(args.text_encoder_index_dir)
                if args.text_retriever_backend == "text_encoder"
                else None
            ),
            "text_encoder_model_id": (
                args.text_encoder_model_id
                if args.text_retriever_backend == "text_encoder"
                else None
            ),
            "text_source_fields": args.text_source_fields,
            "text_reranker_backend": rerank_backend,
            "text_reranker_model_id": rerank_model_id,
            "text_reranker_max_length": args.text_reranker_max_length,
            "text_reranker_batch_size": args.text_reranker_batch_size,
            "context_top_pages": args.context_top_pages,
            "text_context_max_chars": args.text_context_max_chars,
            "context_chars": context_chars,
            "context_truncated": context_truncated,
            "retrieval_scores": [float(candidate.score) for candidate in retrieved],
            "text_rerank_scores": [
                float(candidate.text_rerank_score)
                for candidate in reranked
                if candidate.text_rerank_score is not None
            ],
            "top30_retrieved_pages": [candidate_label(candidate) for candidate in retrieved[:30]],
            "retrieved_candidates": [
                {**candidate.to_json(), "rank": rank}
                for rank, candidate in enumerate(retrieved, start=1)
            ],
            "reranked_candidates": [
                {**candidate.to_json(), "rank": rank}
                for rank, candidate in enumerate(reranked, start=1)
            ],
            "selected_pages": [candidate_label(candidate) for candidate in selected],
            "pages": [
                {**candidate.to_json(), "rank": rank}
                for rank, candidate in enumerate(selected, start=1)
            ],
            "crop_used": False,
            "fallback_used": False,
            **extended_metrics,
            **extra_metrics,
        }
        results.append(result)

        print(f"    generated={answer[:240]}")
        print(f"    expected={expected}")
        print(
            f"    exact={exact} f1={f1:.3f} latency={latency:.2f}s "
            f"(retrieval={retrieval_latency:.2f}s rerank={rerank_latency:.2f}s "
            f"context={context_latency:.2f}s vlm={vlm_latency:.2f}s)"
        )
        write_partial(args, results)

    output = {"summary": summarize(results), "config": vars(args), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
