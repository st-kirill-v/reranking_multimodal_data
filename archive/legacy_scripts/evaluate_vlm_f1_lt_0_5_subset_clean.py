from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_vlm_from_page_candidates_clean import (  # noqa: E402
    compute_extended_metrics,
    compute_similarity,
    effective_context_settings,
    load_images,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline VLM eval. By default it runs on questions whose previous F1 "
            "was below a threshold; with --all-questions it runs the full DocBench split."
        )
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("data/eval_vlm_reranked_adaptive_clean_rerun_full_308.json"),
        help="Previous full VLM eval JSON used only to select low-F1 question IDs/texts.",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        help=(
            "Explicit 86-question subset JSON. If set, --eval-json is not used for selection. "
            "Retrieval and reranking are still re-run from scratch."
        ),
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Run all multimodal-t/multimodal-f DocBench questions instead of the low-F1 subset.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--embed-device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--rerank-device", default="cuda")
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--first-stage-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "text_top3_visual_top5"],
        default="text_top3_visual_top5",
    )
    parser.add_argument("--top-pages", type=int, default=5)
    parser.add_argument("--text-top-pages", type=int, default=3)
    parser.add_argument("--visual-top-pages", type=int, default=5)
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
    parser.add_argument("--max-image-long-edge", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-images", type=int, default=5)
    parser.add_argument(
        "--prompt-style",
        choices=["concise", "legacy", "think_answer", "smart_universal"],
        default="concise",
    )
    parser.add_argument("--answer-refine", choices=["none", "text"], default="none")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only select the 86 questions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_vlm_f1_lt_0_5_subset_full_pipeline.json"),
    )
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def load_low_f1_question_keys(path: Path, threshold: float) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results") or payload.get("rows") or []
    return {
        normalize_question(row.get("question", ""))
        for row in rows
        if float(row.get("f1", 0.0)) < threshold
    }


def load_low_f1_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_questions:
        data_dir = (args.data_dir or ROOT / "data" / "datasets" / "docbench").resolve()
        selected = load_docbench_questions_local(
            data_dir,
            question_types={"multimodal-t", "multimodal-f"},
        )
        if args.limit > 0:
            selected = selected[: args.limit]
        return selected

    if args.questions_json:
        with args.questions_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("rows") or payload.get("results") or []
        selected = [
            {
                "folder": str(row.get("folder") or row.get("expected_folder") or ""),
                "question": row.get("question", ""),
                "answer": row.get("answer") or row.get("expected", ""),
                "type": row.get("type", ""),
            }
            for row in rows
        ]
        if args.limit > 0:
            selected = selected[: args.limit]
        return selected

    bad_keys = load_low_f1_question_keys(args.eval_json, args.threshold)
    data_dir = (args.data_dir or ROOT / "data" / "datasets" / "docbench").resolve()
    questions = load_docbench_questions_local(
        data_dir,
        question_types={"multimodal-t", "multimodal-f"},
    )
    selected = [row for row in questions if normalize_question(row["question"]) in bad_keys]
    if args.limit > 0:
        selected = selected[: args.limit]
    missing = bad_keys - {normalize_question(row["question"]) for row in selected}
    if missing:
        print(f"[WARN] Missing {len(missing)} low-F1 questions in docbench qa files.")
    return selected


def load_docbench_questions_local(
    data_dir: Path,
    question_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for jsonl_file in sorted(data_dir.glob("*/*_qa.jsonl")):
        folder = jsonl_file.parent.name
        if not folder.isdigit():
            continue
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if question_types and row.get("type") not in question_types:
                    continue
                questions.append(
                    {
                        "folder": folder,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "type": row.get("type", ""),
                    }
                )
    return questions


def candidate_to_json(candidate: Any, rank: int | None = None) -> dict[str, Any]:
    data = candidate.to_json()
    if rank is not None:
        data["rank"] = rank
    return data


def load_context_images(
    candidates: list[Any],
    crop_policy: str,
    crop_top_n: int,
    question: str,
) -> list[Image.Image]:
    return load_images(
        [candidate_to_json(candidate) for candidate in candidates],
        crop_policy,
        crop_top_n,
        question,
    )


def create_generator(args: argparse.Namespace) -> Any:
    if args.prompt_style == "smart_universal":
        from src.core.generators.qwen_vl_generator_promt_style import create_table_generator
    else:
        from src.core.generators.qwen_vl_generator import create_table_generator

    return create_table_generator(
        device=args.generator_device,
        max_image_long_edge=args.max_image_long_edge,
        load_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        do_sample=args.do_sample,
        max_images=args.max_context_images,
        answer_refine=args.answer_refine,
    )


def build_config(args: argparse.Namespace) -> Any:
    from src.mmrag.config import (
        EmbedderConfig,
        IndexConfig,
        PipelineConfig,
        ProjectPaths,
        RerankerConfig,
        RetrievalConfig,
    )

    return PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(device=args.embed_device, encoding_api=args.encoding_api),
        index=IndexConfig(name=args.index_name),
        retrieval=RetrievalConfig(
            first_stage_top_k=args.first_stage_top_k,
            rerank_top_k=args.rerank_top_k,
            final_top_k=max(args.top_pages, args.text_top_pages, args.visual_top_pages),
            neighbor_radius=args.neighbor_radius,
        ),
        reranker=RerankerConfig(device=args.rerank_device),
    )


def main() -> None:
    args = parse_args()
    questions = load_low_f1_questions(args)
    print(
        json.dumps(
            {
                "selected_questions": len(questions),
                "eval_json": str(args.eval_json),
                "questions_json": str(args.questions_json) if args.questions_json else None,
                "all_questions": args.all_questions,
                "threshold": args.threshold,
                "prompt_style": args.prompt_style,
                "full_pipeline": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    from src.mmrag.rerank import NemotronVLReranker
    from src.mmrag.retrieval import MultimodalPageRetriever, expand_with_neighbors

    config = build_config(args)
    retriever = MultimodalPageRetriever(config)
    reranker = NemotronVLReranker(config.reranker)
    generator = create_generator(args)

    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for idx, question in enumerate(questions, start=1):
        row_for_context = {
            "question": question["question"],
            "type": question.get("type", ""),
        }
        effective_top_pages, effective_crop_policy, visual_context = effective_context_settings(
            args, row_for_context
        )

        print(f"\n[{idx}/{len(questions)}] {question['question'][:100]}")
        first_stage = retriever.search(question["question"], top_k=args.first_stage_top_k)
        reranked_all = reranker.rerank(question["question"], first_stage)[: args.rerank_top_k]
        context_candidates = reranked_all[:effective_top_pages]
        context_candidates = expand_with_neighbors(
            context_candidates,
            radius=args.neighbor_radius,
            final_limit=effective_top_pages,
        )
        page_labels = [f"{candidate.folder}/{candidate.page}" for candidate in context_candidates]
        print(
            f"    pages={page_labels}\n"
            f"    context=top{effective_top_pages} crop={effective_crop_policy} visual={visual_context}"
        )
        images = load_context_images(
            context_candidates,
            effective_crop_policy,
            args.crop_top_n,
            question["question"],
        )

        start_time = time.time()
        try:
            answer = generator.generate_answer(question["question"], images)
        except Exception as exc:
            print(f"    [ERROR] VLM failed: {exc}")
            generator.last_raw_output = ""
            generator.last_reasoning = ""
            generator.last_answer = "ERROR"
            answer = "ERROR"
        latency = time.time() - start_time
        latencies.append(latency)

        exact, f1 = compute_similarity(answer, question["answer"])
        extended_metrics = compute_extended_metrics(answer, question["answer"])

        result = {
            "question": question["question"],
            "expected": question["answer"],
            "generated": answer,
            "raw_generated": getattr(generator, "last_raw_output", answer),
            "vlm_think": getattr(generator, "last_reasoning", ""),
            "exact": exact,
            "f1": f1,
            "latency": latency,
            "type": question.get("type", ""),
            "expected_folder": question.get("folder"),
            "effective_top_pages": effective_top_pages,
            "effective_crop_policy": effective_crop_policy,
            "visual_context": visual_context,
            "top10_prerank": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(first_stage[:10], start=1)
            ],
            "top10_reranked": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(reranked_all[:10], start=1)
            ],
            "pages": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(context_candidates, start=1)
            ],
            **extended_metrics,
        }
        results.append(result)

        if args.print_think:
            think = (result["vlm_think"] or result["raw_generated"] or "").replace("\n", " ")
            print(f"    think={think[:500]}")
        print(f"    generated={answer[:240]}")
        print(f"    expected={question['answer']}")
        print(f"    exact={exact} f1={f1:.3f} latency={latency:.2f}s")

        partial = {
            "summary": summarize(results, latencies),
            "config": vars(args) | {"full_pipeline": True},
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(partial, handle, indent=2, ensure_ascii=False, default=str)

    output = {
        "summary": summarize(results, latencies),
        "config": vars(args) | {"full_pipeline": True},
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
