from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.dataset import load_docbench_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate first-stage page retrieval only.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--search-top-k", type=int, default=0)
    parser.add_argument("--aggregate-page", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/retrieval_eval_qwen3.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.retrieval import MultimodalPageRetriever, aggregate_candidates_by_page

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    questions = load_docbench_questions(config.paths.data_dir, question_types=set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    retriever = MultimodalPageRetriever(config)
    rows = []
    folder_hits = {1: 0, 5: 0, 10: 0, args.top_k: 0}
    search_top_k = args.search_top_k or args.top_k

    for i, question in enumerate(questions, start=1):
        candidates = retriever.search(question["question"], top_k=search_top_k)
        if args.aggregate_page:
            candidates = aggregate_candidates_by_page(candidates)
        candidates = candidates[: args.top_k]
        folders = [candidate.folder for candidate in candidates]
        expected_folder = str(question["folder"])
        for k in folder_hits:
            if expected_folder in folders[:k]:
                folder_hits[k] += 1
        top = candidates[0] if candidates else None
        rows.append(
            {
                "question": question["question"],
                "expected_folder": expected_folder,
                "expected_answer": question["answer"],
                "type": question["type"],
                "top_folder": top.folder if top else None,
                "top_page": top.page if top else None,
                "top_score": top.score if top else None,
                "folder_rank": (
                    (folders.index(expected_folder) + 1) if expected_folder in folders else None
                ),
                "candidates": [candidate.to_json() for candidate in candidates],
            }
        )
        print(
            f"[{i}/{len(questions)}] folder_rank={rows[-1]['folder_rank']} "
            f"top={rows[-1]['top_folder']}/{rows[-1]['top_page']} "
            f"score={rows[-1]['top_score']}"
        )

    total = max(len(rows), 1)
    summary = {
        "total": len(rows),
        "top_k": args.top_k,
        "search_top_k": search_top_k,
        "aggregate_page": args.aggregate_page,
        "folder_recall": {f"R@{k}": folder_hits[k] / total for k in sorted(folder_hits)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
