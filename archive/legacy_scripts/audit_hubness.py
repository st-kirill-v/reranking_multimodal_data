from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.dataset import load_docbench_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report retrieval hubs across many queries.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/audit_hubness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.retrieval import MultimodalPageRetriever

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

    top1_folders = Counter()
    top1_pages = Counter()
    topk_folders = Counter()
    topk_pages = Counter()
    score_rows = []

    for i, q in enumerate(questions, start=1):
        candidates = retriever.search(q["question"], top_k=args.top_k)
        if candidates:
            top = candidates[0]
            top1_folders[top.folder] += 1
            top1_pages[f"{top.folder}_{top.page}"] += 1
            scores = [candidate.score for candidate in candidates]
            score_rows.append(
                {
                    "question": q["question"],
                    "top1_score": scores[0],
                    "top5_score": scores[min(4, len(scores) - 1)],
                    "topk_score": scores[-1],
                    "top1_top5_gap": scores[0] - scores[min(4, len(scores) - 1)],
                    "top1_topk_gap": scores[0] - scores[-1],
                    "top1_folder": top.folder,
                    "top1_page": top.page,
                    "expected_folder": str(q["folder"]),
                }
            )
        for candidate in candidates:
            topk_folders[candidate.folder] += 1
            topk_pages[f"{candidate.folder}_{candidate.page}"] += 1
        print(
            f"[{i}/{len(questions)}] top1={candidates[0].folder + '/' + str(candidates[0].page) if candidates else None} "
            f"score={candidates[0].score if candidates else None}"
        )

    total = max(len(questions), 1)
    summary = {
        "queries": len(questions),
        "top_k": args.top_k,
        "top1_folder_top10": top1_folders.most_common(10),
        "top1_page_top10": top1_pages.most_common(10),
        "topk_folder_top10": topk_folders.most_common(10),
        "topk_page_top10": topk_pages.most_common(10),
        "top1_page_concentration_top10": sum(count for _, count in top1_pages.most_common(10))
        / total,
        "top1_folder_concentration_top10": sum(count for _, count in top1_folders.most_common(10))
        / total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "score_rows": score_rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
