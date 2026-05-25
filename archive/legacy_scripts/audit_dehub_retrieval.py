from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.dataset import load_docbench_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple hubness penalties without rebuilding index."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encoding-api", choices=["legacy_encode", "docapi"], default="legacy_encode"
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--raw-top-k", type=int, default=200)
    parser.add_argument("--eval-top-k", type=int, default=30)
    parser.add_argument("--folder-penalty", type=float, default=0.01)
    parser.add_argument("--page-penalty", type=float, default=0.01)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/audit_dehub_retrieval.json"))
    return parser.parse_args()


def recall(rows: list[dict], top_k: int) -> dict[str, float]:
    hits = {1: 0, 5: 0, 10: 0, top_k: 0}
    total = max(len(rows), 1)
    for row in rows:
        folders = row["ranked_folders"]
        expected = row["expected_folder"]
        for k in hits:
            if expected in folders[:k]:
                hits[k] += 1
    return {f"R@{k}": hits[k] / total for k in sorted(hits)}


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

    raw_rows = []
    folder_counts = Counter()
    page_counts = Counter()
    all_candidates = []

    for i, q in enumerate(questions, start=1):
        candidates = retriever.search(q["question"], top_k=args.raw_top_k)
        all_candidates.append(candidates)
        for candidate in candidates[: args.eval_top_k]:
            folder_counts[candidate.folder] += 1
            page_counts[candidate.doc_id] += 1
        folders = [candidate.folder for candidate in candidates[: args.eval_top_k]]
        raw_rows.append({"expected_folder": str(q["folder"]), "ranked_folders": folders})
        print(f"[collect {i}/{len(questions)}] top={folders[0] if folders else None}")

    dehub_rows = []
    for q, candidates in zip(questions, all_candidates):
        adjusted = []
        for candidate in candidates:
            folder_bias = math.log1p(folder_counts[candidate.folder])
            page_bias = math.log1p(page_counts[candidate.doc_id])
            score = (
                candidate.score - args.folder_penalty * folder_bias - args.page_penalty * page_bias
            )
            adjusted.append((score, candidate))
        adjusted.sort(key=lambda item: item[0], reverse=True)
        folders = [candidate.folder for _, candidate in adjusted[: args.eval_top_k]]
        dehub_rows.append({"expected_folder": str(q["folder"]), "ranked_folders": folders})

    summary = {
        "questions": len(questions),
        "encoding_api": args.encoding_api,
        "raw_top_k": args.raw_top_k,
        "eval_top_k": args.eval_top_k,
        "folder_penalty": args.folder_penalty,
        "page_penalty": args.page_penalty,
        "raw_recall": recall(raw_rows, args.eval_top_k),
        "dehub_recall": recall(dehub_rows, args.eval_top_k),
        "top_folder_hubs": folder_counts.most_common(10),
        "top_page_hubs": page_counts.most_common(10),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
