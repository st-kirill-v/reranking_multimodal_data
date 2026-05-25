from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.dataset import load_docbench_questions
from src.mmrag.indexing import FaissPageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate document-level routing from existing page vectors."
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
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--score-mode", choices=["centroid", "max_page", "top5_mean"], default="centroid"
    )
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/audit_document_routing.json"))
    return parser.parse_args()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def main() -> None:
    args = parse_args()
    from src.mmrag.embeddings import Qwen3PageEmbedder

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    page_index = FaissPageIndex(config).load()
    vectors = page_index.vectors()
    embedder = Qwen3PageEmbedder(config.embedder)

    by_folder: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(page_index.records):
        by_folder[record.folder].append(idx)
    folders = sorted(by_folder.keys(), key=lambda x: int(x) if x.isdigit() else x)
    centroids = []
    for folder in folders:
        doc_vectors = vectors[by_folder[folder]]
        centroids.append(np.mean(doc_vectors, axis=0))
    centroids = l2_normalize(np.asarray(centroids, dtype=np.float32))

    questions = load_docbench_questions(config.paths.data_dir, question_types=set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    hits = {1: 0, 5: 0, 10: 0, args.top_k: 0}
    rows = []
    for i, q in enumerate(questions, start=1):
        query = embedder.encode_query(q["question"])[0]
        if args.score_mode == "centroid":
            scores = centroids @ query
        else:
            scores = []
            for folder in folders:
                page_scores = vectors[by_folder[folder]] @ query
                if args.score_mode == "max_page":
                    scores.append(float(np.max(page_scores)))
                else:
                    top = np.sort(page_scores)[-min(5, len(page_scores)) :]
                    scores.append(float(np.mean(top)))
            scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-scores)
        ranked_folders = [folders[int(idx)] for idx in order[: args.top_k]]
        expected = str(q["folder"])
        for k in hits:
            if expected in ranked_folders[:k]:
                hits[k] += 1
        rank = ranked_folders.index(expected) + 1 if expected in ranked_folders else None
        rows.append(
            {
                "question": q["question"],
                "expected_folder": expected,
                "folder_rank": rank,
                "top_folders": ranked_folders[:10],
                "top_scores": [float(scores[int(idx)]) for idx in order[:10]],
            }
        )
        print(
            f"[{i}/{len(questions)}] folder_rank={rank} "
            f"top={ranked_folders[0] if ranked_folders else None} score={float(scores[order[0]]) if len(order) else None}"
        )

    total = max(len(rows), 1)
    summary = {
        "questions": len(rows),
        "top_k": args.top_k,
        "score_mode": args.score_mode,
        "encoding_api": args.encoding_api,
        "document_recall": {f"R@{k}": hits[k] / total for k in sorted(hits)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
