from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_colvision_oracle_pages_clean import (
    encode_query,
    find_answer_pages,
    load_model_and_processor,
    load_questions,
    recall_dict,
    score_docs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ColPali/ColQwen top-k page candidates for reranking."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3_merged"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_merged_clean")
    parser.add_argument("--model-id", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/colpali_candidates_top100.json"))
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "mean": None, "p50": None, "p75": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def rank_of_expected_folder(candidates: list[dict[str, Any]], expected: str) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if str(candidate["folder"]) == expected:
            return rank
    return None


def rank_of_oracle_page(
    candidates: list[dict[str, Any]], expected: str, oracle_pages: list[int]
) -> int | None:
    oracle_keys = {f"{expected}_{page}" for page in oracle_pages}
    for rank, candidate in enumerate(candidates, start=1):
        if f"{candidate['folder']}_{candidate['page']}" in oracle_keys:
            return rank
    return None


def update_hits(hits: dict[int, int], rank: int | None) -> None:
    if rank is None:
        return
    for cutoff in hits:
        if rank <= cutoff:
            hits[cutoff] += 1


def main() -> None:
    args = parse_args()
    import torch

    metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    records_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in metadata:
        records_by_shard[record["shard"]].append(record)

    model, processor, family = load_model_and_processor(args.model_id, args.device)
    questions = load_questions(args.data_dir, set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    cutoffs = sorted({1, 5, 10, 30, args.top_k})
    cutoffs = [k for k in cutoffs if k <= args.top_k]
    folder_hits = {k: 0 for k in cutoffs}
    oracle_hits = {k: 0 for k in cutoffs}
    oracle_available = 0
    top1_scores = []
    rows = []

    for qi, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, processor, question["question"], args.device)
        heap: list[tuple[float, int, dict[str, Any]]] = []

        for shard_name, shard_records in records_by_shard.items():
            shard_embeddings = torch.load(
                args.index_dir / "shards" / shard_name, map_location="cpu"
            )
            for offset in range(0, len(shard_records), args.score_batch_size):
                batch_records = shard_records[offset : offset + args.score_batch_size]
                docs = [shard_embeddings[int(record["shard_offset"])] for record in batch_records]
                scores = score_docs(processor, query_embedding, docs, args.device)
                for score, record in zip(scores, batch_records):
                    item = (float(score), int(record["index"]), record)
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            del shard_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        candidates = []
        for rank, (score, _, record) in enumerate(ranked, start=1):
            candidates.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(record["index"]),
                    "folder": str(record["folder"]),
                    "page": int(record["page"]),
                    "path": record["path"],
                }
            )
        if candidates:
            top1_scores.append(float(candidates[0]["score"]))

        expected = str(question["folder"])
        oracle_matches = find_answer_pages(
            args.data_dir / expected,
            question["answer"],
            question.get("evidence", ""),
        )
        oracle_pages = [match.page for match in oracle_matches]
        if oracle_pages:
            oracle_available += 1

        folder_rank = rank_of_expected_folder(candidates, expected)
        oracle_rank = (
            rank_of_oracle_page(candidates, expected, oracle_pages) if oracle_pages else None
        )
        update_hits(folder_hits, folder_rank)
        if oracle_pages:
            update_hits(oracle_hits, oracle_rank)

        rows.append(
            {
                "question": question["question"],
                "answer": question["answer"],
                "evidence": question.get("evidence", ""),
                "type": question["type"],
                "expected_folder": expected,
                "oracle_pages": [asdict(match) for match in oracle_matches[:10]],
                "folder_rank": folder_rank,
                "oracle_rank": oracle_rank,
                "candidates": candidates,
            }
        )
        print(
            f"[{qi}/{len(questions)}] folder_rank={folder_rank} oracle_rank={oracle_rank} expected={expected}"
        )

    summary = {
        "total": len(rows),
        "top_k": args.top_k,
        "index_name": args.index_name,
        "model_id": args.model_id,
        "family": family,
        "manifest_validation": manifest.get("validation"),
        "index": manifest.get("index"),
        "folder_recall": recall_dict(folder_hits, len(rows)),
        "oracle_available": oracle_available,
        "oracle_available_rate": oracle_available / max(len(rows), 1),
        "oracle_page_recall": recall_dict(oracle_hits, oracle_available),
        "score_stats": {"top1": quantiles(top1_scores)},
    }
    output = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
