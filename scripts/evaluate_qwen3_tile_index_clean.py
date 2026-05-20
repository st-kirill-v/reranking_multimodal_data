from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean Qwen3 tile index with page/folder aggregation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="tiles_qwen3_docapi_grid2x3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-top-k", type=int, default=1000)
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument(
        "--aggregate",
        choices=["max_tile", "top3_mean", "top5_mean", "rrf"],
        default="max_tile",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/retrieval_eval_qwen3_tile_clean.json")
    )
    return parser.parse_args()


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('tiles_')}.json"


def load_questions(data_dir: Path, types: set[str]) -> list[dict[str, str]]:
    questions = []
    for jsonl_file in sorted(data_dir.glob("*/*_qa.jsonl")):
        folder = jsonl_file.parent.name
        if not folder.isdigit():
            continue
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if types and row.get("type") not in types:
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


def encode_query(model: Any, query: str) -> np.ndarray:
    embedding = model.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray([embedding], dtype=np.float32)


def recall_dict(hits: dict[int, int], total: int) -> dict[str, float]:
    denominator = max(total, 1)
    return {f"R@{k}": hits[k] / denominator for k in sorted(hits)}


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "mean": None, "p50": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def aggregate_scores(
    tile_hits: list[tuple[float, int, dict]], mode: str
) -> tuple[list[dict], list[dict]]:
    page_scores: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    folder_scores: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)

    for rank, (score, idx, record) in enumerate(tile_hits, start=1):
        page_key = f"{record['folder']}_{record['page']}"
        folder_key = str(record["folder"])
        value = (score, rank, {**record, "index": idx})
        page_scores[page_key].append(value)
        folder_scores[folder_key].append(value)

    def score_group(values: list[tuple[float, int, dict]]) -> float:
        scores = sorted([score for score, _, _ in values], reverse=True)
        if mode == "max_tile":
            return float(scores[0])
        if mode == "top3_mean":
            return float(np.mean(scores[: min(3, len(scores))]))
        if mode == "top5_mean":
            return float(np.mean(scores[: min(5, len(scores))]))
        if mode == "rrf":
            return float(sum(1.0 / (60.0 + rank) for _, rank, _ in values))
        raise ValueError(f"Unknown aggregate mode: {mode}")

    def best_tile(values: list[tuple[float, int, dict]]) -> dict:
        score, rank, record = max(values, key=lambda item: item[0])
        return {
            "tile_score": float(score),
            "tile_rank": int(rank),
            "tile_id": record["tile_id"],
            "crop_box": record["crop_box"],
            "index": int(record["index"]),
            "path": record["path"],
        }

    pages = []
    for page_key, values in page_scores.items():
        folder, page = page_key.split("_", 1)
        pages.append(
            {
                "folder": folder,
                "page": int(page),
                "score": score_group(values),
                "hits": len(values),
                "best_tile": best_tile(values),
            }
        )
    pages.sort(key=lambda row: row["score"], reverse=True)

    folders = []
    for folder, values in folder_scores.items():
        folders.append(
            {
                "folder": folder,
                "score": score_group(values),
                "hits": len(values),
                "best_tile": best_tile(values),
            }
        )
    folders.sort(key=lambda row: row["score"], reverse=True)
    return pages, folders


def main() -> None:
    args = parse_args()
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    index_path = args.index_dir / f"{args.index_name}.index"
    metadata_path = args.index_dir / metadata_filename(args.index_name)
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"

    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if index.ntotal != len(records):
        raise ValueError(f"Index/metadata mismatch: {index.ntotal} != {len(records)}")

    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    model_kwargs = {"dtype": dtype} if dtype is not None else {}
    model = SentenceTransformer(
        args.model_id,
        trust_remote_code=True,
        device=args.device,
        model_kwargs=model_kwargs,
    )

    questions = load_questions(args.data_dir, set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    recall_cutoffs = sorted({1, 5, 10, 30, args.eval_top_k})
    recall_cutoffs = [k for k in recall_cutoffs if k <= args.eval_top_k]
    folder_hits = {k: 0 for k in recall_cutoffs}
    page_hits = {k: 0 for k in recall_cutoffs}
    hits_by_type: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in recall_cutoffs})
    count_by_type: Counter[str] = Counter()

    top1_folder_counts: Counter[str] = Counter()
    top1_page_counts: Counter[str] = Counter()
    top1_tile_counts: Counter[str] = Counter()
    top1_scores = []
    expected_folder_ranks = []
    rows = []
    success_examples = []
    failure_examples = []

    search_k = min(args.tile_top_k, index.ntotal)
    for i, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, question["question"])
        scores, indices = index.search(query_embedding, search_k)
        tile_hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = records[int(idx)]
            tile_hits.append((float(score), int(idx), record))

        pages, folders = aggregate_scores(tile_hits, args.aggregate)
        ranked_folders = [row["folder"] for row in folders[: args.eval_top_k]]
        ranked_pages = [f"{row['folder']}_{row['page']}" for row in pages[: args.eval_top_k]]
        expected_folder = str(question["folder"])
        folder_rank = (
            ranked_folders.index(expected_folder) + 1 if expected_folder in ranked_folders else None
        )
        if folder_rank is not None:
            expected_folder_ranks.append(folder_rank)

        for k in recall_cutoffs:
            if expected_folder in ranked_folders[:k]:
                folder_hits[k] += 1

        q_type = question["type"] or "unknown"
        count_by_type[q_type] += 1
        for k in recall_cutoffs:
            if expected_folder in ranked_folders[:k]:
                hits_by_type[q_type][k] += 1

        # Page recall is only a weak proxy here because DocBench does not provide answer pages.
        # It checks whether any page from the expected folder appears in top-k page candidates.
        for k in recall_cutoffs:
            if any(page_key.startswith(f"{expected_folder}_") for page_key in ranked_pages[:k]):
                page_hits[k] += 1

        if folders:
            top1_folder_counts[folders[0]["folder"]] += 1
            top1_scores.append(float(folders[0]["score"]))
        if pages:
            top_page_key = f"{pages[0]['folder']}_{pages[0]['page']}"
            top1_page_counts[top_page_key] += 1
            tile = pages[0]["best_tile"]
            top1_tile_counts[f"{top_page_key}_{tile['tile_id']}"] += 1

        row = {
            "question": question["question"],
            "answer": question["answer"],
            "type": q_type,
            "expected_folder": expected_folder,
            "folder_rank": folder_rank,
            "top10_folders": folders[:10],
            "top10_pages": pages[:10],
        }
        rows.append(row)
        if folder_rank is None and len(failure_examples) < args.examples:
            failure_examples.append(row)
        if folder_rank is not None and len(success_examples) < args.examples:
            success_examples.append(row)

        top = folders[0] if folders else {}
        print(
            f"[{i}/{len(questions)}] folder_rank={folder_rank} "
            f"expected={expected_folder} top={top.get('folder')} score={top.get('score')}"
        )

    total = len(rows)
    per_type = {}
    for q_type, type_hits in hits_by_type.items():
        per_type[q_type] = {
            "count": count_by_type[q_type],
            "folder_recall": recall_dict(type_hits, count_by_type[q_type]),
        }

    summary = {
        "total": total,
        "index_name": args.index_name,
        "model_id": args.model_id,
        "encoding_api": "docapi",
        "tile_top_k": args.tile_top_k,
        "eval_top_k": args.eval_top_k,
        "aggregate": args.aggregate,
        "manifest_validation": manifest.get("validation"),
        "tiling": manifest.get("tiling"),
        "folder_recall": recall_dict(folder_hits, total),
        "page_proxy_recall": recall_dict(page_hits, total),
        "per_type": per_type,
        "rank_stats_for_found_expected_folder": {
            "count": len(expected_folder_ranks),
            "mean": float(np.mean(expected_folder_ranks)) if expected_folder_ranks else None,
            "median": float(np.median(expected_folder_ranks)) if expected_folder_ranks else None,
            "min": min(expected_folder_ranks) if expected_folder_ranks else None,
            "max": max(expected_folder_ranks) if expected_folder_ranks else None,
        },
        "score_stats": {"top1_folder_score": quantiles(top1_scores)},
        "hubness": {
            "top1_folder_top10": top1_folder_counts.most_common(10),
            "top1_page_top10": top1_page_counts.most_common(10),
            "top1_tile_top10": top1_tile_counts.most_common(10),
            "top1_folder_concentration_top10": sum(
                count for _, count in top1_folder_counts.most_common(10)
            )
            / max(total, 1),
            "top1_page_concentration_top10": sum(
                count for _, count in top1_page_counts.most_common(10)
            )
            / max(total, 1),
        },
    }

    output = {
        "summary": summary,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Folder recall: {summary['folder_recall']}")
    print(f"Hubness top folders: {summary['hubness']['top1_folder_top10'][:5]}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
