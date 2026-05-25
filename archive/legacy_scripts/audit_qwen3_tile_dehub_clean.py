from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dehub penalties for a clean Qwen3 tile index."
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
        default="top3_mean",
    )
    parser.add_argument("--folder-penalty", type=float, default=0.003)
    parser.add_argument("--page-penalty", type=float, default=0.004)
    parser.add_argument("--tile-penalty", type=float, default=0.004)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit_qwen3_tile_dehub_clean.json")
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


def keys(record: dict) -> tuple[str, str, str]:
    folder = str(record["folder"])
    page = f"{folder}_{record['page']}"
    tile = f"{page}_{record['tile_id']}"
    return folder, page, tile


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


def adjusted_score(
    score: float,
    record: dict,
    *,
    folder_counts: Counter[str],
    page_counts: Counter[str],
    tile_counts: Counter[str],
    folder_penalty: float,
    page_penalty: float,
    tile_penalty: float,
) -> float:
    folder, page, tile = keys(record)
    return float(
        score
        - folder_penalty * math.log1p(folder_counts[folder])
        - page_penalty * math.log1p(page_counts[page])
        - tile_penalty * math.log1p(tile_counts[tile])
    )


def aggregate_scores(
    tile_hits: list[tuple[float, float, int, dict]], mode: str
) -> tuple[list[dict], list[dict]]:
    page_scores: dict[str, list[tuple[float, float, int, dict]]] = defaultdict(list)
    folder_scores: dict[str, list[tuple[float, float, int, dict]]] = defaultdict(list)

    for rank, (score, raw_score, idx, record) in enumerate(tile_hits, start=1):
        folder, page, _ = keys(record)
        value = (score, raw_score, rank, {**record, "index": idx})
        page_scores[page].append(value)
        folder_scores[folder].append(value)

    def score_group(values: list[tuple[float, float, int, dict]]) -> float:
        scores = sorted([score for score, _, _, _ in values], reverse=True)
        if mode == "max_tile":
            return float(scores[0])
        if mode == "top3_mean":
            return float(np.mean(scores[: min(3, len(scores))]))
        if mode == "top5_mean":
            return float(np.mean(scores[: min(5, len(scores))]))
        if mode == "rrf":
            return float(sum(1.0 / (60.0 + rank) for _, _, rank, _ in values))
        raise ValueError(f"Unknown aggregate mode: {mode}")

    def best_tile(values: list[tuple[float, float, int, dict]]) -> dict:
        score, raw_score, rank, record = max(values, key=lambda item: item[0])
        return {
            "tile_score": float(score),
            "raw_tile_score": float(raw_score),
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


def evaluate_rows(
    *,
    questions: list[dict[str, str]],
    query_hits: list[list[tuple[float, int, dict]]],
    aggregate: str,
    eval_top_k: int,
    examples: int,
    folder_counts: Counter[str] | None = None,
    page_counts: Counter[str] | None = None,
    tile_counts: Counter[str] | None = None,
    folder_penalty: float = 0.0,
    page_penalty: float = 0.0,
    tile_penalty: float = 0.0,
) -> dict:
    recall_cutoffs = sorted({1, 5, 10, 30, eval_top_k})
    recall_cutoffs = [k for k in recall_cutoffs if k <= eval_top_k]
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

    folder_counts = folder_counts or Counter()
    page_counts = page_counts or Counter()
    tile_counts = tile_counts or Counter()

    for question, raw_hits in zip(questions, query_hits):
        adjusted_hits = []
        for raw_score, idx, record in raw_hits:
            score = adjusted_score(
                raw_score,
                record,
                folder_counts=folder_counts,
                page_counts=page_counts,
                tile_counts=tile_counts,
                folder_penalty=folder_penalty,
                page_penalty=page_penalty,
                tile_penalty=tile_penalty,
            )
            adjusted_hits.append((score, raw_score, idx, record))
        adjusted_hits.sort(key=lambda item: item[0], reverse=True)

        pages, folders = aggregate_scores(adjusted_hits, aggregate)
        ranked_folders = [row["folder"] for row in folders[:eval_top_k]]
        ranked_pages = [f"{row['folder']}_{row['page']}" for row in pages[:eval_top_k]]
        expected_folder = str(question["folder"])
        folder_rank = (
            ranked_folders.index(expected_folder) + 1 if expected_folder in ranked_folders else None
        )
        if folder_rank is not None:
            expected_folder_ranks.append(folder_rank)

        for k in recall_cutoffs:
            if expected_folder in ranked_folders[:k]:
                folder_hits[k] += 1
            if any(page_key.startswith(f"{expected_folder}_") for page_key in ranked_pages[:k]):
                page_hits[k] += 1

        q_type = question["type"] or "unknown"
        count_by_type[q_type] += 1
        for k in recall_cutoffs:
            if expected_folder in ranked_folders[:k]:
                hits_by_type[q_type][k] += 1

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
        if folder_rank is None and len(failure_examples) < examples:
            failure_examples.append(row)
        if folder_rank is not None and len(success_examples) < examples:
            success_examples.append(row)

    total = len(rows)
    per_type = {}
    for q_type, type_hits in hits_by_type.items():
        per_type[q_type] = {
            "count": count_by_type[q_type],
            "folder_recall": recall_dict(type_hits, count_by_type[q_type]),
        }

    summary = {
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
    return {
        "summary": summary,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "rows": rows,
    }


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

    search_k = min(args.tile_top_k, index.ntotal)
    query_hits = []
    folder_counts: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    tile_counts: Counter[str] = Counter()

    for i, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, question["question"])
        scores, indices = index.search(query_embedding, search_k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = records[int(idx)]
            hits.append((float(score), int(idx), record))
            folder, page, tile = keys(record)
            folder_counts[folder] += 1
            page_counts[page] += 1
            tile_counts[tile] += 1
        query_hits.append(hits)
        print(f"[collect {i}/{len(questions)}] hits={len(hits)}")

    raw = evaluate_rows(
        questions=questions,
        query_hits=query_hits,
        aggregate=args.aggregate,
        eval_top_k=args.eval_top_k,
        examples=args.examples,
    )
    dehub = evaluate_rows(
        questions=questions,
        query_hits=query_hits,
        aggregate=args.aggregate,
        eval_top_k=args.eval_top_k,
        examples=args.examples,
        folder_counts=folder_counts,
        page_counts=page_counts,
        tile_counts=tile_counts,
        folder_penalty=args.folder_penalty,
        page_penalty=args.page_penalty,
        tile_penalty=args.tile_penalty,
    )

    summary = {
        "total": len(questions),
        "index_name": args.index_name,
        "model_id": args.model_id,
        "encoding_api": "docapi",
        "tile_top_k": args.tile_top_k,
        "eval_top_k": args.eval_top_k,
        "aggregate": args.aggregate,
        "penalties": {
            "folder_penalty": args.folder_penalty,
            "page_penalty": args.page_penalty,
            "tile_penalty": args.tile_penalty,
        },
        "manifest_validation": manifest.get("validation"),
        "tiling": manifest.get("tiling"),
        "raw": raw["summary"],
        "dehub": dehub["summary"],
        "global_hubs": {
            "folder_top10": folder_counts.most_common(10),
            "page_top10": page_counts.most_common(10),
            "tile_top10": tile_counts.most_common(10),
        },
    }

    output = {
        "summary": summary,
        "raw_success_examples": raw["success_examples"],
        "raw_failure_examples": raw["failure_examples"],
        "dehub_success_examples": dehub["success_examples"],
        "dehub_failure_examples": dehub["failure_examples"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Raw folder recall: {summary['raw']['folder_recall']}")
    print(f"Dehub folder recall: {summary['dehub']['folder_recall']}")
    print(f"Raw top hubs: {summary['raw']['hubness']['top1_folder_top10'][:5]}")
    print(f"Dehub top hubs: {summary['dehub']['hubness']['top1_folder_top10'][:5]}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
