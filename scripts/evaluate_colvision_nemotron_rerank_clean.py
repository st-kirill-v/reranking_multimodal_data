from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
        description=(
            "Evaluate ColPali/ColQwen retrieval followed by Nemotron VL reranking "
            "against answer-derived oracle pages."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3_merged"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_merged_clean")
    parser.add_argument("--retriever-model-id", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--reranker-model-id", default="nvidia/llama-nemotron-rerank-vl-1b-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-top-k", type=int, default=100)
    parser.add_argument("--rerank-input-k", type=int, default=30)
    parser.add_argument("--rerank-output-k", type=int, default=10)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--rerank-max-input-tiles", type=int, default=6)
    parser.add_argument("--rerank-max-length", type=int, default=2048)
    parser.add_argument("--no-thumbnail", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_colpali_nemotron_rerank_clean.json")
    )
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


def load_nemotron_reranker(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=args.device,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.reranker_model_id,
        trust_remote_code=True,
        max_input_tiles=args.rerank_max_input_tiles,
        use_thumbnail=not args.no_thumbnail,
        rerank_max_length=args.rerank_max_length,
    )
    return model, processor


def rerank_candidates(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    model: Any,
    processor: Any,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    reranked: list[dict[str, Any]] = []
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        examples = []
        valid = []
        for candidate in batch:
            try:
                with Image.open(candidate["path"]) as img:
                    page_image = img.convert("RGB").copy()
            except OSError as exc:
                skipped = dict(candidate)
                skipped["rerank_error"] = str(exc)
                reranked.append(skipped)
                continue
            examples.append({"question": query, "doc_text": "", "doc_image": page_image})
            valid.append(candidate)

        if not examples:
            continue

        batch_dict = processor.process_queries_documents_crossencoder(examples)
        batch_dict = {
            key: value.to(device)
            for key, value in batch_dict.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            logits = model(**batch_dict, return_dict=True).logits

        for candidate, logit in zip(valid, logits):
            item = dict(candidate)
            item["rerank_score"] = float(torch.sigmoid(logit).reshape(-1)[0].item())
            reranked.append(item)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reranked.sort(key=lambda row: row.get("rerank_score", float("-inf")), reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rerank_rank"] = rank
    return reranked


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
        page_key = f"{candidate['folder']}_{candidate['page']}"
        if page_key in oracle_keys:
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

    if args.rerank_input_k > args.retrieval_top_k:
        raise ValueError("--rerank-input-k must be <= --retrieval-top-k")

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

    retriever_model, retriever_processor, retriever_family = load_model_and_processor(
        args.retriever_model_id,
        args.device,
    )
    reranker_model, reranker_processor = load_nemotron_reranker(args)

    questions = load_questions(args.data_dir, set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    retrieval_cutoffs = sorted({1, 5, 10, 30, args.retrieval_top_k})
    retrieval_cutoffs = [k for k in retrieval_cutoffs if k <= args.retrieval_top_k]
    rerank_cutoffs = sorted({1, 3, 5, args.rerank_output_k})
    rerank_cutoffs = [k for k in rerank_cutoffs if k <= args.rerank_output_k]

    retrieval_folder_hits = {k: 0 for k in retrieval_cutoffs}
    retrieval_oracle_hits = {k: 0 for k in retrieval_cutoffs}
    prerank_input_folder_hits = {k: 0 for k in rerank_cutoffs}
    prerank_input_oracle_hits = {k: 0 for k in rerank_cutoffs}
    rerank_folder_hits = {k: 0 for k in rerank_cutoffs}
    rerank_oracle_hits = {k: 0 for k in rerank_cutoffs}

    oracle_available = 0
    rerank_promoted_oracle_to_top5 = 0
    rerank_demoted_oracle_from_top5 = 0
    rerank_promoted_folder_to_top5 = 0
    rerank_demoted_folder_from_top5 = 0
    rows = []
    failure_examples = []
    improvement_examples = []
    regression_examples = []
    top1_scores = []
    top1_rerank_scores = []

    for qi, question in enumerate(questions, start=1):
        query_embedding = encode_query(
            retriever_model, retriever_processor, question["question"], args.device
        )
        heap: list[tuple[float, int, dict[str, Any]]] = []

        for shard_name, shard_records in records_by_shard.items():
            shard_embeddings = torch.load(
                args.index_dir / "shards" / shard_name, map_location="cpu"
            )
            for offset in range(0, len(shard_records), args.score_batch_size):
                batch_records = shard_records[offset : offset + args.score_batch_size]
                docs = [shard_embeddings[int(record["shard_offset"])] for record in batch_records]
                scores = score_docs(retriever_processor, query_embedding, docs, args.device)
                for score, record in zip(scores, batch_records):
                    item = (float(score), int(record["index"]), record)
                    if len(heap) < args.retrieval_top_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            del shard_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        retrieved = []
        for rank, (score, _, record) in enumerate(ranked, start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(record["index"]),
                    "folder": str(record["folder"]),
                    "page": int(record["page"]),
                    "path": record["path"],
                }
            )
        if retrieved:
            top1_scores.append(float(retrieved[0]["score"]))

        rerank_input = retrieved[: args.rerank_input_k]
        reranked_all = rerank_candidates(
            query=question["question"],
            candidates=rerank_input,
            model=reranker_model,
            processor=reranker_processor,
            device=args.device,
            batch_size=args.rerank_batch_size,
        )
        reranked = reranked_all[: args.rerank_output_k]
        if reranked and "rerank_score" in reranked[0]:
            top1_rerank_scores.append(float(reranked[0]["rerank_score"]))

        expected = str(question["folder"])
        oracle_matches = find_answer_pages(
            args.data_dir / expected,
            question["answer"],
            question.get("evidence", ""),
        )
        oracle_pages = [match.page for match in oracle_matches]
        if oracle_pages:
            oracle_available += 1

        retrieval_folder_rank = rank_of_expected_folder(retrieved, expected)
        retrieval_oracle_rank = (
            rank_of_oracle_page(retrieved, expected, oracle_pages) if oracle_pages else None
        )
        prerank_folder_rank = rank_of_expected_folder(rerank_input, expected)
        prerank_oracle_rank = (
            rank_of_oracle_page(rerank_input, expected, oracle_pages) if oracle_pages else None
        )
        rerank_folder_rank = rank_of_expected_folder(reranked, expected)
        rerank_oracle_rank = (
            rank_of_oracle_page(reranked, expected, oracle_pages) if oracle_pages else None
        )

        update_hits(retrieval_folder_hits, retrieval_folder_rank)
        update_hits(prerank_input_folder_hits, prerank_folder_rank)
        update_hits(rerank_folder_hits, rerank_folder_rank)
        if oracle_pages:
            update_hits(retrieval_oracle_hits, retrieval_oracle_rank)
            update_hits(prerank_input_oracle_hits, prerank_oracle_rank)
            update_hits(rerank_oracle_hits, rerank_oracle_rank)

        if oracle_pages:
            before_top5 = prerank_oracle_rank is not None and prerank_oracle_rank <= 5
            after_top5 = rerank_oracle_rank is not None and rerank_oracle_rank <= 5
            if not before_top5 and after_top5:
                rerank_promoted_oracle_to_top5 += 1
            if before_top5 and not after_top5:
                rerank_demoted_oracle_from_top5 += 1

        before_folder_top5 = prerank_folder_rank is not None and prerank_folder_rank <= 5
        after_folder_top5 = rerank_folder_rank is not None and rerank_folder_rank <= 5
        if not before_folder_top5 and after_folder_top5:
            rerank_promoted_folder_to_top5 += 1
        if before_folder_top5 and not after_folder_top5:
            rerank_demoted_folder_from_top5 += 1

        row = {
            "question": question["question"],
            "answer": question["answer"],
            "type": question["type"],
            "expected_folder": expected,
            "oracle_pages": oracle_pages[:10],
            "retrieval_folder_rank": retrieval_folder_rank,
            "retrieval_oracle_rank": retrieval_oracle_rank,
            "prerank_input_folder_rank": prerank_folder_rank,
            "prerank_input_oracle_rank": prerank_oracle_rank,
            "rerank_folder_rank": rerank_folder_rank,
            "rerank_oracle_rank": rerank_oracle_rank,
            "top10_retrieved": retrieved[:10],
            "top10_reranked": reranked_all[:10],
        }
        rows.append(row)

        if oracle_pages and rerank_oracle_rank is None and len(failure_examples) < args.examples:
            failure_examples.append(row)
        if (
            oracle_pages
            and prerank_oracle_rank is not None
            and rerank_oracle_rank is not None
            and rerank_oracle_rank < prerank_oracle_rank
            and len(improvement_examples) < args.examples
        ):
            improvement_examples.append(row)
        if (
            oracle_pages
            and prerank_oracle_rank is not None
            and (rerank_oracle_rank is None or rerank_oracle_rank > prerank_oracle_rank)
            and len(regression_examples) < args.examples
        ):
            regression_examples.append(row)

        top_label = f"{retrieved[0]['folder']}/{retrieved[0]['page']}" if retrieved else None
        rerank_top_label = f"{reranked[0]['folder']}/{reranked[0]['page']}" if reranked else None
        print(
            f"[{qi}/{len(questions)}] "
            f"retrieval_oracle={retrieval_oracle_rank} prerank_oracle={prerank_oracle_rank} "
            f"rerank_oracle={rerank_oracle_rank} expected={expected} "
            f"top={top_label} rerank_top={rerank_top_label}"
        )

    summary = {
        "total": len(rows),
        "retriever": {
            "index_name": args.index_name,
            "model_id": args.retriever_model_id,
            "family": retriever_family,
            "retrieval_top_k": args.retrieval_top_k,
            "manifest_validation": manifest.get("validation"),
            "index": manifest.get("index"),
        },
        "reranker": {
            "model_id": args.reranker_model_id,
            "rerank_input_k": args.rerank_input_k,
            "rerank_output_k": args.rerank_output_k,
            "batch_size": args.rerank_batch_size,
            "max_input_tiles": args.rerank_max_input_tiles,
            "max_length": args.rerank_max_length,
            "use_thumbnail": not args.no_thumbnail,
        },
        "oracle_available": oracle_available,
        "oracle_available_rate": oracle_available / max(len(rows), 1),
        "colvision_retrieval_folder_recall": recall_dict(retrieval_folder_hits, len(rows)),
        "colvision_retrieval_oracle_page_recall": recall_dict(
            retrieval_oracle_hits, oracle_available
        ),
        "prerank_input_folder_recall": recall_dict(prerank_input_folder_hits, len(rows)),
        "prerank_input_oracle_page_recall": recall_dict(
            prerank_input_oracle_hits, oracle_available
        ),
        "reranked_folder_recall": recall_dict(rerank_folder_hits, len(rows)),
        "reranked_oracle_page_recall": recall_dict(rerank_oracle_hits, oracle_available),
        "movement": {
            "rerank_promoted_oracle_to_top5": rerank_promoted_oracle_to_top5,
            "rerank_demoted_oracle_from_top5": rerank_demoted_oracle_from_top5,
            "rerank_promoted_folder_to_top5": rerank_promoted_folder_to_top5,
            "rerank_demoted_folder_from_top5": rerank_demoted_folder_from_top5,
        },
        "score_stats": {
            "colvision_top1": quantiles(top1_scores),
            "reranker_top1_sigmoid": quantiles(top1_rerank_scores),
        },
    }

    output = {
        "summary": summary,
        "improvement_examples": improvement_examples,
        "regression_examples": regression_examples,
        "failure_examples": failure_examples,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Before rerank oracle recall: {summary['prerank_input_oracle_page_recall']}")
    print(f"After rerank oracle recall: {summary['reranked_oracle_page_recall']}")
    print(f"Movement: {summary['movement']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
