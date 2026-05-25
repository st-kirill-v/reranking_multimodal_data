from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluate_colvision_oracle_pages_clean import recall_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Nemotron VL reranking from exported ColPali candidates."
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/colpali_candidates_top100.json")
    )
    parser.add_argument("--reranker-model-id", default="nvidia/llama-nemotron-rerank-vl-1b-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerank-input-k", type=int, default=30)
    parser.add_argument("--rerank-output-k", type=int, default=10)
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--rerank-max-input-tiles", type=int, default=6)
    parser.add_argument("--rerank-max-length", type=int, default=2048)
    parser.add_argument("--no-thumbnail", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_nemotron_rerank_from_candidates.json")
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


def oracle_pages_from_row(row: dict[str, Any]) -> list[int]:
    pages = []
    for item in row.get("oracle_pages", []):
        if isinstance(item, dict) and "page" in item:
            pages.append(int(item["page"]))
        elif isinstance(item, int):
            pages.append(item)
    return pages


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


def reciprocal_rank(rank: int | None, cutoff: int) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / rank


def ndcg_binary(rank: int | None, cutoff: int) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def main() -> None:
    args = parse_args()
    import torch

    with args.candidates.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload["rows"]
    if args.limit > 0:
        rows = rows[: args.limit]

    reranker_model, reranker_processor = load_nemotron_reranker(args)

    cutoffs = sorted({1, 3, 5, args.rerank_output_k})
    cutoffs = [k for k in cutoffs if k <= args.rerank_output_k]
    prerank_folder_hits = {k: 0 for k in cutoffs}
    prerank_oracle_hits = {k: 0 for k in cutoffs}
    rerank_folder_hits = {k: 0 for k in cutoffs}
    rerank_oracle_hits = {k: 0 for k in cutoffs}

    oracle_available = 0
    promoted_oracle_to_top5 = 0
    demoted_oracle_from_top5 = 0
    promoted_folder_to_top5 = 0
    demoted_folder_from_top5 = 0
    mrr_before = []
    mrr_after = []
    ndcg5_before = []
    ndcg5_after = []
    top1_rerank_scores = []
    output_rows = []
    improvement_examples = []
    regression_examples = []

    for qi, row in enumerate(rows, start=1):
        expected = str(row["expected_folder"])
        oracle_pages = oracle_pages_from_row(row)
        if oracle_pages:
            oracle_available += 1

        prerank_candidates = row["candidates"][: args.rerank_input_k]
        reranked_all = rerank_candidates(
            query=row["question"],
            candidates=prerank_candidates,
            model=reranker_model,
            processor=reranker_processor,
            device=args.device,
            batch_size=args.rerank_batch_size,
        )
        reranked = reranked_all[: args.rerank_output_k]
        if reranked and "rerank_score" in reranked[0]:
            top1_rerank_scores.append(float(reranked[0]["rerank_score"]))

        prerank_folder_rank = rank_of_expected_folder(prerank_candidates, expected)
        prerank_oracle_rank = (
            rank_of_oracle_page(prerank_candidates, expected, oracle_pages)
            if oracle_pages
            else None
        )
        rerank_folder_rank = rank_of_expected_folder(reranked, expected)
        rerank_oracle_rank = (
            rank_of_oracle_page(reranked, expected, oracle_pages) if oracle_pages else None
        )

        update_hits(prerank_folder_hits, prerank_folder_rank)
        update_hits(rerank_folder_hits, rerank_folder_rank)
        if oracle_pages:
            update_hits(prerank_oracle_hits, prerank_oracle_rank)
            update_hits(rerank_oracle_hits, rerank_oracle_rank)
            mrr_before.append(reciprocal_rank(prerank_oracle_rank, args.rerank_output_k))
            mrr_after.append(reciprocal_rank(rerank_oracle_rank, args.rerank_output_k))
            ndcg5_before.append(ndcg_binary(prerank_oracle_rank, 5))
            ndcg5_after.append(ndcg_binary(rerank_oracle_rank, 5))

            before_top5 = prerank_oracle_rank is not None and prerank_oracle_rank <= 5
            after_top5 = rerank_oracle_rank is not None and rerank_oracle_rank <= 5
            if not before_top5 and after_top5:
                promoted_oracle_to_top5 += 1
            if before_top5 and not after_top5:
                demoted_oracle_from_top5 += 1

        before_folder_top5 = prerank_folder_rank is not None and prerank_folder_rank <= 5
        after_folder_top5 = rerank_folder_rank is not None and rerank_folder_rank <= 5
        if not before_folder_top5 and after_folder_top5:
            promoted_folder_to_top5 += 1
        if before_folder_top5 and not after_folder_top5:
            demoted_folder_from_top5 += 1

        out_row = {
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "expected_folder": expected,
            "oracle_pages": oracle_pages,
            "prerank_folder_rank": prerank_folder_rank,
            "prerank_oracle_rank": prerank_oracle_rank,
            "rerank_folder_rank": rerank_folder_rank,
            "rerank_oracle_rank": rerank_oracle_rank,
            "top10_prerank": prerank_candidates[:10],
            "top10_reranked": reranked_all[:10],
        }
        output_rows.append(out_row)

        if (
            oracle_pages
            and prerank_oracle_rank is not None
            and rerank_oracle_rank is not None
            and rerank_oracle_rank < prerank_oracle_rank
            and len(improvement_examples) < args.examples
        ):
            improvement_examples.append(out_row)
        if (
            oracle_pages
            and prerank_oracle_rank is not None
            and (rerank_oracle_rank is None or rerank_oracle_rank > prerank_oracle_rank)
            and len(regression_examples) < args.examples
        ):
            regression_examples.append(out_row)

        torch.cuda.empty_cache()
        top_label = (
            f"{prerank_candidates[0]['folder']}/{prerank_candidates[0]['page']}"
            if prerank_candidates
            else None
        )
        rerank_top_label = f"{reranked[0]['folder']}/{reranked[0]['page']}" if reranked else None
        print(
            f"[{qi}/{len(rows)}] prerank_oracle={prerank_oracle_rank} "
            f"rerank_oracle={rerank_oracle_rank} expected={expected} "
            f"top={top_label} rerank_top={rerank_top_label}"
        )

    summary = {
        "total": len(output_rows),
        "source_candidates": str(args.candidates),
        "source_summary": payload.get("summary"),
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
        "oracle_available_rate": oracle_available / max(len(output_rows), 1),
        "prerank_input_folder_recall": recall_dict(prerank_folder_hits, len(output_rows)),
        "prerank_input_oracle_page_recall": recall_dict(prerank_oracle_hits, oracle_available),
        "reranked_folder_recall": recall_dict(rerank_folder_hits, len(output_rows)),
        "reranked_oracle_page_recall": recall_dict(rerank_oracle_hits, oracle_available),
        "mrr_oracle_before": float(np.mean(mrr_before)) if mrr_before else None,
        "mrr_oracle_after": float(np.mean(mrr_after)) if mrr_after else None,
        "ndcg5_oracle_before": float(np.mean(ndcg5_before)) if ndcg5_before else None,
        "ndcg5_oracle_after": float(np.mean(ndcg5_after)) if ndcg5_after else None,
        "movement": {
            "rerank_promoted_oracle_to_top5": promoted_oracle_to_top5,
            "rerank_demoted_oracle_from_top5": demoted_oracle_from_top5,
            "rerank_promoted_folder_to_top5": promoted_folder_to_top5,
            "rerank_demoted_folder_from_top5": demoted_folder_from_top5,
        },
        "score_stats": {"reranker_top1_sigmoid": quantiles(top1_rerank_scores)},
    }
    output = {
        "summary": summary,
        "improvement_examples": improvement_examples,
        "regression_examples": regression_examples,
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Before rerank oracle recall: {summary['prerank_input_oracle_page_recall']}")
    print(f"After rerank oracle recall: {summary['reranked_oracle_page_recall']}")
    print(f"MRR before/after: {summary['mrr_oracle_before']} -> {summary['mrr_oracle_after']}")
    print(
        f"nDCG@5 before/after: {summary['ndcg5_oracle_before']} -> {summary['ndcg5_oracle_after']}"
    )
    print(f"Movement: {summary['movement']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
