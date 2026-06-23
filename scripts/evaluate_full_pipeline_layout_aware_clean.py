from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cropping.layout_aware_eval import (  # noqa: E402
    compact_case,
    compute_extended_metrics,
    compute_similarity,
    create_generator,
    effective_context_settings_layout,
    load_context_images,
    summarize_with_crop_metrics,
    write_case_csv,
)
from src.retrieval.colvision import (  # noqa: E402
    encode_query,
    find_answer_pages,
    load_model_and_processor,
    score_docs,
)
from src.mmrag.dataset import load_docbench_questions  # noqa: E402
from src.mmrag.schema import RetrievalCandidate  # noqa: E402
from src.reranking.fusion import (  # noqa: E402
    FusionWeights,
    attach_text_evidence,
    fuse_candidates,
    load_text_evidence_map,
)
from src.reranking.text_reranker import create_text_reranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the honest full DocBench pipeline: question -> page retrieval -> "
            "Nemotron reranking -> layout-aware context -> Qwen VLM answer -> metrics."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument(
        "--types",
        nargs="*",
        default=["multimodal-t", "multimodal-f"],
        help='Question types to evaluate, or "all" for the full dataset.',
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3_merged"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_merged_clean")
    parser.add_argument(
        "--retriever-backend",
        choices=["colvision", "nemotron_image"],
        default="colvision",
    )
    parser.add_argument("--retriever-model-id", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--retrieval-device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=1)

    parser.add_argument("--first-stage-top-k", type=int, default=30)
    parser.add_argument(
        "--reranker-mode",
        choices=["nemotron", "nemotron_text_image", "adaptive", "threshold_skip", "none"],
        default="nemotron",
        help='Use Nemotron reranking or pass retrieved pages through unchanged with "none".',
    )
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--rerank-device", default="cuda")
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument(
        "--rerank-text-source-fields",
        nargs="*",
        default=["table_text", "caption", "page_text", "ocr"],
        help="Text fields used only by --reranker-mode nemotron_text_image.",
    )
    parser.add_argument(
        "--rerank-text-max-chars",
        type=int,
        default=4096,
        help="Maximum extracted text characters per candidate for text+image reranking.",
    )
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument(
        "--adaptive-threshold-top1",
        type=float,
        default=None,
        help="Skip expensive reranking when retrieval top-1 score is at least this value.",
    )
    parser.add_argument(
        "--adaptive-threshold-gap",
        type=float,
        default=None,
        help="Skip expensive reranking when retrieval top-1 minus top-2 score is at least this value.",
    )
    parser.add_argument(
        "--adaptive-high-confidence-strategy",
        choices=["no_reranker", "none", "skip"],
        default="no_reranker",
        help="Strategy used for high-confidence retrieval route.",
    )
    parser.add_argument(
        "--threshold-skip-top1",
        type=float,
        default=0.8,
        help="Skip reranking when retrieval top-1 score is at least this value.",
    )
    parser.add_argument(
        "--threshold-skip-gap",
        type=float,
        default=0.1,
        help="Skip reranking when retrieval top-1 minus top-2 score is at least this value.",
    )

    parser.add_argument(
        "--fusion-mode",
        choices=["none", "score_fusion"],
        default="none",
        help="Optional query-adaptive image+text score fusion after first-stage retrieval.",
    )
    parser.add_argument(
        "--fusion-source-fields",
        nargs="*",
        default=["page_text", "caption", "table_text"],
        help="Text evidence fields for fusion. OCR can be added after pages_ocr.json exists.",
    )
    parser.add_argument("--fusion-alpha", type=float, default=1.0)
    parser.add_argument("--fusion-beta", type=float, default=0.2)
    parser.add_argument("--fusion-gamma", type=float, default=1.0)
    parser.add_argument("--fusion-lambda-number", type=float, default=0.05)
    parser.add_argument("--fusion-lambda-keyword", type=float, default=0.05)
    parser.add_argument("--fusion-lambda-exact-phrase", type=float, default=0.05)
    parser.add_argument("--fusion-lambda-table-header", type=float, default=0.20)
    parser.add_argument("--fusion-text-reranker-model-id", default="BAAI/bge-reranker-large")
    parser.add_argument("--fusion-text-reranker-device", default="cuda")
    parser.add_argument("--fusion-text-reranker-batch-size", type=int, default=4)
    parser.add_argument("--fusion-text-reranker-max-length", type=int, default=512)
    parser.add_argument(
        "--fusion-text-reranker-backend",
        choices=["cross_encoder", "lexical", "none"],
        default="cross_encoder",
    )
    parser.add_argument("--fusion-no-trust-remote-code", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-long-edge", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-images", type=int, default=5)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["concise", "legacy", "think_answer", "smart_universal"],
        default="concise",
    )
    parser.add_argument("--answer-refine", choices=["none", "text"], default="none")
    parser.add_argument(
        "--multimodal-generation-backend",
        choices=["local_vlm", "openai_compatible_vlm"],
        default="local_vlm",
    )
    parser.add_argument("--openai-vlm-base-url", default="")
    parser.add_argument("--openai-vlm-model", default="openai/qwen3-vl-30b")
    parser.add_argument("--openai-vlm-api-key-env", default="OPENAI_COMPAT_API_KEY")
    parser.add_argument("--openai-vlm-api-key", default=None)
    parser.add_argument("--openai-vlm-temperature", type=float, default=0.0)
    parser.add_argument("--openai-vlm-max-tokens", type=int, default=0)
    parser.add_argument("--openai-vlm-timeout", type=float, default=180.0)

    parser.add_argument("--top-pages", type=int, default=5)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "text_top3_visual_top5"],
        default="text_top3_visual_top5",
    )
    parser.add_argument("--text-top-pages", type=int, default=3)
    parser.add_argument("--visual-top-pages", type=int, default=5)
    parser.add_argument(
        "--crop-policy",
        choices=[
            "none",
            "top_2x2",
            "visual_2x2",
            "visual_main",
            "layout_aware",
            "layout_aware_v2",
        ],
        default="none",
    )
    parser.add_argument("--crop-top-n", type=int, default=1)
    parser.add_argument(
        "--visual-crop-policy",
        choices=[
            "none",
            "top_2x2",
            "visual_2x2",
            "visual_main",
            "layout_aware",
            "layout_aware_v2",
        ],
        default="layout_aware_v2",
    )
    parser.add_argument(
        "--layout-context-mode",
        choices=["crop_only", "full_page_plus_crop"],
        default="full_page_plus_crop",
    )
    parser.add_argument(
        "--vlm-text-context-mode",
        choices=["none", "selected_pages"],
        default="none",
        help="Optionally send extracted text evidence together with page images to the VLM.",
    )
    parser.add_argument(
        "--vlm-text-source-fields",
        nargs="*",
        default=["ocr", "page_text", "caption", "table_text"],
        help="Text evidence fields used only when --vlm-text-context-mode is enabled.",
    )
    parser.add_argument(
        "--vlm-text-max-chars",
        type=int,
        default=12000,
        help="Maximum characters of text evidence sent to the VLM with images.",
    )
    parser.add_argument(
        "--debug-crop-dir",
        type=Path,
        default=Path("data/debug_crops/full_pipeline_layout_aware_v2"),
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--print-think", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_full_pipeline_layout_aware_308.json"),
    )
    return parser.parse_args()


def create_multimodal_generator(args: argparse.Namespace) -> Any:
    if args.multimodal_generation_backend == "openai_compatible_vlm":
        from src.generation.openai_compatible_vlm import create_openai_compatible_vlm

        return create_openai_compatible_vlm(
            base_url=args.openai_vlm_base_url,
            model=args.openai_vlm_model,
            api_key_env=args.openai_vlm_api_key_env,
            api_key=args.openai_vlm_api_key,
            temperature=args.openai_vlm_temperature,
            max_tokens=args.openai_vlm_max_tokens or args.max_new_tokens,
            timeout=args.openai_vlm_timeout,
        )
    return create_generator(args)


def ordered_context_image_paths(
    context_candidate_rows: list[dict[str, Any]],
    selected_crop: dict[str, Any] | None,
    layout_context_mode: str,
) -> list[str | None]:
    full_page_paths = [
        str(row.get("path")) if row.get("path") else None for row in context_candidate_rows
    ]
    if not selected_crop or not selected_crop.get("crop_path"):
        return full_page_paths
    if layout_context_mode != "full_page_plus_crop":
        return [str(selected_crop.get("crop_path"))]

    selected_label = selected_crop.get("page_label")
    selected_idx = 0
    if selected_label:
        for idx, row in enumerate(context_candidate_rows):
            if f"{row.get('folder')}/{row.get('page')}" == selected_label:
                selected_idx = idx
                break
    selected_page = full_page_paths[selected_idx] if full_page_paths else None
    remaining = [path for idx, path in enumerate(full_page_paths) if idx != selected_idx]
    return [selected_page, str(selected_crop.get("crop_path")), *remaining]


def build_vlm_text_context(
    candidates: list[RetrievalCandidate],
    evidence_map: dict[tuple[str, int], dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, bool, list[dict[str, Any]]]:
    if max_chars <= 0:
        return "", False, []

    blocks: list[str] = []
    debug_rows: list[dict[str, Any]] = []
    total_chars = 0
    truncated = False
    for rank, candidate in enumerate(candidates, start=1):
        evidence = evidence_map.get((str(candidate.folder), int(candidate.page)), {})
        text = str(evidence.get("text") or "").strip()
        fields = list(evidence.get("available_source_fields") or [])
        if not text:
            debug_rows.append(
                {
                    "rank": rank,
                    "doc_id": str(candidate.folder),
                    "page": int(candidate.page),
                    "chars": 0,
                    "fields": fields,
                    "included": False,
                }
            )
            continue

        header = (
            f"[rank={rank} doc_id={candidate.folder} page={candidate.page} "
            f"fields={','.join(fields) if fields else 'unknown'}]\n"
        )
        remaining = max_chars - total_chars - len(header)
        if remaining <= 0:
            truncated = True
            break
        block_text = text[:remaining]
        if len(block_text) < len(text):
            truncated = True
        block = f"{header}{block_text}"
        blocks.append(block)
        total_chars += len(block) + 2
        debug_rows.append(
            {
                "rank": rank,
                "doc_id": str(candidate.folder),
                "page": int(candidate.page),
                "chars": len(block_text),
                "fields": fields,
                "included": True,
            }
        )
        if truncated:
            break

    return "\n\n".join(blocks), truncated, debug_rows


class ColVisionRetriever:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
        self.manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing ColVision metadata: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.manifest = {}
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        self.records_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.metadata:
            self.records_by_shard[record["shard"]].append(record)
        self.model, self.processor, self.family = load_model_and_processor(
            args.retriever_model_id, args.retrieval_device
        )

    def search(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        import torch

        query_embedding = encode_query(
            self.model, self.processor, query, self.args.retrieval_device
        )
        heap: list[tuple[float, int, dict[str, Any]]] = []
        for shard_name, shard_records in self.records_by_shard.items():
            shard_embeddings = torch.load(
                self.args.index_dir / "shards" / shard_name, map_location="cpu"
            )
            for offset in range(0, len(shard_records), self.args.score_batch_size):
                batch_records = shard_records[offset : offset + self.args.score_batch_size]
                docs = [shard_embeddings[int(record["shard_offset"])] for record in batch_records]
                scores = score_docs(
                    self.processor,
                    query_embedding,
                    docs,
                    self.args.retrieval_device,
                )
                for score, record in zip(scores, batch_records):
                    item = (float(score), int(record["index"]), record)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            del shard_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        candidates = []
        for rank, (score, _, record) in enumerate(
            sorted(heap, key=lambda item: item[0], reverse=True),
            start=1,
        ):
            candidates.append(
                RetrievalCandidate(
                    folder=str(record["folder"]),
                    page=int(record["page"]),
                    path=Path(record["path"]),
                    score=float(score),
                    rank=rank,
                    index=int(record["index"]),
                    source=self.family,
                )
            )
        return candidates


def create_retriever(args: argparse.Namespace) -> Any:
    if args.retriever_backend == "nemotron_image":
        from src.retrieval.nemotron_image import NemotronImageRetriever

        return NemotronImageRetriever(
            index_dir=args.index_dir,
            index_name=args.index_name,
            model_id=args.retriever_model_id,
            device=args.retrieval_device,
        )
    return ColVisionRetriever(args)


def candidate_to_json(candidate: RetrievalCandidate, rank: int | None = None) -> dict[str, Any]:
    row = candidate.to_json()
    if rank is not None:
        row["rank"] = rank
    dynamic_fields = [
        "text_rerank_score",
        "text_rerank_rank",
        "fusion_score",
        "fusion_rank",
        "fusion_features",
        "evidence_fields",
    ]
    for field in dynamic_fields:
        if hasattr(candidate, field):
            row[field] = getattr(candidate, field)
    return row


def expand_with_neighbors_local(
    candidates: list[RetrievalCandidate],
    radius: int,
    final_limit: int | None = None,
) -> list[RetrievalCandidate]:
    if radius <= 0:
        return candidates[:final_limit] if final_limit else candidates

    expanded: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        for delta in range(-radius, radius + 1):
            page = candidate.page + delta
            if page < 1:
                continue
            path = Path(candidate.path).parent / f"page_{page}.png"
            if not path.exists():
                continue
            key = f"{candidate.folder}_{page}"
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                RetrievalCandidate(
                    folder=candidate.folder,
                    page=page,
                    path=path,
                    score=candidate.score if delta == 0 else candidate.score * 0.8,
                    rank=candidate.rank,
                    index=candidate.index,
                    source="neighbor_expansion" if delta else candidate.source,
                    rerank_score=candidate.rerank_score if delta == 0 else None,
                )
            )

    expanded.sort(
        key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
        reverse=True,
    )
    return expanded[:final_limit] if final_limit else expanded


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    question_types = (
        None if "all" in {str(item).lower() for item in args.types} else set(args.types)
    )
    questions = load_docbench_questions(
        args.data_dir,
        question_types=question_types,
    )
    rows: list[dict[str, Any]] = []
    for question_id, question in enumerate(questions, start=1):
        expected_folder = str(question.get("folder"))
        rows.append(
            {
                "question_id": question_id,
                "folder": expected_folder,
                "expected_folder": expected_folder,
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
                "type": question.get("type", ""),
                "evidence": question.get("evidence", ""),
                "oracle_pages": [],
            }
        )
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def latency_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row.get(key, 0.0)) for row in rows]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def optional_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean([float(value) for value in values]))


def normalize_unit(unit: str) -> str:
    unit = (unit or "").lower().strip()
    unit = unit.replace("mtco₂-e", "mtco2e").replace("mtco2-e", "mtco2e")
    unit = unit.replace("$", "usd").replace("€", "eur").replace("£", "gbp").replace("₹", "inr")
    aliases = {
        "percent": "%",
        "percentage": "%",
        "token": "tokens",
        "point": "points",
        "lb": "lbs",
        "crore": "crores",
        "share": "shares",
        "employee": "employees",
        "document": "documents",
        "dialogue": "dialogues",
        "store": "stores",
        "restaurant": "restaurants",
        "sentence": "sentences",
        "page": "pages",
    }
    return aliases.get(unit, unit)


def extract_numeric_mentions(text: str) -> list[dict[str, Any]]:
    unit_pattern = (
        r"%|percent|percentage|million|billion|thousand|tokens?|points?|pp|kg|lbs?|"
        r"mtco2e|mtco₂-e|co2e|dkk|usd|eur|chf|rmb|crores?|shares?|employees?|"
        r"documents?|dialogues?|stores?|restaurants?|sentences?|pages?"
    )
    pattern = re.compile(
        rf"(?P<prefix>[$€£₹])?\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{unit_pattern})?",
        re.IGNORECASE,
    )
    mentions: list[dict[str, Any]] = []
    for match in pattern.finditer(text or ""):
        raw_num = match.group("num").replace(",", "")
        try:
            value = float(raw_num)
        except ValueError:
            continue
        unit = normalize_unit(match.group("unit") or match.group("prefix") or "")
        mentions.append(
            {
                "value": value,
                "unit": unit,
                "is_integer": "." not in raw_num,
                "is_percentage": unit == "%",
            }
        )
    return mentions


def exact_number_match(expected: dict[str, Any], generated: dict[str, Any]) -> bool:
    return abs(float(expected["value"]) - float(generated["value"])) <= 1e-9


def relaxed_number_match(expected: dict[str, Any], generated: dict[str, Any]) -> bool:
    left = float(expected["value"])
    right = float(generated["value"])
    if expected.get("is_percentage") or generated.get("is_percentage"):
        return abs(left - right) <= 0.1
    if expected.get("is_integer") and generated.get("is_integer"):
        return exact_number_match(expected, generated)
    return abs(left - right) / max(abs(left), 1e-9) <= 0.01


def all_expected_numbers_match(
    expected: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    relaxed: bool,
) -> bool | None:
    if not expected:
        return None
    matcher = relaxed_number_match if relaxed else exact_number_match
    return all(any(matcher(exp, gen) for gen in generated) for exp in expected)


def compute_unit_match(
    expected: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> bool | None:
    expected_units = {item["unit"] for item in expected if item.get("unit")}
    generated_units = {item["unit"] for item in generated if item.get("unit")}
    if not expected_units:
        return None
    return expected_units.issubset(generated_units)


def normalize_entity_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    suffixes = {
        "the",
        "ltd",
        "limited",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "group",
        "plc",
        "llc",
    }
    tokens = [
        token
        for token in text.split()
        if token not in suffixes and not token.isdigit() and len(token) > 1
    ]
    return " ".join(tokens)


def compute_entity_match(generated: str, expected: str) -> bool | None:
    expected_norm = normalize_entity_text(expected)
    generated_norm = normalize_entity_text(generated)
    if not expected_norm or re.fullmatch(r"[\d\s.%-]+", expected_norm):
        return None
    if expected_norm in generated_norm or generated_norm in expected_norm:
        return True
    expected_tokens = set(expected_norm.split())
    generated_tokens = set(generated_norm.split())
    if not expected_tokens:
        return None
    return len(expected_tokens & generated_tokens) / len(expected_tokens) >= 0.8


def doc_hit(
    candidates: list[RetrievalCandidate], expected_folder: str | None, top_k: int
) -> bool | None:
    if expected_folder is None:
        return None
    return any(str(candidate.folder) == str(expected_folder) for candidate in candidates[:top_k])


def page_hit(
    candidates: list[RetrievalCandidate], oracle_pages: list[Any], top_k: int
) -> bool | None:
    if not oracle_pages:
        return None
    normalized = set()
    for page in oracle_pages:
        if isinstance(page, dict) and "page" in page:
            normalized.add(str(page["page"]))
        else:
            normalized.add(str(page))
    return any(
        f"{candidate.folder}/{candidate.page}" in normalized
        or f"{candidate.folder}_{candidate.page}" in normalized
        or str(candidate.page) in normalized
        for candidate in candidates[:top_k]
    )


def extra_row_metrics(
    *,
    generated: str,
    expected: str,
    retrieved: list[RetrievalCandidate],
    reranked: list[RetrievalCandidate],
    expected_folder: str | None,
    oracle_pages: list[Any],
) -> dict[str, Any]:
    expected_numbers = extract_numeric_mentions(expected)
    generated_numbers = extract_numeric_mentions(generated)
    return {
        "numeric_exact_match": all_expected_numbers_match(
            expected_numbers, generated_numbers, relaxed=False
        ),
        "numeric_relaxed_match": all_expected_numbers_match(
            expected_numbers, generated_numbers, relaxed=True
        ),
        "unit_match": compute_unit_match(expected_numbers, generated_numbers),
        "entity_match": compute_entity_match(generated, expected),
        "doc_hit_at_1": doc_hit(retrieved, expected_folder, 1),
        "doc_hit_at_5": doc_hit(retrieved, expected_folder, 5),
        "doc_hit_at_10": doc_hit(retrieved, expected_folder, 10),
        "doc_hit_at_30": doc_hit(retrieved, expected_folder, 30),
        "reranked_doc_hit_at_1": doc_hit(reranked, expected_folder, 1),
        "reranked_doc_hit_at_5": doc_hit(reranked, expected_folder, 5),
        "reranked_doc_hit_at_10": doc_hit(reranked, expected_folder, 10),
        "page_hit_at_1": page_hit(retrieved, oracle_pages, 1),
        "page_hit_at_5": page_hit(retrieved, oracle_pages, 5),
        "page_hit_at_10": page_hit(retrieved, oracle_pages, 10),
        "table_hit_at_k": None,
    }


def summarize_full_pipeline(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_with_crop_metrics(results, [row["latency"] for row in results])
    summary["latency_note"] = (
        "latency_seconds is end-to-end per question: retrieval + rerank + crop/context + VLM."
    )
    summary["latency_breakdown_seconds"] = {
        "retrieval": latency_summary(results, "retrieval_latency"),
        "rerank": latency_summary(results, "rerank_latency"),
        "context": latency_summary(results, "context_latency"),
        "vlm": latency_summary(results, "vlm_latency"),
    }
    metric_keys = [
        "numeric_exact_match",
        "numeric_relaxed_match",
        "unit_match",
        "entity_match",
        "doc_hit_at_1",
        "doc_hit_at_5",
        "doc_hit_at_10",
        "doc_hit_at_30",
        "reranked_doc_hit_at_1",
        "reranked_doc_hit_at_5",
        "reranked_doc_hit_at_10",
        "page_hit_at_1",
        "page_hit_at_5",
        "page_hit_at_10",
        "table_hit_at_k",
    ]
    summary["additional_metrics"] = {key: optional_mean(results, key) for key in metric_keys}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row.get("type", "unknown"), []).append(row)
    summary["additional_metrics_by_type"] = {
        group_name: {key: optional_mean(group_rows, key) for key in metric_keys}
        for group_name, group_rows in grouped.items()
    }
    route_groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        route = row.get("adaptive_route")
        if route:
            route_groups.setdefault(str(route), []).append(row)
    if route_groups:
        skipped_rows = [row for row in results if row.get("adaptive_skipped_reranker") is True]
        summary["adaptive_reranking"] = {
            "route_counts": {route: len(route_rows) for route, route_rows in route_groups.items()},
            "skipped_reranker_count": len(skipped_rows),
            "skipped_reranker_rate": len(skipped_rows) / max(len(results), 1),
            "latency_by_route": {
                route: latency_summary(route_rows, "latency")
                for route, route_rows in route_groups.items()
            },
            "rerank_latency_by_route": {
                route: latency_summary(route_rows, "rerank_latency")
                for route, route_rows in route_groups.items()
            },
            "quality_by_route": {
                route: {
                    "count": len(route_rows),
                    "mean_f1": optional_mean(route_rows, "f1"),
                    "exact_match": optional_mean(route_rows, "exact"),
                    "f1_gt_0_5": (
                        float(np.mean([float(row.get("f1", 0.0)) > 0.5 for row in route_rows]))
                        if route_rows
                        else 0.0
                    ),
                }
                for route, route_rows in route_groups.items()
            },
        }
    threshold_rows = [row for row in results if row.get("threshold_skip_route_used")]
    if threshold_rows:
        skipped_rows = [row for row in threshold_rows if row.get("threshold_skip_reranker") is True]
        reranked_rows = [
            row for row in threshold_rows if row.get("threshold_skip_reranker") is False
        ]
        scores = [
            float(row["threshold_skip_top1"])
            for row in threshold_rows
            if row.get("threshold_skip_top1") is not None
        ]
        gaps = [
            float(row["threshold_skip_gap"])
            for row in threshold_rows
            if row.get("threshold_skip_gap") is not None
        ]
        summary["threshold_skip_reranking"] = {
            "threshold_top1": threshold_rows[0].get("threshold_skip_threshold_top1"),
            "threshold_gap": threshold_rows[0].get("threshold_skip_threshold_gap"),
            "skip_rate": len(skipped_rows) / max(len(threshold_rows), 1),
            "number_skipped": len(skipped_rows),
            "number_reranked": len(reranked_rows),
            "latency_skipped": latency_summary(skipped_rows, "latency"),
            "latency_reranked": latency_summary(reranked_rows, "latency"),
            "quality_skipped": {
                "mean_f1": optional_mean(skipped_rows, "f1"),
                "exact_match": optional_mean(skipped_rows, "exact"),
            },
            "quality_reranked": {
                "mean_f1": optional_mean(reranked_rows, "f1"),
                "exact_match": optional_mean(reranked_rows, "exact"),
            },
            "retrieval_score_distribution": {
                "min": float(np.min(scores)) if scores else None,
                "max": float(np.max(scores)) if scores else None,
                "mean": float(np.mean(scores)) if scores else None,
                "median": float(np.percentile(scores, 50)) if scores else None,
                "p25": float(np.percentile(scores, 25)) if scores else None,
                "p75": float(np.percentile(scores, 75)) if scores else None,
                "p90": float(np.percentile(scores, 90)) if scores else None,
                "p95": float(np.percentile(scores, 95)) if scores else None,
            },
            "retrieval_gap_distribution": {
                "min": float(np.min(gaps)) if gaps else None,
                "max": float(np.max(gaps)) if gaps else None,
                "mean": float(np.mean(gaps)) if gaps else None,
                "median": float(np.percentile(gaps, 50)) if gaps else None,
                "p25": float(np.percentile(gaps, 25)) if gaps else None,
                "p75": float(np.percentile(gaps, 75)) if gaps else None,
                "p90": float(np.percentile(gaps, 90)) if gaps else None,
                "p95": float(np.percentile(gaps, 95)) if gaps else None,
            },
        }
    summary["additional_metrics_note"] = (
        "page_hit_at_k and table_hit_at_k are null unless the dataset provides oracle page/table annotations."
    )
    return summary


def write_partial(
    args: argparse.Namespace, results: list[dict[str, Any]], debug_rows: list[dict[str, Any]]
) -> None:
    output = {
        "summary": summarize_full_pipeline(results),
        "config": vars(args),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)
    write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)


def main() -> None:
    args = parse_args()
    rows = load_questions(args)

    print(
        json.dumps(
            {
                "selected_questions": len(rows),
                "data_dir": str(args.data_dir),
                "retrieval": (
                    "nemotron_image"
                    if args.retriever_backend == "nemotron_image"
                    else "colvision_multi_vector"
                ),
                "retriever_backend": args.retriever_backend,
                "colpali_used": args.retriever_backend == "colvision",
                "index_name": args.index_name,
                "index_dir": str(args.index_dir),
                "retriever_model_id": args.retriever_model_id,
                "first_stage_top_k": args.first_stage_top_k,
                "reranker_mode": args.reranker_mode,
                "rerank_top_k": args.rerank_top_k,
                "rerank_text_source_fields": (
                    args.rerank_text_source_fields
                    if args.reranker_mode in {"nemotron_text_image", "adaptive"}
                    else []
                ),
                "rerank_text_max_chars": (
                    args.rerank_text_max_chars
                    if args.reranker_mode in {"nemotron_text_image", "adaptive", "threshold_skip"}
                    else 0
                ),
                "adaptive_threshold_top1": args.adaptive_threshold_top1,
                "adaptive_threshold_gap": args.adaptive_threshold_gap,
                "adaptive_high_confidence_strategy": args.adaptive_high_confidence_strategy,
                "threshold_skip_top1": args.threshold_skip_top1,
                "threshold_skip_gap": args.threshold_skip_gap,
                "fusion_mode": args.fusion_mode,
                "fusion_source_fields": args.fusion_source_fields,
                "adaptive_policy": args.adaptive_policy,
                "visual_crop_policy": args.visual_crop_policy,
                "layout_context_mode": args.layout_context_mode,
                "prompt_style": args.prompt_style,
                "multimodal_generation_backend": args.multimodal_generation_backend,
                "openai_vlm_model": (
                    args.openai_vlm_model
                    if args.multimodal_generation_backend == "openai_compatible_vlm"
                    else None
                ),
                "vlm_text_context_mode": args.vlm_text_context_mode,
                "vlm_text_source_fields": (
                    args.vlm_text_source_fields if args.vlm_text_context_mode != "none" else []
                ),
                "vlm_text_max_chars": (
                    args.vlm_text_max_chars if args.vlm_text_context_mode != "none" else 0
                ),
                "dry_run": args.dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    retriever = create_retriever(args)
    reranker = None
    if args.reranker_mode == "nemotron":
        from src.mmrag.rerank import NemotronVLReranker
        from src.mmrag.config import RerankerConfig

        reranker = NemotronVLReranker(
            RerankerConfig(device=args.rerank_device, batch_size=args.rerank_batch_size)
        )
    elif args.reranker_mode == "nemotron_text_image":
        from src.mmrag.rerank import NemotronVLTextImageReranker
        from src.mmrag.config import RerankerConfig

        print(
            "[Reranker Text+Image] Loading text evidence fields="
            f"{args.rerank_text_source_fields} from {args.data_dir}"
        )
        rerank_text_evidence_map = load_text_evidence_map(
            args.data_dir,
            source_fields=args.rerank_text_source_fields,
        )
        print(
            "[Reranker Text+Image] Loaded text evidence pages: "
            f"{len(rerank_text_evidence_map)} max_chars={args.rerank_text_max_chars}"
        )
        reranker = NemotronVLTextImageReranker(
            RerankerConfig(device=args.rerank_device, batch_size=args.rerank_batch_size),
            evidence_map=rerank_text_evidence_map,
            max_text_chars=args.rerank_text_max_chars,
        )
    elif args.reranker_mode == "adaptive":
        from src.mmrag.config import RerankerConfig
        from src.mmrag.rerank import NemotronVLReranker, NemotronVLTextImageReranker
        from src.reranking.adaptive_reranker import AdaptiveReranker, LazyReranker
        from src.reranking.no_reranker import NoReranker

        print(
            "[Adaptive Reranker] Loading text evidence fields="
            f"{args.rerank_text_source_fields} from {args.data_dir}"
        )
        rerank_text_evidence_map = load_text_evidence_map(
            args.data_dir,
            source_fields=args.rerank_text_source_fields,
        )
        print(
            "[Adaptive Reranker] Loaded text evidence pages: "
            f"{len(rerank_text_evidence_map)} max_chars={args.rerank_text_max_chars}"
        )
        reranker_config = RerankerConfig(
            device=args.rerank_device,
            batch_size=args.rerank_batch_size,
        )
        image_reranker = LazyReranker(
            lambda: NemotronVLReranker(reranker_config),
            "image-only Nemotron VL reranker",
        )
        text_image_reranker = LazyReranker(
            lambda: NemotronVLTextImageReranker(
                reranker_config,
                evidence_map=rerank_text_evidence_map,
                max_text_chars=args.rerank_text_max_chars,
            ),
            "text+image Nemotron VL reranker",
        )
        reranker = AdaptiveReranker(
            image_reranker=image_reranker,
            text_image_reranker=text_image_reranker,
            no_reranker=NoReranker(),
            threshold_top1=args.adaptive_threshold_top1,
            threshold_gap=args.adaptive_threshold_gap,
            high_confidence_strategy=args.adaptive_high_confidence_strategy,
        )
    elif args.reranker_mode == "threshold_skip":
        from src.mmrag.config import RerankerConfig
        from src.mmrag.rerank import NemotronVLTextImageReranker
        from src.reranking.threshold_skip_reranker import ThresholdSkipReranker

        print(
            "[Threshold Skip Reranker] Loading text evidence fields="
            f"{args.rerank_text_source_fields} from {args.data_dir}"
        )
        rerank_text_evidence_map = load_text_evidence_map(
            args.data_dir,
            source_fields=args.rerank_text_source_fields,
        )
        print(
            "[Threshold Skip Reranker] Loaded text evidence pages: "
            f"{len(rerank_text_evidence_map)} max_chars={args.rerank_text_max_chars}"
        )
        fallback_reranker = NemotronVLTextImageReranker(
            RerankerConfig(device=args.rerank_device, batch_size=args.rerank_batch_size),
            evidence_map=rerank_text_evidence_map,
            max_text_chars=args.rerank_text_max_chars,
        )
        reranker = ThresholdSkipReranker(
            fallback_reranker=fallback_reranker,
            threshold_top1=args.threshold_skip_top1,
            threshold_gap=args.threshold_skip_gap,
        )
    fusion_text_reranker = None
    fusion_evidence_map: dict[tuple[str, int], dict[str, Any]] = {}
    vlm_text_evidence_map: dict[tuple[str, int], dict[str, Any]] = {}
    fusion_weights = FusionWeights(
        alpha=args.fusion_alpha,
        beta=args.fusion_beta,
        gamma=args.fusion_gamma,
        lambda_number=args.fusion_lambda_number,
        lambda_keyword=args.fusion_lambda_keyword,
        lambda_exact_phrase=args.fusion_lambda_exact_phrase,
        lambda_table_header=args.fusion_lambda_table_header,
    )
    if args.fusion_mode == "score_fusion":
        print(
            "[Fusion] Loading text evidence fields="
            f"{args.fusion_source_fields} from {args.data_dir}"
        )
        fusion_evidence_map = load_text_evidence_map(
            args.data_dir,
            source_fields=args.fusion_source_fields,
        )
        print(f"[Fusion] Loaded text evidence pages: {len(fusion_evidence_map)}")
        if args.fusion_text_reranker_backend != "none":
            fusion_text_reranker = create_text_reranker(
                args.fusion_text_reranker_model_id,
                device=args.fusion_text_reranker_device,
                batch_size=args.fusion_text_reranker_batch_size,
                max_length=args.fusion_text_reranker_max_length,
                trust_remote_code=not args.fusion_no_trust_remote_code,
                backend=args.fusion_text_reranker_backend,
            )
    if args.vlm_text_context_mode != "none":
        print(
            "[VLM Text Context] Loading text evidence fields="
            f"{args.vlm_text_source_fields} from {args.data_dir}"
        )
        vlm_text_evidence_map = load_text_evidence_map(
            args.data_dir,
            source_fields=args.vlm_text_source_fields,
        )
        print(f"[VLM Text Context] Loaded text evidence pages: {len(vlm_text_evidence_map)}")
    generator = create_multimodal_generator(args)

    results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for display_idx, row in enumerate(rows, start=1 + args.start):
        expected = row.get("answer") or ""
        question = row["question"]
        total_start = time.time()

        retrieval_start = time.time()
        retrieved = retriever.search(question, top_k=args.first_stage_top_k)
        retrieval_latency = time.time() - retrieval_start

        rerank_start = time.time()
        image_rerank_latency = 0.0
        text_rerank_latency = 0.0
        fusion_latency = 0.0
        adaptive_route_info = None
        threshold_skip_decision = None
        fusion_enabled = args.fusion_mode == "score_fusion"
        if fusion_enabled:
            attach_text_evidence(retrieved, fusion_evidence_map)
            image_rerank_start = time.time()
            if reranker is None:
                image_scored = retrieved[:]
            elif args.reranker_mode == "adaptive":
                image_scored = reranker.rerank(question, retrieved, metadata=row)
                adaptive_route_info = getattr(reranker, "last_route_info", None)
            elif args.reranker_mode == "threshold_skip":
                image_scored = reranker.rerank(question, retrieved, metadata=row)
                threshold_skip_decision = getattr(reranker, "last_decision", None)
            else:
                image_scored = reranker.rerank(question, retrieved)
            image_rerank_latency = time.time() - image_rerank_start

            text_rerank_start = time.time()
            if fusion_text_reranker is None:
                text_scored = image_scored
            else:
                text_output = fusion_text_reranker.rerank(question, image_scored)
                text_scored = text_output.candidates
            text_rerank_latency = time.time() - text_rerank_start

            fusion_start = time.time()
            reranked = fuse_candidates(question, text_scored, weights=fusion_weights)[
                : args.rerank_top_k
            ]
            fusion_latency = time.time() - fusion_start
        elif reranker is None:
            reranked = retrieved[: args.rerank_top_k]
        elif args.reranker_mode == "adaptive":
            reranked = reranker.rerank(question, retrieved, metadata=row)[: args.rerank_top_k]
            adaptive_route_info = getattr(reranker, "last_route_info", None)
        elif args.reranker_mode == "threshold_skip":
            reranked = reranker.rerank(question, retrieved, metadata=row)[: args.rerank_top_k]
            threshold_skip_decision = getattr(reranker, "last_decision", None)
        else:
            reranked = reranker.rerank(question, retrieved)[: args.rerank_top_k]
        rerank_latency = time.time() - rerank_start

        effective_top_pages, effective_crop_policy, visual_context = (
            effective_context_settings_layout(args, row)
        )
        context_candidates = expand_with_neighbors_local(
            reranked[:effective_top_pages],
            args.neighbor_radius,
            final_limit=effective_top_pages,
        )
        context_candidate_rows = [
            candidate_to_json(candidate, rank=rank)
            for rank, candidate in enumerate(context_candidates, start=1)
        ]

        context_start = time.time()
        images, layout_debug, selected_crop = load_context_images(
            candidates=context_candidate_rows,
            crop_policy=effective_crop_policy,
            crop_top_n=args.crop_top_n,
            question=question,
            row=row,
            debug_crop_dir=args.debug_crop_dir,
            question_index=display_idx,
            layout_context_mode=args.layout_context_mode,
        )
        context_latency = time.time() - context_start
        image_paths_sent = ordered_context_image_paths(
            context_candidate_rows, selected_crop, args.layout_context_mode
        )
        vlm_text_context = ""
        vlm_text_truncated = False
        vlm_text_debug: list[dict[str, Any]] = []
        if args.vlm_text_context_mode == "selected_pages":
            vlm_text_context, vlm_text_truncated, vlm_text_debug = build_vlm_text_context(
                context_candidates,
                vlm_text_evidence_map,
                max_chars=args.vlm_text_max_chars,
            )

        print(f"\n[{display_idx}/{args.start + len(rows)}] {question[:100]}")
        print(
            "    retrieved_top5="
            f"{[f'{candidate.folder}/{candidate.page}' for candidate in retrieved[:5]]}"
        )
        print(
            "    reranked_top5="
            f"{[f'{candidate.folder}/{candidate.page}' for candidate in reranked[:5]]}"
        )
        if args.reranker_mode == "nemotron_text_image":
            print(
                "    rerank_text_chars_top5="
                f"{[int(getattr(candidate, 'rerank_text_chars', 0) or 0) for candidate in reranked[:5]]}"
            )
        if adaptive_route_info is not None:
            print(
                "    adaptive_route="
                f"{adaptive_route_info.route} strategy={adaptive_route_info.strategy} "
                f"skipped={adaptive_route_info.skipped_reranker} "
                f"top1={adaptive_route_info.retrieval_top1_score} "
                f"gap={adaptive_route_info.retrieval_top1_gap}"
            )
        if threshold_skip_decision is not None:
            stats = threshold_skip_decision.confidence_stats
            print(
                "    threshold_skip="
                f"route={threshold_skip_decision.route_used} "
                f"skip={threshold_skip_decision.skip_reranker} "
                f"top1={stats.get('top1_score')} top2={stats.get('top2_score')} "
                f"gap={stats.get('gap')} reason={threshold_skip_decision.reason}"
            )
        if fusion_enabled:
            print(
                "    fusion_top5="
                f"{[f'{candidate.folder}/{candidate.page}:{getattr(candidate, 'fusion_score', 0.0):.3f}' for candidate in reranked[:5]]}"
            )
            print(
                f"    fusion_latency image={image_rerank_latency:.2f}s "
                f"text={text_rerank_latency:.2f}s fusion={fusion_latency:.2f}s"
            )
        print(
            f"    context=top{effective_top_pages} crop={effective_crop_policy} "
            f"mode={args.layout_context_mode} visual={visual_context}"
        )
        if args.vlm_text_context_mode != "none":
            print(
                "    vlm_text_context="
                f"mode={args.vlm_text_context_mode} chars={len(vlm_text_context)} "
                f"truncated={vlm_text_truncated}"
            )
        if selected_crop:
            crop_score = selected_crop.get("crop_score")
            crop_score_text = f"{crop_score:.3f}" if isinstance(crop_score, int | float) else "None"
            print(
                "    layout_crop="
                f"{selected_crop.get('crop_type')} {selected_crop.get('page_label')} "
                f"score={crop_score_text} path={selected_crop.get('crop_path')} "
                f"fallback={selected_crop.get('fallback_used')}"
            )

        vlm_start = time.time()
        try:
            if getattr(generator, "generation_backend", None) == "openai_compatible_vlm":
                answer = generator.generate_answer(
                    question,
                    images,
                    context_text=vlm_text_context or None,
                    image_paths=image_paths_sent,
                )
            else:
                answer = generator.generate_answer(question, images)
        except Exception as exc:
            print(f"    [ERROR] VLM failed: {exc}")
            generator.last_raw_output = ""
            generator.last_reasoning = ""
            generator.last_answer = "ERROR"
            answer = "ERROR"
        vlm_latency = time.time() - vlm_start
        latency = time.time() - total_start

        exact, f1 = compute_similarity(answer, expected)
        extended_metrics = compute_extended_metrics(answer, expected)
        oracle_matches = find_answer_pages(
            args.data_dir / str(row.get("expected_folder")),
            expected,
            row.get("evidence", ""),
        )
        oracle_pages = [
            {
                "page": match.page,
                "exact_answer": match.exact_answer,
                "number_recall": match.number_recall,
                "keyword_recall": match.keyword_recall,
                "matched_numbers": match.matched_numbers,
                "matched_keywords": match.matched_keywords,
            }
            for match in oracle_matches[:10]
        ]
        extra_metrics = extra_row_metrics(
            generated=answer,
            expected=expected,
            retrieved=retrieved,
            reranked=reranked,
            expected_folder=row.get("expected_folder"),
            oracle_pages=oracle_pages,
        )
        result = {
            "question_id": row.get("question_id"),
            "question": question,
            "expected": expected,
            "generated": answer,
            "raw_generated": getattr(generator, "last_raw_output", answer),
            "raw_generated_answer": getattr(generator, "last_raw_output", answer),
            "postprocessed_answer": answer,
            "vlm_think": getattr(generator, "last_reasoning", ""),
            "exact": exact,
            "f1": f1,
            "latency": latency,
            "retrieval_latency": retrieval_latency,
            "rerank_latency": rerank_latency,
            "image_rerank_latency": image_rerank_latency,
            "text_rerank_latency": text_rerank_latency,
            "fusion_latency": fusion_latency,
            "context_latency": context_latency,
            "vlm_latency": vlm_latency,
            "latency_generation": getattr(generator, "last_latency_generation", vlm_latency),
            "generation_backend": getattr(generator, "generation_backend", "local_vlm"),
            "model_name": (
                getattr(generator, "model", None)
                if getattr(generator, "generation_backend", None) == "openai_compatible_vlm"
                else "Qwen/Qwen3-VL-8B-Instruct"
            ),
            "image_paths_sent": getattr(generator, "last_image_paths_sent", image_paths_sent),
            "num_images_sent": getattr(generator, "last_num_images_sent", len(images)),
            "vlm_text_context_mode": args.vlm_text_context_mode,
            "vlm_text_source_fields": (
                args.vlm_text_source_fields if args.vlm_text_context_mode != "none" else []
            ),
            "vlm_text_context_chars": len(vlm_text_context),
            "vlm_text_context_truncated": vlm_text_truncated,
            "vlm_text_context_debug": vlm_text_debug,
            "generation_error": getattr(generator, "last_error", None),
            "type": row.get("type", ""),
            "expected_folder": row.get("expected_folder"),
            "evidence": row.get("evidence", ""),
            "oracle_pages": oracle_pages,
            "effective_top_pages": effective_top_pages,
            "effective_crop_policy": effective_crop_policy,
            "visual_context": visual_context,
            "reranker_mode": args.reranker_mode,
            "rerank_text_source_fields": (
                args.rerank_text_source_fields
                if args.reranker_mode in {"nemotron_text_image", "adaptive", "threshold_skip"}
                else []
            ),
            "rerank_text_max_chars": (
                args.rerank_text_max_chars
                if args.reranker_mode in {"nemotron_text_image", "adaptive", "threshold_skip"}
                else 0
            ),
            "rerank_text_chars": (
                [int(getattr(candidate, "rerank_text_chars", 0) or 0) for candidate in reranked]
                if args.reranker_mode in {"nemotron_text_image", "adaptive", "threshold_skip"}
                else []
            ),
            "adaptive_route": (
                adaptive_route_info.route if adaptive_route_info is not None else None
            ),
            "adaptive_strategy": (
                adaptive_route_info.strategy if adaptive_route_info is not None else None
            ),
            "adaptive_skipped_reranker": (
                adaptive_route_info.skipped_reranker if adaptive_route_info is not None else None
            ),
            "adaptive_route_reason": (
                adaptive_route_info.reason if adaptive_route_info is not None else None
            ),
            "adaptive_route_latency": (
                adaptive_route_info.latency if adaptive_route_info is not None else None
            ),
            "adaptive_retrieval_top1_score": (
                adaptive_route_info.retrieval_top1_score
                if adaptive_route_info is not None
                else None
            ),
            "adaptive_retrieval_top1_gap": (
                adaptive_route_info.retrieval_top1_gap if adaptive_route_info is not None else None
            ),
            "threshold_skip_top1": (
                threshold_skip_decision.confidence_stats.get("top1_score")
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_top2": (
                threshold_skip_decision.confidence_stats.get("top2_score")
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_gap": (
                threshold_skip_decision.confidence_stats.get("gap")
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_relative_gap": (
                threshold_skip_decision.confidence_stats.get("relative_gap")
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_candidates_count": (
                threshold_skip_decision.confidence_stats.get("candidates_count")
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_reranker": (
                threshold_skip_decision.skip_reranker
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_route_used": (
                threshold_skip_decision.route_used if threshold_skip_decision is not None else None
            ),
            "threshold_skip_reason": (
                threshold_skip_decision.reason if threshold_skip_decision is not None else None
            ),
            "threshold_skip_threshold_top1": (
                threshold_skip_decision.threshold_top1
                if threshold_skip_decision is not None
                else None
            ),
            "threshold_skip_threshold_gap": (
                threshold_skip_decision.threshold_gap
                if threshold_skip_decision is not None
                else None
            ),
            "fusion_mode": args.fusion_mode,
            "fusion_weights": {
                "alpha": args.fusion_alpha,
                "beta": args.fusion_beta,
                "gamma": args.fusion_gamma,
                "lambda_number": args.fusion_lambda_number,
                "lambda_keyword": args.fusion_lambda_keyword,
                "lambda_exact_phrase": args.fusion_lambda_exact_phrase,
                "lambda_table_header": args.fusion_lambda_table_header,
            },
            "fusion_source_fields": args.fusion_source_fields,
            "fusion_text_reranker_model_id": (
                args.fusion_text_reranker_model_id if fusion_enabled else None
            ),
            "retriever_backend": args.retriever_backend,
            "colpali_used": args.retriever_backend == "colvision",
            "retrieval_scores": [float(candidate.score) for candidate in retrieved],
            "top30_retrieved_pages": [
                f"{candidate.folder}/{candidate.page}" for candidate in retrieved[:30]
            ],
            "retrieved_candidates": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(retrieved, start=1)
            ],
            "reranked_candidates": [
                candidate_to_json(candidate, rank=rank)
                for rank, candidate in enumerate(reranked, start=1)
            ],
            "pages": context_candidate_rows,
            "layout_aware_debug": layout_debug,
            "layout_aware_selected_crop": selected_crop,
            "question_crop_intent": (selected_crop or {}).get("question_crop_intent"),
            "explicit_reference": (selected_crop or {}).get("explicit_reference"),
            "selected_crop_type": (selected_crop or {}).get("selected_crop_type")
            or (selected_crop or {}).get("crop_type"),
            "selected_crop_caption": (selected_crop or {}).get("selected_crop_caption"),
            "selected_crop_score": (selected_crop or {}).get("selected_crop_score")
            or (selected_crop or {}).get("crop_score"),
            "crop_type_mismatch": (selected_crop or {}).get("crop_type_mismatch", False),
            "caption_match": (selected_crop or {}).get("caption_match", False),
            "fallback_used": (selected_crop or {}).get("fallback_used", selected_crop is None),
            "crop_used": (selected_crop or {}).get("crop_used", False),
            "full_page_plus_crop": (selected_crop or {}).get("full_page_plus_crop", False),
            "crop_path": (selected_crop or {}).get("crop_path"),
            **extended_metrics,
            **extra_metrics,
        }
        results.append(result)
        debug_rows.append(compact_case(result))

        if args.print_think:
            think = (result["vlm_think"] or result["raw_generated"] or "").replace("\n", " ")
            print(f"    think={think[:500]}")
        print(f"    generated={answer[:240]}")
        print(f"    expected={expected}")
        print(
            f"    exact={exact} f1={f1:.3f} "
            f"latency={latency:.2f}s "
            f"(retrieval={retrieval_latency:.2f}s rerank={rerank_latency:.2f}s "
            f"context={context_latency:.2f}s vlm={vlm_latency:.2f}s)"
        )

        write_partial(args, results, debug_rows)

    output = {
        "summary": summarize_full_pipeline(results),
        "config": vars(args),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)
    write_case_csv(args.output.with_name(args.output.stem + "_layout_debug.csv"), debug_rows)

    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Debug crops: {args.debug_crop_dir}")


if __name__ == "__main__":
    main()
