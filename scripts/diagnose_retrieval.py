from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Qwen page-retrieval behavior.")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--search-top-k", type=int, default=0)
    parser.add_argument("--aggregate-page", action="store_true")
    parser.add_argument("--allow-missing-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.dataset import load_document_domains
    from src.mmrag.diagnostics import summarize_domains
    from src.mmrag.indexing import FaissPageIndex

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
    print(json.dumps(page_index.manifest or {"warning": "missing manifest"}, indent=2))
    print(page_index.vector_stats())

    if not args.query:
        return

    from src.mmrag.retrieval import MultimodalPageRetriever, aggregate_candidates_by_page

    retriever = MultimodalPageRetriever(
        config,
        strict_manifest=not args.allow_missing_manifest,
    )
    search_top_k = args.search_top_k or args.top_k
    candidates = retriever.search(args.query, top_k=search_top_k)
    if args.aggregate_page:
        candidates = aggregate_candidates_by_page(candidates)
    candidates = candidates[: args.top_k]
    domains = load_document_domains(config.paths.data_dir)
    print("domain_summary:", summarize_domains(candidates, domains))
    for candidate in candidates:
        print(
            f"{candidate.rank:02d} score={candidate.score:.5f} "
            f"doc={candidate.folder} page={candidate.page} tile={candidate.tile_id} "
            f"crop={candidate.crop_box} path={candidate.path}"
        )


if __name__ == "__main__":
    main()
