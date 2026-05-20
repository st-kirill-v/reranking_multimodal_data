from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index over page image tiles.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3_tiles_2x2")
    parser.add_argument(
        "--tile-mode", choices=["2x2", "vertical_3", "horizontal_2", "full"], default="2x2"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.indexing import build_tile_index

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    manifest = build_tile_index(config, tile_mode=args.tile_mode)
    print(
        "Built tile index: "
        f"{manifest['index']['vectors']} vectors, dim={manifest['index']['dim']}, "
        f"mode={manifest['tiling']['mode']}, model={manifest['embedding']['model_id']}"
    )


if __name__ == "__main__":
    main()
