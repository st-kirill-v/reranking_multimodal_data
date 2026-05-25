from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit visual quality/statistics of clean Qwen3 tile metadata."
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="tiles_qwen3_docapi_grid2x3")
    parser.add_argument("--hub-json", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit_qwen3_tile_quality_clean.json")
    )
    return parser.parse_args()


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('tiles_')}.json"


def tile_key(record: dict) -> str:
    return f"{record['folder']}_{record['page']}_{record['tile_id']}"


def load_tile(record: dict) -> Image.Image:
    with Image.open(record["path"]) as img:
        image = img.convert("RGB")
        return image.crop(tuple(record["crop_box"])).copy()


def image_stats(image: Image.Image) -> dict:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    nonzero = prob[prob > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))

    # Conservative document-content proxies. They do not read text; they only
    # measure whether the crop has visible dark/structured pixels.
    ink_ratio = float(np.mean(gray < 245))
    dark_ratio = float(np.mean(gray < 180))
    very_dark_ratio = float(np.mean(gray < 80))
    white_ratio = float(np.mean(gray > 250))
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))

    gy, gx = np.gradient(gray.astype(np.float32))
    edge_density = float(np.mean(np.sqrt(gx * gx + gy * gy) > 20.0))

    return {
        "width": int(image.width),
        "height": int(image.height),
        "brightness": brightness,
        "contrast": contrast,
        "entropy": entropy,
        "ink_ratio": ink_ratio,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "white_ratio": white_ratio,
        "edge_density": edge_density,
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p05": None,
            "p25": None,
            "mean": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def blankish(stats: dict) -> bool:
    return (
        stats["ink_ratio"] < 0.01
        or stats["contrast"] < 5.0
        or stats["entropy"] < 0.45
        or stats["edge_density"] < 0.002
    )


def load_hub_keys(path: Path | None) -> Counter[str]:
    if path is None or not path.exists():
        return Counter()
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    counter: Counter[str] = Counter()

    def add_pairs(pairs) -> None:
        for key, count in pairs or []:
            counter[str(key)] += int(count)

    summary = data.get("summary", data)
    for block_name in ["global_hubs", "hubness"]:
        block = summary.get(block_name, {})
        add_pairs(block.get("tile_top10"))
        add_pairs(block.get("top1_tile_top10"))
    for variant in ["raw", "dehub"]:
        block = summary.get(variant, {}).get("hubness", {})
        add_pairs(block.get("top1_tile_top10"))
    return counter


def main() -> None:
    args = parse_args()
    metadata_path = args.index_dir / metadata_filename(args.index_name)
    with metadata_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    rng = random.Random(args.seed)
    sample_records = records
    if args.sample > 0 and args.sample < len(records):
        sample_records = rng.sample(records, args.sample)

    hub_counts = load_hub_keys(args.hub_json)
    by_key = {tile_key(record): record for record in records}

    rows = []
    stats_by_metric: dict[str, list[float]] = {
        "brightness": [],
        "contrast": [],
        "entropy": [],
        "ink_ratio": [],
        "dark_ratio": [],
        "very_dark_ratio": [],
        "white_ratio": [],
        "edge_density": [],
    }
    position_counts: Counter[str] = Counter()
    blankish_count = 0

    for i, record in enumerate(sample_records, start=1):
        stats = image_stats(load_tile(record))
        is_blankish = blankish(stats)
        blankish_count += 1 if is_blankish else 0
        position_counts[str(record["tile_id"])] += 1
        for metric in stats_by_metric:
            stats_by_metric[metric].append(float(stats[metric]))
        if i <= 30:
            rows.append(
                {**record, "tile_key": tile_key(record), "stats": stats, "blankish": is_blankish}
            )
        if i % 500 == 0:
            print(f"[sample {i}/{len(sample_records)}]")

    hub_rows = []
    for key, count in hub_counts.most_common(50):
        record = by_key.get(key)
        if record is None:
            continue
        stats = image_stats(load_tile(record))
        hub_rows.append(
            {
                **record,
                "tile_key": key,
                "hub_count": count,
                "stats": stats,
                "blankish": blankish(stats),
            }
        )

    metric_summary = {metric: quantiles(values) for metric, values in stats_by_metric.items()}
    summary = {
        "index_name": args.index_name,
        "tiles_total": len(records),
        "sampled_tiles": len(sample_records),
        "blankish_rate_in_sample": blankish_count / max(len(sample_records), 1),
        "tile_position_counts_in_sample": position_counts.most_common(),
        "metrics": metric_summary,
        "hub_json": str(args.hub_json) if args.hub_json else None,
        "hub_tiles_loaded": len(hub_rows),
        "hub_blankish_rate": sum(1 for row in hub_rows if row["blankish"]) / max(len(hub_rows), 1),
        "suggested_filter": {
            "min_ink_ratio": 0.01,
            "min_contrast": 5.0,
            "min_entropy": 0.45,
            "min_edge_density": 0.002,
        },
    }
    output = {
        "summary": summary,
        "hub_tiles": hub_rows,
        "sample_examples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Blankish rate in sample: {summary['blankish_rate_in_sample']:.3f}")
    print(f"Hub blankish rate: {summary['hub_blankish_rate']:.3f}")
    print(f"Suggested filter: {summary['suggested_filter']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
