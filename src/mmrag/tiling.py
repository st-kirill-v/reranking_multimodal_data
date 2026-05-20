from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.mmrag.schema import PageRecord


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    crop_box: tuple[int, int, int, int]


def make_tile_specs(width: int, height: int, mode: str) -> list[TileSpec]:
    if mode == "2x2":
        x_mid = width // 2
        y_mid = height // 2
        return [
            TileSpec("r0_c0", (0, 0, x_mid, y_mid)),
            TileSpec("r0_c1", (x_mid, 0, width, y_mid)),
            TileSpec("r1_c0", (0, y_mid, x_mid, height)),
            TileSpec("r1_c1", (x_mid, y_mid, width, height)),
        ]
    if mode == "vertical_3":
        y1 = height // 3
        y2 = (height * 2) // 3
        return [
            TileSpec("v0", (0, 0, width, y1)),
            TileSpec("v1", (0, y1, width, y2)),
            TileSpec("v2", (0, y2, width, height)),
        ]
    if mode == "horizontal_2":
        x_mid = width // 2
        return [
            TileSpec("h0", (0, 0, x_mid, height)),
            TileSpec("h1", (x_mid, 0, width, height)),
        ]
    if mode == "full":
        return [TileSpec("full", (0, 0, width, height))]
    raise ValueError(f"Unknown tile mode: {mode}")


def iter_page_tiles(
    records: Iterable[PageRecord], mode: str
) -> Iterable[tuple[PageRecord, Image.Image]]:
    tile_index = 0
    for record in records:
        with Image.open(record.path) as img:
            page_image = img.convert("RGB")
            width, height = page_image.size
            for spec in make_tile_specs(width, height, mode):
                tile = page_image.crop(spec.crop_box)
                tile_record = PageRecord(
                    folder=record.folder,
                    page=record.page,
                    path=Path(record.path),
                    index=tile_index,
                    tile_id=spec.tile_id,
                    crop_box=spec.crop_box,
                    source_page_width=width,
                    source_page_height=height,
                )
                tile_index += 1
                yield tile_record, tile
