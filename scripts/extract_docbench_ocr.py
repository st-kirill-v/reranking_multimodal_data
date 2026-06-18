#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PAGE_IMAGE_RE = re.compile(r"^page_(?P<page>\d+)\.(?:png|jpg|jpeg|webp)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract page-level OCR sidecar files for DocBench page images."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument(
        "--backend",
        choices=["easyocr", "tesseract"],
        default="easyocr",
        help="OCR backend. easyocr can use CUDA; tesseract is CPU fallback via pytesseract.",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code. Use 'eng' for DocBench unless extra language packs are installed.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode. 6 works well for dense page text.",
    )
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for easyocr. Ignored by tesseract.",
    )
    parser.add_argument(
        "--easyocr-batch-size",
        type=int,
        default=16,
        help="Recognition batch size for easyocr.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root for sidecar output. By default writes to each "
            "doc_id/extracted directory. With this option writes to "
            "output_root/doc_id/extracted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute OCR even when pages_ocr.json already exists.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable simple grayscale/autocontrast preprocessing before OCR.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def numeric_doc_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def page_images(pages_dir: Path) -> list[tuple[int, Path]]:
    if not pages_dir.exists():
        return []
    items: list[tuple[int, Path]] = []
    for path in pages_dir.iterdir():
        if not path.is_file():
            continue
        match = PAGE_IMAGE_RE.match(path.name)
        if not match:
            continue
        items.append((int(match.group("page")), path))
    return sorted(items, key=lambda item: item[0])


def load_tesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python package 'pytesseract'. Install it with: uv add pytesseract"
        ) from exc
    return pytesseract


def load_easyocr(*, lang: str, gpu: bool):
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python package 'easyocr'. Install it with: uv add easyocr"
        ) from exc
    languages = [item.strip() for item in lang.split("+") if item.strip()]
    if not languages:
        languages = ["en"]
    languages = ["en" if item == "eng" else item for item in languages]
    return easyocr.Reader(languages, gpu=gpu, verbose=False)


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    return ImageOps.autocontrast(image)


def normalize_ocr_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def ocr_image(
    image_path: Path,
    *,
    backend: str,
    ocr_engine: Any,
    lang: str,
    psm: int,
    preprocess: bool,
    easyocr_batch_size: int,
) -> str:
    with Image.open(image_path) as image:
        if backend == "tesseract" and preprocess:
            image = preprocess_image(image)
        if backend == "tesseract":
            config = f"--psm {psm}"
            text = ocr_engine.image_to_string(image, lang=lang, config=config)
        elif backend == "easyocr":
            rgb = image.convert("RGB")
            result = ocr_engine.readtext(
                np.asarray(rgb),
                detail=0,
                paragraph=True,
                batch_size=easyocr_batch_size,
            )
            text = "\n".join(str(item) for item in result)
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")
    return normalize_ocr_text(text)


def write_json(path: Path, payload: Any, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    folders = numeric_doc_folders(args.dataset_root)
    if args.limit_docs > 0:
        folders = folders[: args.limit_docs]

    ocr_engine = None
    if not args.dry_run:
        if args.backend == "tesseract":
            ocr_engine = load_tesseract()
        elif args.backend == "easyocr":
            ocr_engine = load_easyocr(lang=args.lang, gpu=args.gpu)
        else:
            raise ValueError(f"Unsupported OCR backend: {args.backend}")
    started = time.time()
    total_pages_seen = 0
    total_pages_ocr = 0
    total_nonempty_pages = 0
    total_chars = 0
    skipped_existing = 0
    warnings: list[str] = []

    print(f"[OCR] Dataset root: {args.dataset_root}")
    print(f"[OCR] Document folders: {len(folders)}")
    print(
        f"[OCR] backend={args.backend} lang={args.lang} psm={args.psm} gpu={args.gpu} "
        f"preprocess={not args.no_preprocess} dry_run={args.dry_run}"
    )

    for doc_idx, doc_folder in enumerate(folders, start=1):
        extracted_dir = doc_folder / "extracted"
        pages_dir = extracted_dir / "pages"
        output_extracted_dir = (
            args.output_root / doc_folder.name / "extracted"
            if args.output_root is not None
            else extracted_dir
        )
        output_path = output_extracted_dir / "pages_ocr.json"
        report_path = output_extracted_dir / "ocr_report.json"

        images = page_images(pages_dir)
        if args.limit_pages > 0:
            images = images[: args.limit_pages]
        total_pages_seen += len(images)

        if not images:
            warnings.append(f"Missing page images for doc {doc_folder.name}: {pages_dir}")
            continue
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(
                f"[OCR] [{doc_idx}/{len(folders)}] doc={doc_folder.name} "
                f"skip existing={output_path}"
            )
            continue

        rows: list[dict[str, Any]] = []
        doc_chars = 0
        doc_nonempty = 0
        doc_started = time.time()
        for page, image_path in images:
            if args.dry_run:
                text = ""
            else:
                try:
                    text = ocr_image(
                        image_path,
                        backend=args.backend,
                        ocr_engine=ocr_engine,
                        lang=args.lang,
                        psm=args.psm,
                        preprocess=not args.no_preprocess,
                        easyocr_batch_size=args.easyocr_batch_size,
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"OCR failed doc={doc_folder.name} page={page}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    text = ""
            if text:
                doc_nonempty += 1
                doc_chars += len(text)
            rows.append(
                {
                    "page": page,
                    "ocr_text": text,
                    "image_path": image_path.as_posix(),
                }
            )

        elapsed = time.time() - doc_started
        total_pages_ocr += len(rows)
        total_nonempty_pages += doc_nonempty
        total_chars += doc_chars
        write_json(output_path, rows, dry_run=args.dry_run)
        write_json(
            report_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend": args.backend,
                "lang": args.lang,
                "psm": args.psm,
                "gpu": args.gpu,
                "easyocr_batch_size": args.easyocr_batch_size,
                "preprocess": not args.no_preprocess,
                "pages_dir": pages_dir.as_posix(),
                "ocr_file": output_path.as_posix(),
                "pages_seen": len(images),
                "pages_ocr": len(rows),
                "pages_nonempty": doc_nonempty,
                "chars": doc_chars,
                "elapsed_seconds": elapsed,
            },
            dry_run=args.dry_run,
        )
        print(
            f"[OCR] [{doc_idx}/{len(folders)}] doc={doc_folder.name} "
            f"pages={len(rows)} nonempty={doc_nonempty} chars={doc_chars} "
            f"time={elapsed:.1f}s"
        )

    summary = {
        "dataset_root": args.dataset_root.as_posix(),
        "documents_processed": len(folders),
        "pages_seen": total_pages_seen,
        "pages_ocr": total_pages_ocr,
        "pages_nonempty": total_nonempty_pages,
        "chars": total_chars,
        "skipped_existing_docs": skipped_existing,
        "elapsed_seconds": time.time() - started,
        "warnings": warnings,
        "dry_run": args.dry_run,
    }
    print("[OCR] Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
