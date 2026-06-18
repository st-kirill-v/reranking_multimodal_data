import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


def find_pdf_in_folder(folder_path: Path) -> Path:
    for file in folder_path.iterdir():
        if file.suffix.lower() == ".pdf":
            return file
    return None


def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"  Processing: {pdf_path.name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.do_code_enrichment = False
    pipeline_options.images_scale = 1.0

    accelerator_options = AcceleratorOptions(device="cuda", num_threads=4)

    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
            accelerator_options=accelerator_options,
        )
    }

    converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options=format_options)

    start_time = time.time()
    result = converter.convert(str(pdf_path))
    elapsed = time.time() - start_time

    doc = result.document
    markdown = doc.export_to_markdown()

    output_file = output_dir / "content.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"    Time: {elapsed:.2f}s, Markdown size: {len(markdown)} chars")

    return len(markdown)


if __name__ == "__main__":
    base_dir = Path("/home/user-13/reranking_multimodal_data/data/datasets/docbench")
    output_base = Path("/home/user-13/reranking_multimodal_data/data/docling_markdown_fast")
    output_base.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_chars = 0

    for folder_num in range(0, 229):
        folder = base_dir / str(folder_num)
        if not folder.exists():
            continue

        pdf_file = find_pdf_in_folder(folder)
        if pdf_file is None:
            print(f"Folder {folder_num}: No PDF found")
            continue

        output_dir = output_base / str(folder_num)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFolder {folder_num}:")
        chars = process_pdf(pdf_file, output_dir)

        processed += 1
        total_chars += chars

        if processed % 20 == 0:
            print(f"\n--- Progress: {processed} folders processed, {total_chars} total chars ---")

    print(f"\n=== COMPLETED ===")
    print(f"Processed: {processed} folders")
    print(f"Total characters: {total_chars}")
    print(f"Output directory: {output_base}")
