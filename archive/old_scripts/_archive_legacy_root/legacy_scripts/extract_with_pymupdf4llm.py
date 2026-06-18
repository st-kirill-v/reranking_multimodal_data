import sys
from pathlib import Path
import json
import pymupdf4llm
import traceback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_pdf_in_folder(folder_path: Path) -> Path:
    for file in folder_path.iterdir():
        if file.suffix.lower() == ".pdf":
            return file
    return None


def process_pdf(pdf_path: Path, output_dir: Path):
    print(f"  Processing PDF: {pdf_path.name}")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        md_data = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=True,
            image_path=str(images_dir),
            dpi=150,
            force_text=True,
        )
    except Exception as e:
        print(f"  ERROR: Failed to process {pdf_path.name}: {e}")
        print(f"  Skipping this file")
        return 0

    output_data = []
    for page_data in md_data:
        page_info = {
            "page_number": page_data.get("metadata", {}).get("page_number", 0),
            "type": "markdown_with_images",
            "content": page_data.get("text", ""),
            "image_files": [str(img_path) for img_path in page_data.get("images", [])],
        }
        output_data.append(page_info)

    with open(output_dir / "content.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return len(output_data)


if __name__ == "__main__":
    base_dir = Path("/home/user-13/reranking_multimodal_data/data/datasets/docbench")
    blocks_base = Path("/home/user-13/reranking_multimodal_data/data/blocks_pymupdf")
    blocks_base.mkdir(parents=True, exist_ok=True)

    processed_folders = 0
    total_pages = 0

    for folder_num in range(0, 229):
        folder = base_dir / str(folder_num)
        if not folder.exists():
            continue

        pdf_file = find_pdf_in_folder(folder)
        if pdf_file is None:
            print(f"Folder {folder_num}: No PDF found")
            continue

        output_dir = blocks_base / str(folder_num)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing folder {folder_num}:")
        pages = process_pdf(pdf_file, output_dir)

        if pages > 0:
            total_pages += pages
            processed_folders += 1
            print(f"  Saved {pages} pages and images to {output_dir}")

    print(f"\nCompleted: processed {processed_folders} folders, {total_pages} total pages")
