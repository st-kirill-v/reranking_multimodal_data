import fitz
import json
from pathlib import Path

# Путь к данным (относительный)
data_path = Path(__file__).parent.parent / "data" / "datasets" / "docbench"

doc_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and f.name.isdigit()])
print(f"Found {len(doc_folders)} document folders")

total_stats = {
    "total_docs": 0,
    "total_pages": 0,
    "total_questions": 0,
    "multimodal_t": 0,
    "multimodal_f": 0,
    "errors": [],
}

for folder in doc_folders:
    try:
        print(f"\nProcessing folder {folder.name}")

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            total_stats["errors"].append(f"{folder.name}: no PDF")
            continue
        pdf_file = pdf_files[0]

        jsonl_file = folder / f"{folder.name}_qa.jsonl"
        if not jsonl_file.exists():
            total_stats["errors"].append(f"{folder.name}: no {folder.name}_qa.jsonl")
            continue

        doc = fitz.open(pdf_file)

        extracted_dir = folder / "extracted"
        extracted_dir.mkdir(exist_ok=True)
        pages_dir = extracted_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append({"page": page_num + 1, "text": text})

            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_path = pages_dir / f"page_{page_num + 1}.png"
            pix.save(img_path)

        with open(extracted_dir / "pages_text.json", "w", encoding="utf-8") as f:
            json.dump(pages_text, f, ensure_ascii=False, indent=2)

        questions = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line.strip()))

        multimodal_t = sum(1 for q in questions if q.get("type") == "multimodal-t")
        multimodal_f = sum(1 for q in questions if q.get("type") == "multimodal-f")

        total_stats["total_docs"] += 1
        total_stats["total_pages"] += len(doc)
        total_stats["total_questions"] += len(questions)
        total_stats["multimodal_t"] += multimodal_t
        total_stats["multimodal_f"] += multimodal_f

        doc_report = {
            "folder": folder.name,
            "pdf_file": pdf_file.name,
            "total_pages": len(doc),
            "total_questions": len(questions),
            "multimodal_t": multimodal_t,
            "multimodal_f": multimodal_f,
        }

        with open(extracted_dir / "doc_report.json", "w", encoding="utf-8") as f:
            json.dump(doc_report, f, indent=2)

        if doc and not doc.is_closed:
            doc.close()

        print(f"Done: {len(doc)} pages, {len(questions)} questions")

    except Exception as e:
        total_stats["errors"].append(f"{folder.name}: error - {str(e)[:100]}")
        continue

print("\n" + "=" * 60)
print("FINAL STATISTICS")
print("=" * 60)

print(f"\nSuccessfully processed: {total_stats['total_docs']} / {len(doc_folders)}")
print(f"Total pages: {total_stats['total_pages']}")
print(f"Total questions: {total_stats['total_questions']}")
print(f"multimodal-t (tables): {total_stats['multimodal_t']}")
print(f"multimodal-f (figures): {total_stats['multimodal_f']}")

if total_stats["errors"]:
    print(f"\nErrors: {len(total_stats['errors'])}")
    for err in total_stats["errors"][:10]:
        print(f"  • {err}")

stats_file = data_path / "dataset_stats.json"
with open(stats_file, "w", encoding="utf-8") as f:
    json.dump(total_stats, f, indent=2)

print(f"\nStats saved to {stats_file}")
