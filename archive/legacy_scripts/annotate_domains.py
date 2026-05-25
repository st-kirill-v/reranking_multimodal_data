import os
import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/user-13/reranking_multimodal_data/data/datasets/docbench")
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DOMAINS = {"academic", "financial", "government", "legal", "news"}
SKIP_DONE = True

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a document domain classifier. "
    "Classify the given document page into exactly one of these domains: "
    "academic, financial, government, legal, news.\n\n"
    "Classification rules:\n"
    "- academic: research papers, two-column layout, citations, abstract, model names (BERT, LSTM, KGLM)\n"
    "- financial: annual reports, company logos, revenue, profit, million, billion\n"
    "- government: official decrees, U.S. Department of State, GovInfo, policy documents\n"
    "- legal: Library of Congress, contracts, clauses, liability, court, jurisdiction\n"
    "- news: New York Times front pages, newspaper layout, headlines, bylines, 'according to'\n\n"
    "Respond with ONLY a single word — the domain name."
)

USER_PROMPT = (
    "Analyze this document page and classify it into one domain: "
    "academic, financial, government, legal, news.\n"
    "Output only the domain name, nothing else."
)


# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded\n")
    return model, processor


# ─── INFERENCE ────────────────────────────────────────────────────────────────
def classify_document(image_path: Path, model, processor) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    domain = re.sub(r"[^a-z]", "", response.strip().lower())
    return domain if domain in DOMAINS else _fallback(response)


def _fallback(raw: str) -> str:
    raw = raw.lower()
    for d in DOMAINS:
        if d in raw:
            return d
    return "unknown"


# ─── СОЗДАЁМ doc_report.json ЕСЛИ ЕГО НЕТ ─────────────────────────────────────
def ensure_report_file(folder: Path, pdf_file: str = None) -> dict:
    """Создаёт doc_report.json если его нет"""
    report_path = folder / "doc_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Если нет — создаём из того, что есть
    if pdf_file is None:
        # Ищем любой PDF в папке
        pdf_files = list(folder.glob("*.pdf"))
        pdf_name = pdf_files[0].name if pdf_files else f"{folder.name}.pdf"
    else:
        pdf_name = pdf_file

    report = {
        "folder": folder.name,
        "pdf_file": pdf_name,
        "total_pages": None,  # можно будет заполнить позже
        "total_questions": None,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():

    torch.cuda.empty_cache()
    model, processor = load_model()

    folders = []
    for p in BASE_DIR.iterdir():
        if p.is_dir() and p.name.isdigit():
            folder_id = int(p.name)
            if 181 <= folder_id <= 228:
                folders.append(p)
    folders.sort(key=lambda p: int(p.name))

    total = len(folders)
    print(f"Found {total} folders\n")

    stats = {d: 0 for d in DOMAINS}
    stats["unknown"] = 0
    errors = []
    skipped = 0

    for idx, folder in enumerate(folders, start=1):
        report_path = folder / "doc_report.json"
        image_path = folder / "extracted" / "pages" / "page_1.png"

        # Создаём report если нет
        report = ensure_report_file(folder)
        pdf_name = report.get("pdf_file", "unknown")

        if SKIP_DONE and "domain" in report and report["domain"] in DOMAINS:
            print(
                f"[{idx}/{total}] {folder.name} | {pdf_name}: already classified -> {report['domain']}, skipping"
            )
            skipped += 1
            stats[report["domain"]] = stats.get(report["domain"], 0) + 1
            continue

        if not image_path.exists():
            print(f"[{idx}/{total}] {folder.name} | {pdf_name}: page_1.png not found, skipping")
            errors.append({"folder": folder.name, "error": "page_1.png not found"})
            continue

        print(f"[{idx}/{total}] {folder.name} | {pdf_name}: classifying...", end=" ", flush=True)

        try:
            domain = classify_document(image_path, model, processor)
        except Exception as e:
            print(f"error: {e}")
            errors.append({"folder": folder.name, "error": str(e)})
            domain = "unknown"

        report["domain"] = domain
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"done -> {domain}")
        stats[domain] = stats.get(domain, 0) + 1

        torch.cuda.empty_cache()

    # ─── SUMMARY ──────────────────────────────────────────────────────────────
    print("Classification summary:")
    for domain, count in sorted(stats.items()):
        print(f"  {domain:<12}: {count}")
    print(f"\n  Total   : {total}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {len(errors)}")

    if errors:
        err_path = BASE_DIR / "classification_errors.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n  Error log saved: {err_path}")


if __name__ == "__main__":
    main()
