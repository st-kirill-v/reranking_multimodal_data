import torch
from transformers import AutoModel, AutoProcessor
from pathlib import Path
import faiss
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import time
import os

os.environ["TORCH_LOAD_META_DEVICE"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"
index_dir.mkdir(exist_ok=True)

print("\nLoading Nemotron Embed model...")
start_time = time.time()
model_path = project_root / "models" / "nemotron" / "embed-vl-1b-v2"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
    local_files_only=True,
).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
print(f"Model loaded in {time.time() - start_time:.2f}s")

print("\nCollecting pages from all folders...")
pages = []

doc_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and f.name.isdigit()])

for folder in tqdm(doc_folders, desc="Scanning folders"):
    pages_dir = folder / "extracted" / "pages"
    if not pages_dir.exists():
        continue
    for img_path in sorted(pages_dir.glob("*.png")):
        page_num = int(img_path.stem.split("_")[1])
        pages.append({"folder": folder.name, "page": page_num, "path": str(img_path)})

print(f"Total pages found: {len(pages)}")
if not pages:
    raise RuntimeError("No pages found")

embed_dim = 2048
index = faiss.IndexFlatIP(embed_dim)

print("\nEncoding pages with Nemotron Embed...")
batch_size = 64
all_embeddings = []
metadata = []

total_batches = (len(pages) + batch_size - 1) // batch_size
pbar = tqdm(total=total_batches, desc="Encoding batches", unit="batch")

for i in range(0, len(pages), batch_size):
    batch = pages[i : i + batch_size]
    images = []
    valid_indices = []
    for j, p in enumerate(batch):
        try:
            img = Image.open(p["path"]).convert("RGB")
            images.append(img)
            valid_indices.append(j)
        except Exception as e:
            print(f"Warning: Could not load {p['path']}: {e}")
            continue
    if not images:
        pbar.update(1)
        continue

    with torch.inference_mode():
        embs = model.encode_documents(images=images)
        embs = embs / embs.norm(p=2, dim=-1, keepdim=True)

    all_embeddings.append(embs.cpu().float().numpy())
    for idx_in_batch, orig_idx in enumerate(valid_indices):
        p = batch[orig_idx]
        metadata.append(
            {"folder": p["folder"], "page": p["page"], "path": p["path"], "index": len(metadata)}
        )
    pbar.update(1)
    pbar.set_postfix({"batch": f"{len(images)}/{batch_size}", "processed": len(metadata)})

pbar.close()

all_embeddings = np.vstack(all_embeddings)
index.add(all_embeddings)
faiss.write_index(index, str(index_dir / "pages_nemotron.index"))
with open(index_dir / "metadata_nemotron.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nNemotron index saved to {index_dir / 'pages_nemotron.index'}")
print(f"Total vectors: {len(metadata)}")
print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
