import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Glazkov/qwen2.5-vl-table-extraction",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
    ignore_mismatched_sizes=True,
).eval()

processor = AutoProcessor.from_pretrained(
    "Glazkov/qwen2.5-vl-table-extraction", trust_remote_code=True
)

# Load test image
img_path = (
    "/home/user-13/reranking_multimodal_data/data/datasets/docbench/1/extracted/pages/page_4.png"
)
img = Image.open(img_path).convert("RGB")

print("Diagnostic: Image tokenization")

print("\n1. Testing processor with text only (no image):")
inputs1 = processor(text="Hello", return_tensors="pt")
print(f"   Input shape: {inputs1['input_ids'].shape}")

print("\n2. Testing processor with image only:")
inputs2 = processor(images=[img], return_tensors="pt")
print(f"   Keys: {inputs2.keys()}")
if "pixel_values" in inputs2:
    print(f"   pixel_values shape: {inputs2['pixel_values'].shape}")

print("\n3. Testing processor with text + image (with image token):")
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs3 = processor(text=text, images=[img], return_tensors="pt")
print(f"   text: {text[:200]}...")
print(f"   input_ids shape: {inputs3['input_ids'].shape}")
print(f"   pixel_values shape: {inputs3['pixel_values'].shape}")

print("\n4. Testing forward pass:")
try:
    with torch.no_grad():
        outputs = model(**inputs3)
    print("   Forward pass successful!")
    print(f"   Output logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"   Forward pass failed: {e}")

print("Diagnostic complete")
