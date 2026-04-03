import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from typing import List, Dict, Any
import os
import re


class QwenVLTableGenerator:
    def __init__(self, device: str = None, use_lora: bool = True, lora_path: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_lora = use_lora

        if lora_path is None:
            self.lora_path = "/home/user-13/reranking_multimodal_data/models/qwen2vl_rag_lora"
        else:
            self.lora_path = lora_path

        print(f"Loading Qwen2-VL-7B-Instruct on {self.device}")

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()

        if use_lora and os.path.exists(self.lora_path):
            print(f"Loading LoRA weights from {self.lora_path}")
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            print("LoRA weights loaded successfully")
        else:
            if use_lora:
                print(f"LoRA path not found: {self.lora_path}")
                print("Using base model without LoRA")
            self.model = base_model

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True
        )

        print("Qwen2-VL-7B-Instruct loaded successfully")

    def generate_answer(self, query: str, context_images: List[Image.Image]) -> str:
        if not query or not query.strip():
            return "Please provide a question."

        if not context_images:
            return f"No relevant pages found for: '{query}'"

        system_prompt = """You are a precise information extraction assistant.

Your task is to extract ONLY the information that directly answers the question.

Rules:
1. Do not output entire tables.
2. Output only the specific row, column, and value that answers the question.
3. If the question asks for "aligned BLEU score for GL -> EN", find the row with GL -> EN and output the aligned score.
4. Output format: "Value: X" where X is the extracted number or text.
5. Do not add explanations or extra text.

Example:
Question: What is the aligned BLEU score for GL -> EN?
Output: Value: 11.5

Now extract only the information that answers the question."""

        for img in context_images[:5]:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=text, images=[img], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3000,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )

            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            if "assistant" in answer:
                answer = answer.split("assistant")[-1].strip()

            answer = answer.replace("Answer:", "").replace("answer:", "").strip()

            if answer and answer != "NOT FOUND" and len(answer) > 10:
                return answer

        return "NOT FOUND"

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "context_extractor",
            "name": "qwen2-vl-context-extractor",
            "model": "Qwen/Qwen2-VL-7B-Instruct",
            "device": self.device,
            "temperature": 0.7,
            "max_tokens": 3000,
            "license": "Apache 2.0",
            "use_lora": self.use_lora,
            "lora_path": self.lora_path if self.use_lora else None,
        }


def create_table_generator(device: str = None, use_lora: bool = True, lora_path: str = None):
    return QwenVLTableGenerator(device=device, use_lora=use_lora, lora_path=lora_path)


def create_qwen_generator(device: str = None, use_lora: bool = True, lora_path: str = None):
    return QwenVLTableGenerator(device=device, use_lora=use_lora, lora_path=lora_path)
