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

    def _clean_prefix(self, answer: str) -> str:
        answer = answer.replace("assistant", "").strip()
        answer = answer.replace("Answer:", "").replace("answer:", "").strip()
        answer = answer.strip('"').strip("'")
        return answer

    def _extract_numbers(self, answer: str) -> List[str]:
        return re.findall(r"\d+(?:\.\d+)?", answer.replace(",", ""))

    def _extract_term(self, answer: str) -> str:
        quoted = re.findall(r'"([^"]+)"', answer)
        if quoted:
            return quoted[0].strip()

        patterns = [
            r"([A-Z][A-Za-z0-9]*[-+][A-Za-z0-9]+)",
            r"([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*)",
            r"([A-Z]+(?:-[A-Z]+)+)",
            r"([A-Z]{2,})",
        ]

        for pattern in patterns:
            match = re.search(pattern, answer)
            if match:
                return match.group(1)

        words = answer.lower().split()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "according", "to", "table"}
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        if filtered:
            return " ".join(filtered[:2])

        return ""

    def _clean_description(self, answer: str) -> str:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "according",
            "to",
            "table",
            "figure",
            "function",
            "used",
            "for",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "as",
            "be",
            "have",
            "has",
            "had",
            "this",
            "that",
            "and",
            "or",
            "but",
        }

        words = answer.lower().split()
        filtered = [w for w in words if w not in stop_words and len(w) > 1]

        if filtered:
            return " ".join(filtered)

        return answer.strip()

    def _get_answer_type(self, query: str) -> str:
        q = query.lower()

        numeric_patterns = [
            "how many",
            "how much",
            "what is the accuracy",
            "what is the score",
            "what is the percentage",
            "what is the value",
            "by how much",
            "what is the total",
            "calculate",
            "what is the number",
            "what is the perplexity",
            "what is the ratio",
            "what is the f1",
        ]
        if any(p in q for p in numeric_patterns):
            return "numeric"

        term_patterns = [
            "which model",
            "which function",
            "what function",
            "what method",
            "what technique",
            "what is the name",
            "which dataset",
            "which algorithm",
        ]
        if any(p in q for p in term_patterns):
            return "term"

        hybrid_patterns = [
            "which model achieved",
            "which dataset experienced",
            "which variant has the highest",
            "which model has the highest",
            "which method achieved",
            "what model achieved",
        ]
        if any(p in q for p in hybrid_patterns):
            return "hybrid"

        return "descriptive"

    def _normalize_answer(self, answer: str, query: str) -> str:
        if not answer or answer == "NOT FOUND":
            return answer

        answer = self._clean_prefix(answer)
        numbers = self._extract_numbers(answer)
        answer_type = self._get_answer_type(query)

        if answer_type == "numeric" and numbers:
            return numbers[0]

        if answer_type == "term":
            term = self._extract_term(answer)
            if term:
                return term.lower()

        if answer_type == "hybrid":
            term = self._extract_term(answer)
            if numbers and term:
                return f"{term} {numbers[0]}"
            if numbers:
                return numbers[0]
            if term:
                return term

        if answer_type == "descriptive":
            return self._clean_description(answer)

        if numbers:
            return numbers[0]

        return self._clean_description(answer)

    def generate_answer(self, query: str, context_images: List[Image.Image]) -> str:
        if not query or not query.strip():
            return "Please provide a question."

        if not context_images:
            return f"No relevant pages found for: '{query}'"

        system_prompt = """You are a precise table extraction assistant.
You MUST extract the EXACT value from the table.

CRITICAL RULES:
1. Look at the table in the image.
2. Find the row and column mentioned in the question.
3. Output ONLY the value at that intersection.
4. DO NOT add any extra words.
5. DO NOT explain your answer.
6. If the answer is a number, output just the number.
7. If you need to sum two numbers, calculate and output only the result.
8. NEVER use commas in numbers: write 10000 not 10,000.

EXAMPLES:

Example 1 (extract number from table):
Image: [Table: GL -> EN = 11.5, PT -> EN = 30.6, AZ -> EN = 2.1]
Question: What is the aligned BLEU score for GL -> EN?
Answer: 11.5

Example 2 (extract number from table):
Image: [Table: Training sentences: GL-EN=10017, PT-EN=51785]
Question: How many training sentences for PT-EN?
Answer: 51785

Example 3 (sum two numbers):
Image: [Table: Q1=12500, Q2=13800, Q3=14200, Q4=15600]
Question: What is the total sales for Q3 and Q4?
Answer: 29800

Example 4 (extract name from table):
Image: [Table: Model A=92%, Model B=95.8%, Model C=88.1%]
Question: Which model achieved the highest accuracy?
Answer: Model B

Example 5 (extract name with score):
Image: [Table: Tesla=85%, BMW=92%, Audi=78%, Mercedes=89%]
Question: Which car brand achieved the highest performance score?
Answer: BMW 92

Now answer the question using the table in the image. Output ONLY the answer, no explanations."""

        for img in context_images[:5]:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Question: {query}\nAnswer:"},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=text, images=[img], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=500, temperature=0.1, do_sample=False
                )

            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            answer = answer.replace("Answer:", "").replace("answer:", "").strip()

            answer = self._normalize_answer(answer, query)

            if answer and answer != "NOT FOUND" and len(answer) > 0:
                return answer

        return "NOT FOUND"

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "table_extractor",
            "name": "qwen2-vl-table-extractor",
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
