import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List, Dict, Any
import os
import re


class QwenVLTableGenerator:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Qwen3-VL-8B-Instruct on {self.device}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True
        )

        print("Qwen3-VL-8B-Instruct loaded successfully")

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

        system_prompt = """You are a precise RAG assistant. Extract or calculate answers from tables and text.

RULES:
1. Output a COMPLETE sentence that answers the question.
2. Include the exact value in the sentence.
3. Do not output just the number or just the term.
4. If calculation is needed, perform it and include the result in the sentence.
5. NEVER use commas in numbers: write 10000 not 10,000.

Example 1 (number):
Image: [Table: Countries and their capitals populations: Tokyo=14M, Delhi=32M, Shanghai=24M]
Question: What is the population of Shanghai?
Answer: The population of Shanghai is 24 million.

Example 2 (term):
Image: [Text: "The Adam optimizer is widely used for training neural networks due to its adaptive learning rate"]
Question: What optimizer is mentioned?
Answer: The Adam optimizer is mentioned.

Example 3 (description):
Image: [Text: "The main contribution of this paper is a novel attention mechanism that reduces computational complexity from quadratic to linear"]
Question: What is the main contribution of this paper?
Answer: The main contribution is a novel attention mechanism that reduces computational complexity from quadratic to linear.

Example 4 (sum):
Image: [Table: Company sales: Q1=12500, Q2=13800, Q3=14200, Q4=15600]
Question: What is the total sales for Q3 and Q4?
Answer: The total sales for Q3 and Q4 is 29800.

Example 5 (model name + score):
Image: [Table: Car performance: Tesla=85%, BMW=92%, Audi=78%, Mercedes=89%]
Question: Which car brand achieved the highest performance score?
Answer: BMW achieved the highest performance score with 92%.

Now answer the question following the same format."""

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
                    **inputs, max_new_tokens=300, temperature=0.5, do_sample=True, top_p=0.9
                )

            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            answer = answer.replace("Answer:", "").replace("answer:", "").strip()

            if answer and answer != "NOT FOUND" and len(answer) > 0:
                return answer

        return "NOT FOUND"

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "table_extractor",
            "name": "qwen3-vl-table-extractor",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "device": self.device,
            "temperature": 0.5,
            "max_tokens": 300,
            "license": "Apache 2.0",
        }


def create_table_generator(device: str = None):
    return QwenVLTableGenerator(device=device)


def create_qwen_generator(device: str = None):
    return QwenVLTableGenerator(device=device)