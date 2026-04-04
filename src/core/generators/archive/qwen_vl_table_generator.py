import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from typing import List, Dict, Any
import os
import re


class QwenVLTableGenerator:
    """Генератор для извлечения данных из таблиц с помощью Qwen2.5-VL"""

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

        print(f"Loading Qwen2.5-VL-Table-Extraction (specialized for tables) on {self.device}...")

        # Загрузка базовой модели
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Glazkov/qwen2.5-vl-table-extraction",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,
            ignore_mismatched_sizes=True,
        ).eval()

        # Загрузка LoRA весов если необходимо
        if use_lora and os.path.exists(self.lora_path):
            print(f"Loading LoRA weights from {self.lora_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            print("LoRA weights loaded successfully")
        else:
            if use_lora:
                print(f"LoRA path not found: {self.lora_path}")
                print("   Using base model without LoRA")
            self.model = base_model

        self.processor = AutoProcessor.from_pretrained(
            "Glazkov/qwen2.5-vl-table-extraction", trust_remote_code=True
        )

        print("Qwen2.5-VL-Table-Extraction loaded successfully")

    def _clean_prefix(self, answer: str) -> str:
        """Очищает префиксы из ответа модели"""
        answer = answer.replace("assistant", "").strip()
        answer = answer.replace("Answer:", "").replace("answer:", "").strip()
        answer = answer.strip('"').strip("'")
        return answer

    def _extract_numbers(self, answer: str) -> List[str]:
        """Извлекает числа из ответа"""
        return re.findall(r"\d+(?:\.\d+)?", answer.replace(",", ""))

    def _extract_term(self, answer: str) -> str:
        """Извлекает термин/название из ответа"""
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
        """Очищает описательный ответ от стоп-слов"""
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
        """Определяет тип ожидаемого ответа по вопросу"""
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
        """Нормализует ответ в соответствии с типом вопроса"""
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
        """Генерирует ответ на вопрос на основе предоставленных изображений"""
        if not query or not query.strip():
            return "Please provide a question."

        if not context_images:
            return f"No relevant pages found for: '{query}'"

        # Системный промпт с инструкциями
        system_prompt = """You are a precise RAG assistant. Extract or calculate answers from tables and text.

RULES:
1. Find the EXACT row and column mentioned in the question.
2. Read the value at their intersection.
3. If the answer is a NUMBER: output only the number (e.g., "425", "88.3").
4. If the answer is a TERM/NAME: output the term (e.g., "Transformer", "SGD").
5. If the answer is a MODEL NAME with a SCORE: output both (e.g., "BMW 92").
6. If the answer is a DESCRIPTION: output a concise but complete sentence.
7. NEVER use commas in numbers: write 10000 not 10,000.
8. If calculation is needed (sum, difference), perform it and output the result.

EXAMPLES:

Example 1 (number):
Image: [Table: Countries and their capitals populations: Tokyo=14M, Delhi=32M, Shanghai=24M]
Question: What is the population of Shanghai?
Answer: 24

Example 2 (term):
Image: [Text: "The Adam optimizer is widely used for training neural networks due to its adaptive learning rate"]
Question: What optimizer is mentioned?
Answer: Adam

Example 3 (description):
Image: [Text: "The main contribution of this paper is a novel attention mechanism that reduces computational complexity from quadratic to linear"]
Question: What is the main contribution of this paper?
Answer: The main contribution is a novel attention mechanism that reduces computational complexity from quadratic to linear.

Example 4 (sum):
Image: [Table: Company sales: Q1=12500, Q2=13800, Q3=14200, Q4=15600]
Question: What is the total sales for Q3 and Q4?
Answer: 29800

Example 5 (model name + score):
Image: [Table: Car performance: Tesla=85%, BMW=92%, Audi=78%, Mercedes=89%]
Question: Which car brand achieved the highest performance score?
Answer: BMW 92

Now answer the question following the same format."""

        for img in context_images[:5]:
            # Формируем prompt с правильными токенами
            prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Question: {query}
Answer:
<|im_end|>
<|im_start|>assistant
"""

            inputs = self.processor(
                text=prompt, images=[img], return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Извлекаем ответ после assistant
            if "<|im_start|>assistant" in answer:
                answer = answer.split("<|im_start|>assistant")[-1].strip()
            elif "assistant" in answer:
                answer = answer.split("assistant")[-1].strip()

            answer = answer.replace("</s>", "").replace("<|im_end|>", "").strip()

            answer = self._normalize_answer(answer, query)

            if answer and answer != "NOT FOUND" and len(answer) > 0:
                return answer

        return "NOT FOUND"

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        return {
            "type": "table_extractor",
            "name": "qwen2.5-vl-table-extraction",
            "model": "Glazkov/qwen2.5-vl-table-extraction",
            "device": self.device,
            "temperature": 0.1,
            "max_tokens": 150,
            "license": "Apache 2.0",
            "use_lora": self.use_lora,
            "lora_path": self.lora_path if self.use_lora else None,
        }


def create_table_generator(device: str = None, use_lora: bool = False, lora_path: str = None):
    """
    Создаёт генератор для извлечения данных из таблиц.

    Args:
        device: устройство ('cuda' или 'cpu')
        use_lora: использовать ли LoRA адаптеры (по умолчанию False)
        lora_path: путь к LoRA весам (если None, используется стандартный путь)
    """
    return QwenVLTableGenerator(device=device, use_lora=use_lora, lora_path=lora_path)
