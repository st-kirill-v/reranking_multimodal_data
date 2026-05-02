import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List, Dict, Any
import time


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

    def _resize_image(self, img: Image.Image, max_size: int = 800) -> Image.Image:
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def generate_answer(
        self, query: str, context_images: List[Image.Image] = None, context_text: str = None
    ) -> str:
        if not query or not query.strip():
            return "Please provide a question."

        if not context_images and not context_text:
            return "NOT FOUND"

        system_prompt = """You are a precise RAG assistant. Extract or calculate answers from tables and text.

RULES:
1. Output a COMPLETE sentence that answers the question.
2. Include the exact value in the sentence.
3. Do not output just the number or just the term.
4. If calculation is needed, perform it and include the result in the sentence.
5. NEVER use commas in numbers: write 10000 not 10,000.
6. When extracting values from tables: verify the exact row name and column name match the question. Double-check the value before outputting.
7. When extracting terms from tables: compare your extracted term with the exact text in the table cell. Ensure it matches.
8. If the question contains words 'photo', 'image', or 'picture': look for visual content in the image. If no relevant visual information exists, return 'NOT FOUND'.
9. Pay attention to units of measurement. Check the question to understand what unit is requested (percentage, number, million, etc.). Output the value in the requested unit.
10. When the answer is a percentage, include the percent sign in the output.
11. When the question asks for a difference or change, calculate the absolute difference and include the sign if negative.
12. If the table contains multiple rows with similar names (like 'Multi Task' and '+Coreference'), prefer the row that exactly matches the keyword in the question.


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

        # Уменьшаем изображения
        resized_images = []
        if context_images:
            for img in context_images[:5]:
                resized_images.append(self._resize_image(img, max_size=800))

        # Собираем текстовый контент
        user_text = ""
        if context_text:
            user_text += f"Tables:\n{context_text}\n\n"
        user_text += f"Question: {query}\n\n"

        # Формируем сообщение
        if resized_images:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
            for img in resized_images:
                messages[1]["content"].insert(0, {"type": "image", "image": img})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text, images=resized_images if resized_images else None, return_tensors="pt"
        ).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        print(f"    [VLM] Generation time: {time.time() - start_time:.2f}s")

        answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем ответ из <ANSWER> блока
        if "<ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        elif "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()

        # Очищаем от лишнего
        answer = answer.replace("assistant", "").strip()
        answer = answer.replace("user", "").strip()
        answer = answer.strip('"').strip("'")

        # Если ответ пустой или слишком короткий
        if not answer or len(answer) < 1:
            return "NOT FOUND"

        # Убираем возможный дубликат вопроса из ответа
        if query.lower() in answer.lower():
            # Оставляем только часть после вопроса
            parts = answer.lower().split(query.lower())
            if len(parts) > 1:
                answer = parts[-1].strip()

        return answer

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "table_extractor",
            "name": "qwen3-vl-table-extractor",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "device": self.device,
        }


def create_table_generator(device: str = None):
    return QwenVLTableGenerator(device=device)
