import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from typing import List, Dict, Any
import time
import re
import importlib.util


class QwenVLTableGenerator:
    def __init__(
        self,
        device: str = None,
        timeout_seconds: int = 180,
        max_image_long_edge: int | None = 1600,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_image_long_edge = max_image_long_edge

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Qwen3-VL-8B-Instruct on {self.device}")
        print(f"VLM timeout set to {self.timeout_seconds} seconds")

        quantization_config = None
        if importlib.util.find_spec("bitsandbytes"):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            print("bitsandbytes is not installed; loading Qwen3-VL without 4-bit quantization")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True
        )

        print("Qwen3-VL-8B-Instruct loaded successfully")

    def _resize_image(self, img: Image.Image) -> Image.Image:
        if self.max_image_long_edge and max(img.size) > self.max_image_long_edge:
            ratio = self.max_image_long_edge / max(img.size)
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

        system_prompt = """You are a precise RAG assistant. Your task is to extract or calculate information from images (tables, charts, text, or photos).

ANALYSIS STEP:
Before providing the final answer, perform these steps internally:
1. Locate the specific table or text block relevant to the question.
2. Identify the EXACT row header and column header.
3. If calculation is required, write out the formula: (Value A [op] Value B = Result).
4. If a photo is involved, scan for specific objects and their relative positions.

RULES:
1. ANSWER FORMAT: Output exactly one COMPLETE sentence. No preamble.
2. PRECISION: Use the exact values and terminology from the source. Do not paraphrase names of models, datasets, or metrics.
3. NUMBERS: Write numbers exactly as they appear in the source, unless a specific calculation is requested.
4. CALCULATIONS: If the question asks for a total, difference, or change, you MUST perform the math and state the result in the sentence. Show no intermediate steps in the final output, but ensure the math is correct.
5. UNITS: Always include units (e.g., %, million, lbs, tokens, DKK) as specified in the table or question.
6. VISUALS: For questions about "photos", "pictures", or "images", describe the visual evidence. If the information is not visually present (even if it's in the text), return "NOT FOUND".
7. AMBIGUITY: If multiple tables exist, use the one that most closely matches the keywords in the question. Verify column names like "Test", "Train", or "Dev".. If the table contains multiple rows with similar names (like 'Multi Task' and '+Coreference'), prefer the row that exactly matches the keyword in the question.

EXAMPLES:

Example 1 (number):
Image: [Table: Countries and their capitals populations: Tokyo=14M, Delhi=32M, Shanghai=24M]
Question: What is the population of Shanghai?
Internal Thought: (Locate row: Shanghai -> Identify value: 24M)
Answer: The population of Shanghai is 24 million.

Example 2 (term):
Image: [Text: "The Adam optimizer is widely used for training neural networks due to its adaptive learning rate"]
Question: What optimizer is mentioned?
Internal Thought: (Scan text for optimizer name -> Match found: Adam)
Answer: The Adam optimizer is mentioned.

Example 3 (description):
Image: [Text: "The main contribution of this paper is a novel attention mechanism that reduces computational complexity from quadratic to linear"]
Question: What is the main contribution of this paper?
Internal Thought: (Scan text for 'main contribution' -> Extract exact phrasing)
Answer: The main contribution is a novel attention mechanism that reduces computational complexity from quadratic to linear.

Example 4 (sum):
Image: [Table: Company sales: Q1=12500, Q2=13800, Q3=14200, Q4=15600]
Question: What is the total sales for Q3 and Q4?
Internal Thought: (Locate Q3: 14200 -> Locate Q4: 15600 -> Calculate: 14200 + 15600 = 29800)
Answer: The total sales for Q3 and Q4 is 29800.

Example 5 (model name + score):
Image: [Table: Car performance: Tesla=85%, BMW=92%, Audi=78%, Mercedes=89%]
Question: Which car brand achieved the highest performance score?
Internal Thought: (Scan values: 85, 92, 78, 89 -> Find max: 92 -> Match to brand: BMW)
Answer: BMW achieved the highest performance score with 92%.

Now answer the question following the exact same format. Output your internal reasoning starting with "Internal Thought:"""

        # Уменьшаем изображения
        resized_images = []
        if context_images:
            for img in context_images[:5]:
                resized_images.append(self._resize_image(img))

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

        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)

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

    def postprocess_answer(answer, query):
        if not answer or answer in ["NOT FOUND", "TIMEOUT", "ERROR"]:
            return answer

        # 1. Проценты
        if "%" not in answer and any(
            w in query.lower() for w in ["%", "percent", "percentage", "accuracy"]
        ):
            numbers = re.findall(r"\d+(?:\.\d+)?", answer)
            for n in numbers:
                if float(n) < 1:
                    answer = f"{float(n) * 100:.0f}%"
                    break

        # 2. Удаление запятых из чисел
        answer = re.sub(r"(\d),(\d)", r"\1\2", answer)

        # 3. Удаление преамбулы
        for prefix in ["assistant", "user", "I think", "The answer is", "Answer:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix) :].strip()

        return answer

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "table_extractor",
            "name": "qwen3-vl-table-extractor",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "device": self.device,
        }


def create_table_generator(device: str = None, max_image_long_edge: int | None = 1600):
    return QwenVLTableGenerator(device=device, max_image_long_edge=max_image_long_edge)
