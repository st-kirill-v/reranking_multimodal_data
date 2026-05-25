from __future__ import annotations

import os
import re
import time
from typing import Any

import torch
from PIL import Image

from src.core.generators.qwen_vl_generator import QwenVLTableGenerator as BaseQwenVLTableGenerator


SMART_UNIVERSAL_SYSTEM_PROMPT = """You are a precise multimodal RAG assistant. Your task is to answer questions using only the provided page images, including tables, charts, diagrams, text blocks, and photos.

Before answering, internally choose the needed evidence type:
- text: use the exact relevant phrase or sentence.
- table: identify the relevant table, exact row, exact column, and cell value.
- calculation: identify the source values, compute the requested result, and preserve units.
- visual: inspect the relevant title, labels, axes, legend, annotations, components, arrows, or objects.

INTERNAL STEPS:
1. Locate the single most relevant table, figure, chart, diagram, photo, or text block.
2. Match the question wording to the exact entity, row, column, label, component, or text span.
3. Extract the exact value or phrase.
4. If calculation is required, compute it from the extracted source values.
5. Verify that the answer comes from the same document/page/table/figure unless the question explicitly asks for comparison.

RULES:
1. Output exactly one complete sentence.
2. Do not output reasoning, internal steps, or question type.
3. Use exact terminology from the source. Do not paraphrase model names, datasets, metrics, components, entities, row names, or column names.
4. Write numbers exactly as they appear in the source, unless calculation is required.
5. Always include visible units such as %, million, billion, lbs, tokens, DKK, or MtCO2e.
6. If the question asks for a total, difference, ratio, increase, decrease, or percentage change, include the source values and the final result in one sentence.
7. If multiple similar tables, rows, columns, figures, or entities exist, choose the one that exactly matches the question wording.
8. Do not mix information from different documents, pages, tables, or figures unless the question explicitly requires it.
9. If the answer is not visible in the provided images, return exactly: NOT FOUND.
10. EXTRACTION PRIORITY:
If a final answer, metric, score, percentage, or result is explicitly written in the image, copy it directly.
Do not recompute or approximate values unless calculation is explicitly required and no final value is shown.
11. NO INFERENCE:
Do not infer missing values, labels, entities, or trends from context.
Only use information explicitly visible in the provided images.

12. EXACT MATCH PRIORITY:
When multiple similar row names, model names, datasets, metrics, or entities exist, prefer the exact textual match from the question instead of the closest semantic match.

13. CONSERVATIVE CALCULATION:
Only perform calculations when the question explicitly asks for a computed result such as total, difference, ratio, increase, decrease, or percentage change.
Otherwise copy the explicitly written value directly.

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

Now answer the question. Return only:
Answer: <one complete sentence>"""


class QwenVLPromptStyleGenerator(BaseQwenVLTableGenerator):
    def generate_answer(
        self,
        query: str,
        context_images: list[Image.Image] | None = None,
        context_text: str | None = None,
    ) -> str:
        if self.prompt_style != "smart_universal":
            return super().generate_answer(query, context_images, context_text)

        if not query or not query.strip():
            return "Please provide a question."

        if not context_images and not context_text:
            return "NOT FOUND"

        resized_images = []
        if context_images:
            for img in context_images[: self.max_images]:
                resized_images.append(self._resize_image(img))

        user_text = ""
        if context_text:
            user_text += f"Tables:\n{context_text}\n\n"
        user_text += f"Question: {query}\n\n"

        if resized_images:
            messages = [
                {"role": "system", "content": SMART_UNIVERSAL_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
            for img in resized_images:
                messages[1]["content"].insert(0, {"type": "image", "image": img})
        else:
            messages = [
                {"role": "system", "content": SMART_UNIVERSAL_SYSTEM_PROMPT},
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
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "use_cache": True,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            if self.do_sample:
                generation_kwargs.update({"temperature": 0.2, "top_p": 0.9})
            outputs = self.model.generate(**inputs, **generation_kwargs)
        print(f"    [VLM] Generation time: {time.time() - start_time:.2f}s")

        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        raw_answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        self.last_raw_output = raw_answer
        self.last_reasoning = self._extract_reasoning(raw_answer)

        answer = self._extract_final_answer(raw_answer)
        if self.answer_refine != "none":
            answer = self._refine_answer(query, raw_answer)
        self.last_answer = answer

        if "<ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        elif "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        elif "Internal Thought:" in answer:
            answer = answer.split("Internal Thought:")[-1].strip()
            if "Final answer:" in answer:
                answer = answer.split("Final answer:")[-1].strip()
            elif "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

        answer = re.sub(r"^(assistant|user)\s*:?\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = answer.strip('"').strip("'")

        if not answer or len(answer) < 1:
            return "NOT FOUND"

        if query.lower() in answer.lower():
            parts = answer.lower().split(query.lower())
            if len(parts) > 1:
                answer = parts[-1].strip()

        return answer


def create_table_generator(
    device: str | None = None,
    max_image_long_edge: int | None = 1600,
    load_4bit: bool | None = None,
    max_new_tokens: int | None = None,
    prompt_style: str | None = None,
    do_sample: bool | None = None,
    max_images: int | None = None,
    answer_refine: str | None = None,
) -> QwenVLPromptStyleGenerator:
    return QwenVLPromptStyleGenerator(
        device=device,
        max_image_long_edge=max_image_long_edge,
        load_4bit=load_4bit,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style or os.getenv("MMRAG_QWEN_PROMPT_STYLE", "concise"),
        do_sample=do_sample,
        max_images=max_images,
        answer_refine=answer_refine,
    )
