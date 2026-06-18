from __future__ import annotations

import re
import time
from typing import Any

import torch
from PIL import Image

from src.core.generators.qwen_vl_generator import QwenVLTableGenerator
from src.prompts.docbench_hybrid_prompts import (
    get_docbench_prompt,
    get_docbench_prompt_name,
)


class DocBenchHybridGenerator(QwenVLTableGenerator):
    prompt_profile = "docbench_hybrid_v1"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_prompt_name = ""
        self.last_prompt_profile = self.prompt_profile

    def generate_answer_for_type(
        self,
        *,
        query: str,
        question_type: str,
        context_mode: str,
        context_images: list[Image.Image] | None = None,
        context_text: str | None = None,
    ) -> str:
        self.last_prompt_name = get_docbench_prompt_name(question_type, context_mode)
        system_prompt = get_docbench_prompt(question_type, context_mode)
        return self._generate_with_system_prompt(
            query=query,
            system_prompt=system_prompt,
            context_images=context_images,
            context_text=context_text,
        )

    def generate_answer(
        self,
        query: str,
        context_images: list[Image.Image] | None = None,
        context_text: str | None = None,
    ) -> str:
        question_type = "multimodal-t" if context_images else "text-only"
        context_mode = "image" if context_images else "text"
        return self.generate_answer_for_type(
            query=query,
            question_type=question_type,
            context_mode=context_mode,
            context_images=context_images,
            context_text=context_text,
        )

    def _generate_with_system_prompt(
        self,
        *,
        query: str,
        system_prompt: str,
        context_images: list[Image.Image] | None,
        context_text: str | None,
    ) -> str:
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
            user_text += f"Context:\n{context_text}\n\n"
        user_text += f"Question: {query}\n\n"

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
            text=text,
            images=resized_images if resized_images else None,
            return_tensors="pt",
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

        answer = re.sub(r"^(assistant|user)\s*:?\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = answer.strip('"').strip("'")
        return answer or "NOT FOUND"


def create_docbench_hybrid_generator(
    device: str | None = None,
    max_image_long_edge: int | None = 1600,
    load_4bit: bool | None = None,
    max_new_tokens: int | None = None,
    prompt_style: str | None = None,
    do_sample: bool | None = None,
    max_images: int | None = None,
    answer_refine: str | None = None,
) -> DocBenchHybridGenerator:
    return DocBenchHybridGenerator(
        device=device,
        max_image_long_edge=max_image_long_edge,
        load_4bit=load_4bit,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style,
        do_sample=do_sample,
        max_images=max_images,
        answer_refine=answer_refine,
    )
