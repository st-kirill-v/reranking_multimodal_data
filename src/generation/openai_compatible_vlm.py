from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from src.prompts.docbench_hybrid_prompts import (
    DOCBENCH_MULTIMODAL_PROMPT_V2,
    get_docbench_prompt,
    get_docbench_prompt_name,
)


class OpenAICompatibleVLM:
    """OpenAI-compatible multimodal backend for DocBench VLM ablations."""

    generation_backend = "openai_compatible_vlm"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_COMPAT_API_KEY",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 192,
        timeout: float = 180.0,
    ) -> None:
        resolved_api_key = os.getenv(api_key_env) or api_key
        if not resolved_api_key:
            raise RuntimeError(
                f"Missing OpenAI-compatible API key. Set {api_key_env} in the same shell "
                "that runs the experiment, or pass api_key in config."
            )
        self.client = OpenAI(api_key=resolved_api_key, base_url=base_url, timeout=timeout)
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.last_raw_output = ""
        self.last_answer = ""
        self.last_reasoning = ""
        self.last_prompt_profile = "docbench_hybrid_v1"
        self.last_prompt_name = "DOCBENCH_MULTIMODAL_PROMPT_V2"
        self.last_error: str | None = None
        self.last_latency_generation = 0.0
        self.last_image_paths_sent: list[str | None] = []
        self.last_num_images_sent = 0

    def generate_answer(
        self,
        query: str,
        context_images: list[Image.Image | str | Path] | None = None,
        context_text: str | None = None,
        *,
        image_paths: list[str | Path | None] | None = None,
    ) -> str:
        return self.generate_answer_for_type(
            query=query,
            question_type="multimodal-t",
            context_mode="image",
            context_images=context_images,
            context_text=context_text,
            image_paths=image_paths,
        )

    def generate_answer_for_type(
        self,
        *,
        query: str,
        question_type: str,
        context_mode: str,
        context_images: list[Image.Image | str | Path] | None = None,
        context_text: str | None = None,
        image_paths: list[str | Path | None] | None = None,
    ) -> str:
        self.last_error = None
        self.last_raw_output = ""
        self.last_answer = ""
        self.last_reasoning = ""
        self.last_latency_generation = 0.0
        self.last_image_paths_sent = [
            str(path) if path is not None else None for path in image_paths or []
        ]
        self.last_prompt_profile = "docbench_hybrid_v1"
        self.last_prompt_name = get_docbench_prompt_name(question_type, context_mode)

        if not query or not query.strip():
            self.last_answer = "NOT FOUND"
            return self.last_answer
        if not context_images and not context_text:
            self.last_answer = "NOT FOUND"
            return self.last_answer

        if context_images:
            prompt = DOCBENCH_MULTIMODAL_PROMPT_V2
            evidence_block = ""
            if context_text and context_text.strip():
                evidence_block = (
                    "\n\nAdditional text evidence extracted from the same selected pages:\n"
                    f"{context_text.strip()}\n"
                )
            text = (
                f"{prompt}\n\n"
                f"{evidence_block}"
                f"Question: {query}\n\n"
                "Return only:\n"
                "Answer: <one complete sentence>"
            )
        else:
            prompt = get_docbench_prompt(question_type, context_mode)
            text = (
                f"{prompt}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Return only:\n"
                "Answer: <one complete sentence>"
            )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": text,
            }
        ]
        for image in context_images or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image)},
                }
            )

        self.last_num_images_sent = len(content) - 1
        if not self.last_image_paths_sent:
            self.last_image_paths_sent = [None] * self.last_num_images_sent

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self.last_latency_generation = time.time() - start_time
            raw_answer = response.choices[0].message.content or ""
            self.last_raw_output = raw_answer
            self.last_answer = self._postprocess_answer(raw_answer)
            print(f"    [OpenAI VLM] Generation time: {self.last_latency_generation:.2f}s")
            return self.last_answer
        except Exception as exc:  # noqa: BLE001 - keep eval running and persist the failure.
            self.last_latency_generation = time.time() - start_time
            self.last_error = str(exc)
            self.last_raw_output = f"ERROR: {exc}"
            self.last_answer = "ERROR"
            print(f"    [OpenAI VLM ERROR] {exc}")
            return self.last_answer

    def _image_to_data_url(self, image: Image.Image | str | Path) -> str:
        if isinstance(image, str | Path):
            path = Path(image)
            suffix = path.suffix.lower()
            mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _postprocess_answer(raw_answer: str) -> str:
        answer = (raw_answer or "").strip()
        if not answer:
            return "NOT FOUND"
        if "<ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        elif "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        answer = re.sub(r"^(assistant|user)\s*:?\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = answer.strip('"').strip("'")
        return answer or "NOT FOUND"


def create_openai_compatible_vlm(**kwargs: Any) -> OpenAICompatibleVLM:
    return OpenAICompatibleVLM(**kwargs)
