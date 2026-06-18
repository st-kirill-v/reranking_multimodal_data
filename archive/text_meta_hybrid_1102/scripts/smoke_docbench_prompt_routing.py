from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts.docbench_hybrid_prompts import get_docbench_prompt_name


SMOKE_CASES = [
    (
        "text-only",
        "What is the primary challenge addressed by the introduction of the Linked WikiText-2 dataset?",
        "DOCBENCH_TEXT_ONLY_PROMPT",
    ),
    (
        "meta-data",
        "Who is the last author of the paper?",
        "DOCBENCH_METADATA_PROMPT",
    ),
    (
        "multimodal-t",
        "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
        "DOCBENCH_MULTIMODAL_PROMPT_V2",
    ),
]


def main() -> None:
    for question_type, question, expected_prompt in SMOKE_CASES:
        prompt_name = get_docbench_prompt_name(question_type)
        print(f"{question_type}: {prompt_name} | {question}")
        if prompt_name != expected_prompt:
            raise SystemExit(f"Expected {expected_prompt} for {question_type}, got {prompt_name}")
    print("Prompt routing smoke test passed.")


if __name__ == "__main__":
    main()
