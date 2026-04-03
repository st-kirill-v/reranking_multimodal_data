import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import torch
from PIL import Image

from scripts.full_pipeline import (
    full_search,
    bm25,
    embed_model,
    index,
    metadata,
    rerank_model,
    rerank_processor,
    device,
    data_path,
)
from src.core.generators.yandex_gpt_generator import YandexGPTRAGGenerator

llm = YandexGPTRAGGenerator()

problem_queries = [
    {
        "question": "What is the total number of sentences in the training sets for Romance languages as given in Table 1?",
        "expected_answer": "61802",
        "expected_numbers": ["10017", "51785", "61802"],
    },
    {
        "question": "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
        "expected_answer": "67.5",
    },
    {
        "question": "What function is used to determine a probability distribution over the two warrants in the proposed architecture?",
        "expected_answer": "softmax",
    },
]


def get_page_text(folder, page_num):
    text_path = data_path / folder / "extracted" / "pages_text.json"
    if not text_path.exists():
        return ""
    with open(text_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    for page in pages:
        if page["page"] == page_num:
            return page["text"]
    return ""


def check_answer_in_text(text, expected_answer, expected_numbers=None):
    if not text:
        return False
    text_lower = text.lower()
    expected_lower = expected_answer.lower()

    if expected_lower in text_lower:
        return True

    if expected_numbers:
        for num in expected_numbers:
            if num in text_lower:
                return True
    return False


def diagnose():

    print("Diagnostic search (all folders)")

    for i, q in enumerate(problem_queries, 1):
        print(f"\n{'='*100}")
        print(f"[{i}] {q['question'][:100]}...")
        print(f"    Expected answer: {q['expected_answer']}")

        print("\nSearch phase")

        start_time = time.time()

        try:
            candidates, duration = full_search(
                q["question"], top_k_initial=400, top_k_rerank=150, final_k=30, use_expansion=False
            )
            print(f"   Search time: {duration:.2f}s")
            print(f"   Found {len(candidates)} candidates")

            print(f"\n   Top-10 after rerank:")
            for rank, cand in enumerate(candidates[:10]):
                print(
                    f"      {rank+1}. Doc {cand['folder']}, Page {cand['page']}, Score: {cand['rerank_score']:.4f}"
                )

            print("\nSearching for answer in top-30")
            print("-" * 50)

            answer_found = False
            answer_page = None
            answer_text = None

            for rank, cand in enumerate(candidates[:30]):
                page_text = get_page_text(cand["folder"], cand["page"])
                if page_text:
                    expected_numbers = q.get("expected_numbers", [])
                    if check_answer_in_text(page_text, q["expected_answer"], expected_numbers):
                        answer_found = True
                        answer_page = f"Doc {cand['folder']}, Page {cand['page']}"
                        answer_text = page_text[:500]
                        print(f"   Answer found at rank {rank+1}: {answer_page}")
                        break

            if not answer_found:
                print(f"   Answer not found in top-30")
                print(f"\n   Top-5 candidates:")
                for rank, cand in enumerate(candidates[:5]):
                    print(f"      {rank+1}. Doc {cand['folder']}, Page {cand['page']}")

            print("\nLLM generation")

            if answer_found and answer_text:
                try:
                    context_docs = [{"content": answer_text[:4000]}]
                    llm_answer = llm.generate_answer(q["question"], context_docs)
                    print(f"   LLM answer: {llm_answer[:150]}...")

                    if q["expected_answer"].lower() in llm_answer.lower():
                        print(f"   LLM extracted correct answer!")
                    else:
                        print(f"   LLM did not extract correct answer")
                        print(f"   Expected: {q['expected_answer']}")
                        print(f"   Got: {llm_answer[:100]}...")
                except Exception as e:
                    print(f"   LLM error: {e}")
            else:
                print(f"   Cannot generate (answer page not found)")

            print("\n" + "-" * 50)
            print("Diagnosis summary")
            if answer_found:
                print(f"   Search success - answer page found at {answer_page}")
                if llm_answer and q["expected_answer"].lower() in llm_answer.lower():
                    print(f"   LLM success - answer extracted correctly")
                else:
                    print(f"   LLM failed - answer in page but not extracted")
            else:
                print(f"   Search failed - answer not found in top-30")

        except Exception as e:
            print(f"   Error: {e}")
            continue

        if i < len(problem_queries):
            input("\nPress Enter for next question...")


if __name__ == "__main__":
    diagnose()
