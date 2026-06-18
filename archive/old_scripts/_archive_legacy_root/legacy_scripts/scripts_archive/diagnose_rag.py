import sys
from pathlib import Path
import re
import json
import time
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.core.generators.qwen_vl_generator import create_qwen_generator

qwen = create_qwen_generator(device="cuda" if torch.cuda.is_available() else "cpu")

import torch

from scripts.full_pipeline import full_search

data_path = project_root / "data" / "datasets" / "docbench"

problem_queries = [
    {
        "question": "How many tokens are in the training set of the Linked WikiText-2 Corpus?",
        "folder": "0",
        "page": 5,
        "expected_answer": "2019195",
    },
    {
        "question": "How many documents are there in the training set of the Linked WikiText-2 Corpus?",
        "folder": "0",
        "page": 5,
        "expected_answer": "600",
    },
    {
        "question": "What is the total number of sentences in the training sets for Romance languages as given in Table 1?",
        "folder": "0",
        "page": None,
        "expected_answer": "61802",
    },
    {
        "question": "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
        "folder": "0",
        "page": None,
        "expected_answer": "67.5",
    },
    {
        "question": "What function is used to determine a probability distribution over the two warrants in the proposed architecture?",
        "folder": "0",
        "page": None,
        "expected_answer": "softmax",
    },
]


def get_page_image(folder, page_num):
    from PIL import Image

    img_path = data_path / folder / "extracted" / "pages" / f"page_{page_num}.png"
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


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


def diagnose():
    print("Rag diagnostic - analysis of problematic questions (Qwen2-VL)")

    for i, q in enumerate(problem_queries, 1):
        print(f"\n[{i}] {q['question'][:100]}...")
        print(f"    Expected answer: {q['expected_answer']}")
        print(f"    Expected folder: {q['folder']}, page: {q['page']}")

        print("\nSearch phase:")
        start_time = time.time()
        try:
            candidates, duration = full_search(
                q["question"], top_k_initial=400, top_k_rerank=150, final_k=30
            )

            print(f"   Search time: {duration:.2f}s")
            print(f"   Found {len(candidates)} candidates")

            correct_found = False
            correct_rank = None
            correct_score = None
            correct_page_text = None
            correct_page_image = None

            for rank, cand in enumerate(candidates[:20]):
                if cand["folder"] == q["folder"]:
                    if q["page"] is None or cand["page"] == q["page"]:
                        correct_found = True
                        correct_rank = rank + 1
                        correct_score = cand.get("rerank_score", cand.get("score", 0))
                        correct_page_text = get_page_text(cand["folder"], cand["page"])
                        correct_page_image = get_page_image(cand["folder"], cand["page"])
                        break

            if correct_found:
                print(f"   Correct page found at rank {correct_rank} (score: {correct_score:.4f})")
            else:
                print(f"   Correct page not in top-20")
                print(f"\n   Top-5 candidates:")
                for rank, cand in enumerate(candidates[:5]):
                    print(
                        f"      {rank+1}. Doc {cand['folder']}, Page {cand['page']}, Score: {cand.get('rerank_score', cand.get('score', 0)):.4f}"
                    )

        except Exception as e:
            print(f"   Search error: {e}")
            continue

        print("\nPage text analysis:")
        if correct_found and correct_page_text:
            has_answer = q["expected_answer"].lower() in correct_page_text.lower()
            if has_answer:
                print(f"   Answer found in page text!")
            else:
                print(f"   Answer not found in page text")

        if correct_found and correct_page_image:
            print("\nQwen2-VL generation:")
            try:
                answer = qwen.generate_answer(q["question"], [correct_page_image])
                print(f"   Generated answer: {answer}")

                if str(q["expected_answer"]).lower() in str(answer).lower():
                    print(f"   Qwen found correct answer!")
                else:
                    print(f"   Qwen did not find correct answer")
                    print(f"   Expected: {q['expected_answer']}")

            except Exception as e:
                print(f"   Generation error: {e}")

        if i < len(problem_queries):
            input("\nPress Enter for next question...")


if __name__ == "__main__":
    diagnose()
