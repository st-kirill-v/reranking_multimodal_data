import sys
from pathlib import Path
import json
import time
import numpy as np
from collections import defaultdict
import re
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.full_pipeline import full_search
from src.core.generators.qwen_vl_generator import create_table_generator
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Qwen3-VL...")
qwen = create_table_generator(device=device)

data_path = project_root / "data" / "datasets" / "docbench"


def load_all_questions():
    questions = []
    for jsonl_file in sorted(data_path.glob("*/*_qa.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    q = json.loads(line)
                    if q.get("type") in ["multimodal-t", "multimodal-f"]:
                        questions.append(
                            {
                                "question": q["question"],
                                "expected_answer": q["answer"],
                                "folder": jsonl_file.parent.name,
                                "type": q["type"],
                            }
                        )
                except json.JSONDecodeError:
                    continue
    return questions


def get_page_image(folder, page_num):
    img_path = data_path / folder / "extracted" / "pages" / f"page_{page_num}.png"
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


def generate_answer_with_context(question, context_images):
    if not context_images:
        return "NOT FOUND"

    try:
        answer = qwen.generate_answer(question, context_images)
        return answer.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def compute_similarity(generated, expected):
    if generated in ["NOT FOUND", "ERROR"]:
        return 0.0, 0.0

    gen = generated.lower().strip()
    exp = expected.lower().strip()

    exact = (
        1.0
        if gen.replace(",", "").replace(" ", "") == exp.replace(",", "").replace(" ", "")
        else 0.0
    )

    gen_clean = gen.replace(",", "").replace(" ", "")
    exp_clean = exp.replace(",", "").replace(" ", "")

    gen_numbers = re.findall(r"\d+(?:\.\d+)?", gen_clean)
    exp_numbers = re.findall(r"\d+(?:\.\d+)?", exp_clean)

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "has",
        "have",
        "had",
        "does",
        "do",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "can",
    }

    gen_words = set(w for w in re.findall(r"[a-z]+", gen) if w not in stop_words and len(w) > 2)
    exp_words = set(w for w in re.findall(r"[a-z]+", exp) if w not in stop_words and len(w) > 2)

    word_intersection = gen_words & exp_words
    word_precision = len(word_intersection) / len(gen_words) if gen_words else 0
    word_recall = len(word_intersection) / len(exp_words) if exp_words else 0
    word_f1 = (
        2 * word_precision * word_recall / (word_precision + word_recall)
        if (word_precision + word_recall) > 0
        else 0
    )

    num_intersection = set(gen_numbers) & set(exp_numbers)
    num_precision = len(num_intersection) / len(gen_numbers) if gen_numbers else 0
    num_recall = len(num_intersection) / len(exp_numbers) if exp_numbers else 0
    num_f1 = (
        2 * num_precision * num_recall / (num_precision + num_recall)
        if (num_precision + num_recall) > 0
        else 0
    )

    f1 = word_f1 * 0.6 + num_f1 * 0.4
    return exact, f1


def evaluate_rag(questions, limit=308):
    results = []
    latencies = []

    total_questions = min(len(questions), limit)

    for i, q in enumerate(questions[:total_questions], start=1):
        print(f"\n[{i}/{total_questions}] {q['question'][:80]}...")

        start_time = time.time()

        try:
            candidates, search_time = full_search(
                q["question"], top_k_initial=400, top_k_rerank=150, final_k=20
            )
        except Exception as e:
            print(f"  Search error: {e}")
            continue

        context_images = []
        for cand in candidates[:20]:
            img = get_page_image(cand["folder"], cand["page"])
            if img:
                context_images.append(img)

        if not context_images:
            print("  No images found for context.")

        answer = generate_answer_with_context(q["question"], context_images)

        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

        exact_match, f1_score = compute_similarity(answer, q["expected_answer"])

        results.append(
            {
                "question": q["question"],
                "expected": q["expected_answer"],
                "generated": answer,
                "exact": exact_match,
                "f1": f1_score,
                "latency": elapsed_time,
                "folder": q["folder"],
                "type": q["type"],
            }
        )

        print(f"  Generated: {answer[:200]}...")
        print(f"  Expected: {q['expected_answer']}")
        print(f"  Exact: {exact_match}, F1: {f1_score:.3f} | Time: {elapsed_time:.2f}s")

    return results, latencies


def print_metrics(results, latencies):
    if not results:
        print("No results to evaluate.")
        return

    exact_matches = [r["exact"] for r in results]
    f1_scores = [r["f1"] for r in results]

    print("\nFinal evaluation results Qwen3")

    print(f"\nTotal questions evaluated: {len(results)}")

    print(f"\nAccuracy metrics:")
    print(f"  Exact match (em): {np.mean(exact_matches):.4f}")
    print(f"  Mean f1-score: {np.mean(f1_scores):.4f}")

    accuracy_f1_50 = np.mean([1 if f > 0.5 else 0 for f in f1_scores])
    print(f"  Accuracy (f1 > 0.5): {accuracy_f1_50:.4f}")

    print(f"\nLatency metrics (per question):")
    print(f"  Average: {np.mean(latencies):.2f}s")
    print(f"  Median (p50): {np.percentile(latencies, 50):.2f}s")
    print(f"  p95 (95th percentile): {np.percentile(latencies, 95):.2f}s")

    print("\nMetrics by question type:")
    type_groups = defaultdict(list)
    for r in results:
        type_groups[r["type"]].append(r["f1"])

    for q_type, scores in type_groups.items():
        print(f"  {q_type}: f1={np.mean(scores):.3f} (n={len(scores)})")


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":

    print("Starting rag evaluation with table extractor")

    print("\nLoading questions from dataset...")
    all_questions = load_all_questions()

    if not all_questions:
        print("No questions found in the dataset.")
        sys.exit(1)

    print(f"Loaded {len(all_questions)} multimodal questions.")

    print("\nStarting evaluation pipeline...")

    try:
        eval_results, eval_latencies = evaluate_rag(all_questions)

        print_metrics(eval_results, eval_latencies)

        output_file = project_root / "data" / "evaluation_results_table_extractor.json"
        save_results(eval_results, output_file)

        print(f"\nResults saved to {output_file}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
