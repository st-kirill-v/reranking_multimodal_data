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

from scripts.full_pipeline import full_search, expand_with_neighbors
from src.core.generators.qwen_vl_generator import create_table_generator
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Qwen3-VL...")
qwen = create_table_generator(device=device)

data_path = project_root / "data" / "datasets" / "docbench"


def load_problematic_questions():
    problem_list = [
        {
            "question": "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
            "expected_answer": "The top-1 accuracy of the Oracle KGLM on birthdate prediction is 65%.",
            "folder": "1",
            "type": "multimodal-t",
        },
        {
            "question": "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
            "expected_answer": "67.5",
            "folder": "11",
            "type": "multimodal-t",
        },
        {
            "question": "What is the test set accuracy of BERT (Large) as reported in the best run according to Table 1?",
            "expected_answer": "The test set accuracy of BERT (Large) in its best run is 77%.",
            "folder": "2",
            "type": "multimodal-t",
        },
        {
            "question": "How does the performance of BERT+DL on SST-5 compare to that of BERT+DSC?",
            "expected_answer": "BERT+DL performs worse on SST-5 with an accuracy of 54.63 compared to BERT+DSC which has an accuracy of 55.19.",
            "folder": "2",
            "type": "multimodal-t",
        },
        {
            "question": "By how much does the accuracy of BERT+CE on SST-2 exceed that of BERT+DL?",
            "expected_answer": "The accuracy of BERT+CE on SST-2 exceeds that of BERT+DL by 0.53.",
            "folder": "2",
            "type": "multimodal-t",
        },
        {
            "question": "How much does SenseBERT_BASE improve upon BERT_BASE in the SemEval-SS Frozen task?",
            "expected_answer": "SenseBERT_BASE improves by 10.5 points over BERT_BASE (75.6 - 65.1 = 10.5).",
            "folder": "3",
            "type": "multimodal-t",
        },
        {
            "question": "Which model achieved the highest accuracy on the random split according to Table 5?",
            "expected_answer": "BERT-LARGE achieved the highest accuracy on the random split with 55.9%.",
            "folder": "4",
            "type": "multimodal-t",
        },
        {
            "question": "Which CONCEPTNET relation has the highest frequency in COMMONSENSEQA according to Table 2?",
            "expected_answer": "The AtLocation relation.",
            "folder": "4",
            "type": "multimodal-t",
        },
        {
            "question": "What percentage of questions analyzed begin with a WH word according to Figure 4?",
            "expected_answer": "44% of the questions begin with a WH word.",
            "folder": "4",
            "type": "multimodal-t",
        },
        {
            "question": "What is the percentage of unverified claims out of the total claims for the SE dataset?",
            "expected_answer": "The percentage of unverified claims out of the total claims for the SE dataset is approximately 34.93%.",
            "folder": "5",
            "type": "multimodal-t",
        },
        {
            "question": "In the picture at the top of the newspaper, what are the soldiers standing on?",
            "expected_answer": "A tunk",
            "folder": "6",
            "type": "multimodal-t",
        },
        {
            "question": "How many soldiers are shown in the photo of the Ukrainian troops?",
            "expected_answer": "5",
            "folder": "6",
            "type": "multimodal-t",
        },
        {
            "question": "What category has the highest count in the CHAI corpus according to Table 2?",
            "expected_answer": "Temporal coordination of sub-goals has the highest count in the CHAI corpus with 68 examples.",
            "folder": "7",
            "type": "multimodal-t",
        },
        {
            "question": "What is the CO2 equivalent emission for training a Transformer (big) NLP model on a GPU?",
            "expected_answer": "The CO2 equivalent emission for training a Transformer (big) NLP model on a GPU is 192 lbs.",
            "folder": "8",
            "type": "multimodal-t",
        },
        {
            "question": "How much faster is the SWEM model compared to the LSTM model in terms of training speed?",
            "expected_answer": "The SWEM model is approximately 9.5 times faster than the LSTM model in terms of training speed.",
            "folder": "8",
            "type": "multimodal-t",
        },
        {
            "question": "What model achieved the highest accuracy on the Yelp Polarity sentiment analysis task according to Table 2?",
            "expected_answer": "SWEM-hier achieved the highest accuracy on the Yelp Polarity sentiment analysis task with an accuracy of 95.81%.",
            "folder": "8",
            "type": "multimodal-t",
        },
        {
            "question": "How much does the test accuracy decrease when using a shuffled training set as opposed to the original training set on the Yelp polarity dataset?",
            "expected_answer": "1.62%",
            "folder": "8",
            "type": "multimodal-t",
        },
        {
            "question": "Which model achieved the highest accuracy on the SST-2 dataset?",
            "expected_answer": "Constituency Tree-LSTM with an accuracy of 88.0%.",
            "folder": "8",
            "type": "multimodal-t",
        },
        {
            "question": "What method achieved the highest Macro-F1 score on dataset D1?",
            "expected_answer": "Ours: PRET+MULT with a Macro-F1 score of 69.73.",
            "folder": "8",
            "type": "multimodal-t",
        },
    ]
    return problem_list


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


def expand_with_neighbors(candidates, max_pages=15):
    expanded = []
    for cand in candidates:
        expanded.append(cand)
        if cand["page"] > 1:
            prev_cand = cand.copy()
            prev_cand["page"] = cand["page"] - 1
            prev_cand["path"] = str(Path(cand["path"]).parent / f"page_{prev_cand['page']}.png")
            expanded.append(prev_cand)
        next_cand = cand.copy()
        next_cand["page"] = cand["page"] + 1
        next_cand["path"] = str(Path(cand["path"]).parent / f"page_{next_cand['page']}.png")
        expanded.append(next_cand)

    unique = {}
    for cand in expanded:
        key = f"{cand['folder']}_{cand['page']}"
        if key not in unique:
            unique[key] = cand
    return list(unique.values())[:max_pages]


def evaluate_problematic_v1():
    print("=" * 60)
    print("Evaluation on problematic questions with V1 pipeline (pages only)")
    print("=" * 60)

    questions = load_problematic_questions()
    print(f"\nLoaded {len(questions)} problematic questions")

    results = []
    latencies = []

    for i, q in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] {q['question'][:80]}...")

        start_time = time.time()

        try:
            candidates, search_time = full_search(
                q["question"], top_k_initial=400, top_k_rerank=150, final_k=20
            )
        except Exception as e:
            print(f"  Search error: {e}")
            continue

        candidates_with_neighbors = expand_with_neighbors(candidates, max_pages=15)

        context_images = []
        for cand in candidates_with_neighbors[:15]:
            img = get_page_image(cand["folder"], cand["page"])
            if img:
                context_images.append(img)

        if not context_images:
            print("  No images found for context.")
            answer = "NOT FOUND"
        else:
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

        print(f"  Generated: {answer[:150]}")
        print(f"  Expected: {q['expected_answer'][:150]}")
        print(f"  Exact: {exact_match}, F1: {f1_score:.3f} | Time: {elapsed_time:.2f}s")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY V1 PIPELINE (pages only)")
    print("=" * 60)

    f1_scores = [r["f1"] for r in results]
    exact_matches = [r["exact"] for r in results]

    print(f"\nTotal questions: {len(results)}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Exact Match: {np.mean(exact_matches):.4f}")
    print(f"F1 > 0.5: {np.mean([1 if f > 0.5 else 0 for f in f1_scores]):.4f}")
    print(f"F1 = 0: {sum(1 for f in f1_scores if f == 0)}")

    output_file = project_root / "data" / "problematic_results_v1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    evaluate_problematic_v1()
