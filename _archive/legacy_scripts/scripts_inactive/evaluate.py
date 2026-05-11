import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from sklearn.metrics import ndcg_score
import time

data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"

from scripts.full_pipeline import full_search

test_questions = []
with open(data_path / "0" / "0_qa.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        if q.get("type") in ["multimodal-t", "multimodal-f"]:
            page = None
            question_text = q["question"]
            if "top-1 accuracy of the Oracle KGLM on birthdate prediction" in question_text:
                page = 7
            elif (
                "documents are there in the training set of the Linked WikiText-2 Corpus"
                in question_text
            ):
                page = 5
            elif "lowest Perplexity (PPL) according to Table 3" in question_text:
                page = 7

            if page is not None:
                test_questions.append(
                    {"question": q["question"], "folder": "0", "page": page, "answer": q["answer"]}
                )

print(f"Loaded {len(test_questions)} test questions with ground truth pages")


def compute_metrics(results, ground_truth, k_values=[5, 10, 20]):
    metrics = {}
    for k in k_values:
        retrieved = [
            r["folder"] == ground_truth["folder"] and r["page"] == ground_truth["page"]
            for r in results[:k]
        ]
        metrics[f"recall@{k}"] = float(any(retrieved))

        true_rel = np.zeros(len(results[:k]))
        if any(retrieved):
            true_rel[retrieved.index(True)] = 1
        pred_scores = [r.get("rerank_score", r.get("score", 0)) for r in results[:k]]
        if len(pred_scores) > 1:
            metrics[f"ndcg@{k}"] = ndcg_score([true_rel], [pred_scores])
        else:
            metrics[f"ndcg@{k}"] = float(true_rel[0])

    mrr = 0
    for i, r in enumerate(results):
        if r["folder"] == ground_truth["folder"] and r["page"] == ground_truth["page"]:
            mrr = 1 / (i + 1)
            break
    metrics["mrr"] = mrr
    return metrics


latencies = []
all_metrics = []

for q in test_questions:
    print(f"\nProcessing: {q['question'][:80]}...")
    try:
        results, duration = full_search(q["question"])
        print(f"Results: {[f'{r['folder']}_{r['page']}' for r in results[:5]]}")
        latencies.append(duration)
        metrics = compute_metrics(results, {"folder": q["folder"], "page": q["page"]})
        all_metrics.append(metrics)
    except Exception as e:
        print(f"Error: {e}")
        continue

if all_metrics:
    print("\nEvaluation results")
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\nLatency p50: {np.percentile(latencies, 50):.2f}s")
    print(f"Latency p95: {np.percentile(latencies, 95):.2f}s")
    print(f"Average latency: {np.mean(latencies):.2f}s")
else:
    print("No queries processed successfully")
