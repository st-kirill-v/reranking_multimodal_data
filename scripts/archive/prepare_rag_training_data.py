import json
import sys
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

# Добавляем корень проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импортируем те же функции, что в инференсе
from scripts.full_pipeline import full_search, expand_with_neighbors
from scripts.evaluate_rag import load_all_questions

# Пути
output_dir = project_root / "data" / "rag_train"
output_dir.mkdir(exist_ok=True)

print("Preparing rag training data (with real search pipeline)")

# Загружаем все вопросы
print("\nLoading all questions...")
all_questions = load_all_questions()
print(f"Loaded {len(all_questions)} questions")

# Разделяем на train/val/test по папкам
questions_by_folder = defaultdict(list)
for q in all_questions:
    questions_by_folder[q["folder"]].append(q)

random.seed(42)
folders = list(questions_by_folder.keys())
random.shuffle(folders)

n_folders = len(folders)
n_train = int(n_folders * 0.7)
n_val = int(n_folders * 0.15)

train_folders = folders[:n_train]
val_folders = folders[n_train : n_train + n_val]
test_folders = folders[n_train + n_val :]

train_questions = []
for folder in train_folders:
    train_questions.extend(questions_by_folder[folder])

val_questions = []
for folder in val_folders:
    val_questions.extend(questions_by_folder[folder])

test_questions = []
for folder in test_folders:
    test_questions.extend(questions_by_folder[folder])

print(f"\nSplits:")
print(f"  Train: {len(train_questions)} questions ({len(train_folders)} folders)")
print(f"  Val: {len(val_questions)} questions ({len(val_folders)} folders)")
print(f"  Test: {len(test_questions)} questions ({len(test_folders)} folders)")


# Функция сбора контекста
def collect_context_for_questions(questions, split_name):
    """Для каждого вопроса запускает поиск и расширяет соседними страницами"""
    data = []

    print(f"\nProcessing {split_name} split...")
    print(f"   Pipeline: BM25 + Qwen3-VL-Embedding + RRF + Nemotron Rerank + neighbors")

    for i, q in enumerate(tqdm(questions, desc=f"  {split_name}")):
        try:
            # Поиск (как в evaluate_rag.py)
            candidates, search_time = full_search(
                q["question"], top_k_initial=400, top_k_rerank=150, final_k=30
            )

            # Расширяем соседними страницами (как в full_pipeline.py)
            candidates_with_neighbors = expand_with_neighbors(candidates, max_pages=15)

            # Сохраняем пути к страницам
            page_paths = []
            for rank, cand in enumerate(candidates_with_neighbors[:15]):
                page_paths.append(
                    {
                        "folder": cand["folder"],
                        "page": cand["page"],
                        "path": cand["path"],
                        "rerank_score": cand.get("rerank_score", 0),
                        "context_rank": rank + 1,
                    }
                )

            data.append(
                {
                    "question": q["question"],
                    "expected_answer": q["expected_answer"],
                    "type": q["type"],
                    "folder": q["folder"],
                    "page_paths": page_paths,
                    "num_pages": len(page_paths),
                    "search_time": search_time,
                }
            )

        except Exception as e:
            print(f"\n  Error: {q['question'][:50]}... | {e}")
            continue

    # Сохраняем
    output_file = output_dir / f"{split_name}_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if data:
        avg_pages = sum(d["num_pages"] for d in data) / len(data)
        avg_time = sum(d["search_time"] for d in data) / len(data)
        print(
            f"\n  {split_name}: {len(data)} examples, avg_pages={avg_pages:.1f}, avg_time={avg_time:.2f}s"
        )

    return data


# Запускаем
print("\nRunning full_search for all questions...")
print("   Estimated: 2-3 hours\n")

train_data = collect_context_for_questions(train_questions, "train")
val_data = collect_context_for_questions(val_questions, "val")
test_data = collect_context_for_questions(test_questions, "test")

# Статистика
stats = {
    "train": {"count": len(train_data), "folders": len(train_folders)},
    "val": {"count": len(val_data), "folders": len(val_folders)},
    "test": {"count": len(test_data), "folders": len(test_folders)},
    "pipeline": {
        "search": "BM25 + Qwen3-VL-Embedding + RRF + Nemotron Rerank",
        "neighbors": "expand_with_neighbors(max_pages=15)",
        "top_k_initial": 400,
        "top_k_rerank": 150,
        "final_k": 30,
    },
}

with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("\nRag training data preparation completed!")
print(f"Data saved to: {output_dir}")
