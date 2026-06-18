#!/usr/bin/env python3
"""
Диагностический скрипт для анализа контекста, передаваемого в VLM.

Запускает full_pipeline_v3 на 30 вопросах (разные типы и домены)
и логирует ВСЁ, что передаётся в VLM:
  - количество и размеры изображений (страницы PNG + вырезанные изображения)
  - полный текст Markdown таблиц (БЕЗ сокращений)
  - какие страницы были выбраны (номера)
  - какой домен определён
  - какие лимиты контекста применены
  - итоговый ответ модели
  - F1 и Exact Match метрики

Вопросы взяты из реальных тестов evaluate_rag_v2.py
Использование:
    python scripts/diagnose_context.py
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.full_pipeline_v3 import full_pipeline_v3, normalize_answer
from src.core.generators.qwen_vl_generator import create_table_generator

# ============================================================
# ТЕСТОВЫЕ ВОПРОСЫ (30 штук, разные типы и домены)
# С ожидаемыми ответами из реальных тестов evaluate_rag_v2.py
# ============================================================

TEST_QUERIES = [
    # ========== ACADEMIC / TABLE ==========
    {
        "id": 1,
        "name": "Academic: KGLM accuracy",
        "query": "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
        "expected": "65%",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 2,
        "name": "Academic: Linked WikiText-2 documents",
        "query": "How many documents are there in the training set of the Linked WikiText-2 Corpus?",
        "expected": "600",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 3,
        "name": "Academic: Lowest PPL model",
        "query": "Which language model has the lowest Perplexity (PPL) according to Table 3?",
        "expected": "KGLM",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 4,
        "name": "Academic: Largest BLEU decrease",
        "query": "Which dataset experienced the largest decrease in BLEU score after alignment according to Table 4?",
        "expected": "GL→EN",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 5,
        "name": "Academic: Romance languages total",
        "query": "What is the total number of sentences in the training sets for Romance languages as given in Table 1?",
        "expected": "61802",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 6,
        "name": "Academic: Entity Recognition + Coref",
        "query": "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
        "expected": "67.5",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 7,
        "name": "Academic: Highest F1 entity recognition",
        "query": "Which model has the highest F1 score for entity recognition on the Test set?",
        "expected": "SCIIE",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 8,
        "name": "Academic: Highest F1 span identification",
        "query": "Which model achieved the highest F1 score in span identification?",
        "expected": "SciIE",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 9,
        "name": "Academic: BERT test accuracy",
        "query": "What is the test set accuracy of BERT (Large) as reported in the best run according to Table 1?",
        "expected": "77%",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 10,
        "name": "Academic: Softmax function",
        "query": "What function is used to determine a probability distribution over the two warrants in the proposed architecture?",
        "expected": "softmax",
        "type": "formula",
        "domain": "academic",
    },
    {
        "id": 11,
        "name": "Academic: Highest improvement QuoRef",
        "query": "Which model variant has the highest improvement in F1 score for the QuoRef dataset when compared to the base XLNet model?",
        "expected": "XLNet+DSC",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 12,
        "name": "Academic: Best augmentation method",
        "query": "What data augmentation method resulted in the highest F1-score for the BERT model according to Table 8?",
        "expected": "+ positive & negative",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 13,
        "name": "Academic: Highest F1 Chinese OntoNotes",
        "query": "What is the highest F1 score achieved on the Chinese OntoNotes4.0 dataset, according to Table 10?",
        "expected": "84.67",
        "type": "table",
        "domain": "academic",
    },
    # ========== FINANCIAL / TABLE ==========
    {
        "id": 14,
        "name": "Financial: Total revenue 2020",
        "query": "What was the total revenue for UnitedHealth Group in 2020?",
        "expected": "$242,155 million",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 15,
        "name": "Financial: Net income change",
        "query": "By how much did the company's net income change from 2018 to 2020?",
        "expected": "$1,178 million",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 16,
        "name": "Financial: Employee count",
        "query": "Which region has the highest number of employees of the company and how many?",
        "expected": "Europe with 43,181",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 17,
        "name": "Financial: Total revenue 2019",
        "query": "What was the total revenue for the company in 2019?",
        "expected": "$21.365 billion",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 18,
        "name": "Financial: Operating profit 2020",
        "query": "What was the operating profit for the full year 2020?",
        "expected": "$4,553 million",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 19,
        "name": "Financial: Gross margin 2020",
        "query": "What was the gross margin percentage for the year 2020?",
        "expected": "11.20%",
        "type": "table",
        "domain": "financial",
    },
    {
        "id": 20,
        "name": "Financial: Total employees 2020",
        "query": "What was the total number of full-time employees in 2020 for the company?",
        "expected": "44,723",
        "type": "table",
        "domain": "financial",
    },
    # ========== GOVERNMENT / TEXT ==========
    {
        "id": 21,
        "name": "Government: Policy focus",
        "query": "What is the primary focus of Bureau Objective 3.4?",
        "expected": "The report does not contain such objective",
        "type": "text",
        "domain": "government",
    },
    # ========== LEGAL / TEXT ==========
    {
        "id": 22,
        "name": "Legal: Mention count",
        "query": "How many times does the report mention 'scientific ethics'?",
        "expected": "11 times",
        "type": "text",
        "domain": "legal",
    },
    # ========== NEWS / IMAGE ==========
    {
        "id": 23,
        "name": "News: Soldiers on tank",
        "query": "In the picture at the top of the newspaper, what are the soldiers standing on?",
        "expected": "A tunk",
        "type": "image",
        "domain": "news",
    },
    {
        "id": 24,
        "name": "News: Ukrainian soldiers count",
        "query": "How many soldiers are shown in the photo of the Ukrainian troops?",
        "expected": "5",
        "type": "image",
        "domain": "news",
    },
    # ========== ДОПОЛНИТЕЛЬНЫЕ ACADEMIC ==========
    {
        "id": 25,
        "name": "Academic: SWEM vs LSTM speed",
        "query": "How much faster is the SWEM model compared to the LSTM model in terms of training speed?",
        "expected": "9.5 times faster",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 26,
        "name": "Academic: BERT+DSC MRPC boost",
        "query": "What performance boost did BERT+DSC achieve for the MRPC?",
        "expected": "+0.92",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 27,
        "name": "Academic: BERT+CE vs BERT+DL",
        "query": "By how much does the accuracy of BERT+CE on SST-2 exceed that of BERT+DL?",
        "expected": "0.53",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 28,
        "name": "Academic: Quoref ratio",
        "query": "What is the ratio of negative to positive examples for the Quoref task?",
        "expected": "169",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 29,
        "name": "Academic: Total users",
        "query": "What is the total number of users included in the dataset?",
        "expected": "25000",
        "type": "table",
        "domain": "academic",
    },
    {
        "id": 30,
        "name": "Academic: Users not in training",
        "query": "How many users from the dataset were not included in the training set?",
        "expected": "2062",
        "type": "table",
        "domain": "academic",
    },
]


# ============================================================
# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК
# ============================================================


def compute_f1(generated: str, expected: str) -> float:
    """Simple token-level F1 score"""
    gen_tokens = generated.lower().split()
    exp_tokens = expected.lower().split()

    if not gen_tokens or not exp_tokens:
        return 0.0

    gen_counter = defaultdict(int)
    exp_counter = defaultdict(int)

    for token in gen_tokens:
        gen_counter[token] += 1
    for token in exp_tokens:
        exp_counter[token] += 1

    common = 0
    for token in gen_counter:
        common += min(gen_counter[token], exp_counter.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(gen_tokens)
    recall = common / len(exp_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_metrics(generated: str, expected: str) -> dict:
    """Вычисляет Exact Match и F1"""
    gen_norm = normalize_answer(generated)
    exp_norm = normalize_answer(expected)

    exact = 1.0 if gen_norm == exp_norm else 0.0

    # Для числовых ответов пробуем извлечь число
    gen_nums = re.findall(r"\d+(?:\.\d+)?", gen_norm)
    exp_nums = re.findall(r"\d+(?:\.\d+)?", exp_norm)

    if gen_nums and exp_nums and not exact:
        try:
            g = float(gen_nums[0])
            e = float(exp_nums[0])
            if abs(g - e) < 0.01:
                exact = 1.0
            elif abs(g * 100 - e) < 0.5 or abs(g - e * 100) < 0.5:
                exact = 1.0
        except ValueError:
            pass

    f1 = compute_f1(gen_norm, exp_norm)

    # Если числа совпадают, но F1 низкий из-за лишних слов
    if not exact and gen_nums and exp_nums:
        try:
            g = float(gen_nums[0])
            e = float(exp_nums[0])
            if abs(g - e) < 0.01 or abs(g * 100 - e) < 0.5:
                f1 = max(f1, 0.95)
        except ValueError:
            pass

    return {"exact": exact, "f1": f1}


def print_separator(title: str = None, char: str = "=", width: int = 100):
    print("\n" + char * width)
    if title:
        print(f" {title} ".center(width, char))
    else:
        print(char * width)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================


def main():
    print_separator("DIAGNOSTIC RUN: CONTEXT ANALYSIS + METRICS")
    print(f"Testing {len(TEST_QUERIES)} questions...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        print_separator(f"QUESTION {i}/{len(TEST_QUERIES)}: {test['name']}", char="-")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print(f"Type: {test['type']} | Domain: {test['domain']}")

        try:
            # Запускаем пайплайн
            answer, answer_norm, total_time = full_pipeline_v3(test["query"])

            # Вычисляем метрики
            metrics = compute_metrics(answer, test["expected"])

            results.append(
                {
                    "id": test["id"],
                    "name": test["name"],
                    "query": test["query"],
                    "expected": test["expected"],
                    "generated": answer,
                    "generated_normalized": answer_norm,
                    "type": test["type"],
                    "domain": test["domain"],
                    "exact": metrics["exact"],
                    "f1": metrics["f1"],
                    "total_time": total_time,
                    "error": None,
                }
            )

            print(f"\n✅ RESULT:")
            print(f"   Generated: {answer}")
            print(f"   Expected:  {test['expected']}")
            print(f"   Exact: {metrics['exact']} | F1: {metrics['f1']:.3f}")
            print(f"   Time: {total_time:.2f}s")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "id": test["id"],
                    "name": test["name"],
                    "query": test["query"],
                    "expected": test["expected"],
                    "error": str(e),
                }
            )

    # ============================================================
    # ИТОГОВАЯ СВОДКА С МЕТРИКАМИ
    # ============================================================

    print_separator("FINAL SUMMARY WITH METRICS")

    success_results = [r for r in results if r.get("error") is None]
    failed_results = [r for r in results if r.get("error") is not None]

    print(f"\n📊 TOTAL: {len(results)} questions")
    print(f"   ✅ Successful: {len(success_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")

    if success_results:
        exact_scores = [r["exact"] for r in success_results]
        f1_scores = [r["f1"] for r in success_results]
        times = [r["total_time"] for r in success_results]

        print(f"\n📈 OVERALL METRICS:")
        print(
            f"   Exact Match: {sum(exact_scores) / len(exact_scores):.2%} ({sum(exact_scores)}/{len(exact_scores)})"
        )
        print(f"   Mean F1: {sum(f1_scores) / len(f1_scores):.3f}")
        print(
            f"   Accuracy (F1 > 0.5): {sum(1 for f in f1_scores if f > 0.5) / len(f1_scores):.2%}"
        )
        print(f"   Avg Time: {sum(times) / len(times):.2f}s")
        print(f"   Median Time: {sorted(times)[len(times)//2]:.2f}s")

    # По типам вопросов
    print(f"\n📊 BY QUESTION TYPE:")
    by_type = defaultdict(list)
    for r in success_results:
        by_type[r["type"]].append(r)

    for t, items in by_type.items():
        avg_f1 = sum(i["f1"] for i in items) / len(items)
        exact_count = sum(i["exact"] for i in items)
        print(f"   {t:<10} | n={len(items):2d} | F1={avg_f1:.3f} | EM={exact_count/len(items):.2%}")

    # По доменам
    print(f"\n📊 BY DOMAIN:")
    by_domain = defaultdict(list)
    for r in success_results:
        by_domain[r["domain"]].append(r)

    for d, items in by_domain.items():
        avg_f1 = sum(i["f1"] for i in items) / len(items)
        exact_count = sum(i["exact"] for i in items)
        print(f"   {d:<12} | n={len(items):2d} | F1={avg_f1:.3f} | EM={exact_count/len(items):.2%}")

    # Детальные результаты по каждому вопросу
    print_separator("DETAILED RESULTS")
    print(
        f"\n{'ID':<4} | {'Type':<8} | {'Domain':<12} | {'Exact':<6} | {'F1':<6} | {'Time':<6} | {'Expected'}"
    )
    print("-" * 100)
    for r in success_results:
        status = "✅" if r["exact"] else "⚠️" if r["f1"] > 0.5 else "❌"
        print(
            f"{r['id']:<4} | {r['type']:<8} | {r['domain']:<12} | {r['exact']!s:<6} | {r['f1']:.3f} | {r['total_time']:.1f}s | {r['expected'][:30]}"
        )

    if failed_results:
        print(f"\n❌ FAILED QUESTIONS:")
        for r in failed_results:
            print(f"   ID {r['id']}: {r['name']} - {r.get('error', 'Unknown error')}")

    # Сохраняем результаты
    output_file = Path(__file__).parent.parent / "data" / "diagnostic_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic script for VLM context analysis")
    parser.add_argument("--questions", type=int, nargs="+", help="Specific question IDs to test")
    parser.add_argument(
        "--type", type=str, help="Filter by question type (table/image/formula/text)"
    )
    parser.add_argument(
        "--domain", type=str, help="Filter by domain (academic/financial/government/legal/news)"
    )

    args = parser.parse_args()

    # Фильтрация вопросов
    test_queries = TEST_QUERIES.copy()
    if args.questions:
        test_queries = [q for q in test_queries if q["id"] in args.questions]
    if args.type:
        test_queries = [q for q in test_queries if q["type"] == args.type]
    if args.domain:
        test_queries = [q for q in test_queries if q["domain"] == args.domain]

    print(f"Filtered: {len(test_queries)} questions (from {len(TEST_QUERIES)} total)")

    # Переопределяем TEST_QUERIES
    globals()["TEST_QUERIES"] = test_queries
    main()
