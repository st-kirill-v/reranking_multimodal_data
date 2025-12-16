"""
Метрики оценки поиска: Recall@k, nDCG@k, MRR, точность, латентность.
Простой и работающий модуль для вычисления основных метрик.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class SearchMetrics:
    """Вычисление метрик качества поиска."""

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Precision@k: доля релевантных среди первых k результатов."""
        if k == 0 or not retrieved_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_count = len(set(top_k) & set(relevant_ids))
        return relevant_count / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Recall@k: доля найденных релевантных среди первых k результатов."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_found = len(set(top_k) & set(relevant_ids))
        return relevant_found / len(relevant_ids)

    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Average Precision: средняя точность по позициям релевантных документов."""
        if not relevant_ids:
            return 0.0

        ap_sum = 0.0
        relevant_count = 0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap_sum += precision_at_i

        return ap_sum / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids_list: List[List[str]], relevant_ids_list: List[List[str]]
    ) -> float:
        """MRR: среднее обратного ранга первого релевантного документа."""
        if len(retrieved_ids_list) != len(relevant_ids_list):
            raise ValueError("Количество запросов должно совпадать")

        reciprocal_ranks = []

        for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
            if not relevant_ids:
                reciprocal_ranks.append(0.0)
                continue

            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """nDCG@k: нормализованный дисконтированный кумулятивный выигрыш."""
        if k == 0:
            return 0.0

        # DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 потому что индексация с 0

        # Ideal DCG@k (отсортированные релевантности)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_relevances))):
            idcg += ideal_relevances[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        """Вычисление перцентилей латентности: p50, p95."""
        if not latencies_ms:
            return {"p50": 0.0, "p95": 0.0}

        sorted_latencies = sorted(latencies_ms)
        p50_index = int(len(sorted_latencies) * 0.5)
        p95_index = int(len(sorted_latencies) * 0.95)

        return {"p50": sorted_latencies[p50_index], "p95": sorted_latencies[p95_index]}


class MetricsReporter:
    """Сбор и отчетность по метрикам."""

    def __init__(self):
        self.metrics_history = []
        self.latencies_ms = []

    def add_query_result(
        self,
        query_id: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: Dict[str, float] = None,
        latency_ms: float = 0,
    ):
        """Добавление результатов одного запроса."""
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        metrics = {
            "query_id": query_id,
            "precision_at_5": SearchMetrics.precision_at_k(retrieved_ids, relevant_ids, 5),
            "precision_at_10": SearchMetrics.precision_at_k(retrieved_ids, relevant_ids, 10),
            "recall_at_5": SearchMetrics.recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall_at_10": SearchMetrics.recall_at_k(retrieved_ids, relevant_ids, 10),
            "average_precision": SearchMetrics.average_precision(retrieved_ids, relevant_ids),
            "ndcg_at_5": SearchMetrics.ndcg_at_k(retrieved_ids, relevance_scores, 5),
            "ndcg_at_10": SearchMetrics.ndcg_at_k(retrieved_ids, relevance_scores, 10),
        }

        self.metrics_history.append(metrics)
        self.latencies_ms.append(latency_ms)

    def calculate_mrr(
        self, all_retrieved_ids: List[List[str]], all_relevant_ids: List[List[str]]
    ) -> float:
        """Вычисление MRR для набора запросов."""
        return SearchMetrics.mean_reciprocal_rank(all_retrieved_ids, all_relevant_ids)

    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки по всем метрикам."""
        if not self.metrics_history:
            return {}

        summary = {}
        for metric_name in self.metrics_history[0].keys():
            if metric_name != "query_id":
                values = [m[metric_name] for m in self.metrics_history]
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        # Добавляем MRR если есть данные
        if len(self.metrics_history) > 1:
            all_retrieved = []
            all_relevant = []
            # Нужно собрать данные для MRR отдельно

        # Добавляем латентность
        if self.latencies_ms:
            latency_percentiles = SearchMetrics.calculate_latency_percentiles(self.latencies_ms)
            summary["latency_ms"] = {
                "mean": np.mean(self.latencies_ms),
                "p50": latency_percentiles["p50"],
                "p95": latency_percentiles["p95"],
            }

        return summary

    def print_summary(self):
        """Краткий вывод метрик."""
        summary = self.get_summary()

        if not summary:
            print("Нет данных метрик.")
            return

        print("Метрики оценки поиска:")
        print(f"Количество запросов: {len(self.metrics_history)}")

        for metric_name, stats in summary.items():
            if metric_name != "latency_ms":
                print(f"{metric_name}: {stats['mean']:.3f} (стд: {stats['std']:.3f})")

        if "latency_ms" in summary:
            lat = summary["latency_ms"]
            print(
                f"Латентность, мс: средняя {lat['mean']:.1f}, p50 {lat['p50']:.1f}, p95 {lat['p95']:.1f}"
            )


# Пример использования
if __name__ == "__main__":
    # Тестовые данные
    reporter = MetricsReporter()

    # Добавляем результаты для нескольких запросов
    reporter.add_query_result(
        query_id="q1",
        retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        relevant_ids=["doc1", "doc3"],
        relevance_scores={"doc1": 1.0, "doc2": 0.0, "doc3": 1.0, "doc4": 0.5, "doc5": 0.0},
        latency_ms=150.5,
    )

    reporter.add_query_result(
        query_id="q2",
        retrieved_ids=["doc2", "doc1", "doc5", "doc4", "doc3"],
        relevant_ids=["doc1", "doc4"],
        relevance_scores={"doc1": 1.0, "doc2": 0.0, "doc3": 0.0, "doc4": 1.0, "doc5": 0.0},
        latency_ms=120.0,
    )

    # Вывод метрик
    reporter.print_summary()

    # Пример вычисления отдельных метрик
    print("\nОтдельные метрики для запроса 1:")
    print(
        f"Precision@5: {SearchMetrics.precision_at_k(['doc1', 'doc2', 'doc3', 'doc4', 'doc5'], ['doc1', 'doc3'], 5):.3f}"
    )
    print(
        f"Recall@5: {SearchMetrics.recall_at_k(['doc1', 'doc2', 'doc3', 'doc4', 'doc5'], ['doc1', 'doc3'], 5):.3f}"
    )
    print(
        f"Average Precision: {SearchMetrics.average_precision(['doc1', 'doc2', 'doc3', 'doc4', 'doc5'], ['doc1', 'doc3']):.3f}"
    )
