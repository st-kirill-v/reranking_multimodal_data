# Обновление финальной статьи

## Изменения в экспериментальной части

- Добавлен контролируемый эксперимент `Nemotron image + BGE-reranker-large + Qwen3-VL-30B text-only`:
  `Mean F1 = 0.5674`, `F1 > 0.5 = 0.5942`, `Latency = 1.1160 s`.
- В сравнении с `BM25 + BGE-reranker-large + Qwen3-VL-30B` изменён только механизм генерации начальных 30 кандидатов. В обоих вариантах используются `page_text`, BGE-reranker-large, top-5 текстовых страниц и text-only Qwen3-VL-30B.
- В статье зафиксирован прирост Mean F1 с `0.5497` до `0.5674` при переходе от BM25 к Nemotron image retrieval.
- Из статьи и парной ablation-таблицы исключён результат `Nemotron full image+text + BGE-reranker-large + Qwen30B`; исходные файлы эксперимента сохранены.
- Обновлены таблицы RQ1--RQ5, агрегаты по reranker, candidate generation и evidence, а также Pareto-таблица.

## Adaptive Reranking

- Режим перенесён из главной экспериментальной линии в подраздел дополнительных экспериментов.
- Adaptive Reranking перенесён в дополнительный подраздел и исключён из основной линии сравнения.
- Adaptive Reranking исключён из основных таблиц, Pareto frontier, ключевых выводов и главных рисунков.
- Данные дополнительного эксперимента и его показатели сохранены в статье, но не используются в основных выводах о качестве и latency.

## Рисунки

- Созданы новые основные версии рисунков без Adaptive Reranking:
  - `reports/figures/reranking_quality_latency_scatter_main.png`;
  - `reports/figures/reranking_mean_f1_barplot_main.png`.
- Старые графические файлы сохранены без удаления.
