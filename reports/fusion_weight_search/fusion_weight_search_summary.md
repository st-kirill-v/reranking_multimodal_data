# Fusion Weight Search Summary

Статус: additional exploratory fusion experiment.

## Что проверяется

Подбор весов `alpha`, `beta`, `gamma` для Score Fusion:

```text
final_score = alpha * normalized_retrieval_score
            + beta  * normalized_text_rerank_score
            + gamma * normalized_image_rerank_score
            + heuristic_bonuses
```

В этой сетке используется Nemotron fusion режим, где доступны все три score. Для режимов без image reranker вес `gamma` должен быть перенормирован на доступные score; такие результаты в данной сетке не смешиваются с основной таблицей.

## Methodological note

DocBench subset содержит 308 вопросов, поэтому подбор весов несет риск overfitting. Если найденный результат окажется лучше baseline, он не должен автоматически заменять основной результат статьи без отдельной валидации.

## Results

| alpha | beta | gamma | Mean F1 | EM | F1 > 0.5 | MM-T F1 | MM-F F1 | latency | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.2 | 0.2 | 0.6 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.2 | 0.3 | 0.5 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.2 | 0.4 | 0.4 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.3 | 0.2 | 0.5 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.3 | 0.3 | 0.4 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.3 | 0.4 | 0.3 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.4 | 0.2 | 0.4 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.4 | 0.3 | 0.3 |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.5 | 0.2 | 0.3 |  |  |  |  |  |  | not_run_or_metrics_missing |

## Best candidates

- Best quality: not available
- Best F1/latency trade-off: not available

## Baseline comparison

| Baseline | Baseline Mean F1 | Baseline latency | Best fusion delta F1 | Best fusion delta latency |
|---|---:|---:|---:|---:|
| not available |  |  |  |  |

## Output

- results root: `results\fusion_weight_search_308_qwen3vl30b`
- CSV: `reports\fusion_weight_search\fusion_weight_search_results.csv`
