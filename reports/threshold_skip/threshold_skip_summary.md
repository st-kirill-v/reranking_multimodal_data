# Threshold Skip Reranking Summary

Статус: additional exploratory experiment.

## 1. Что реализовано

Добавлен отдельный режим `threshold_skip`: перед запуском дорогого text+image Nemotron VL reranker проверяется уверенность первичного retrieval. Если `top1_score >= threshold_top1` и `gap >= threshold_gap`, реранкер пропускается.

## 2. Проверенные threshold

- threshold pairs: `[(0.7, 0.05), (0.7, 0.1), (0.75, 0.1), (0.75, 0.15), (0.8, 0.1), (0.8, 0.15), (0.85, 0.1), (0.85, 0.2), (0.9, 0.25)]`
- questions per threshold: `30`

## 3. Используемый score

Используется только `RetrievalCandidate.score` после первого retrieval. `fusion_score` и `rerank_score` не используются для принятия skip/run решения.

## 4. Распределение retrieval scores

- top1 distribution: `{'min': None, 'max': None, 'mean': None, 'median': None, 'p25': None, 'p75': None, 'p90': None, 'p95': None}`
- gap distribution: `{'min': None, 'max': None, 'mean': None, 'median': None, 'p25': None, 'p75': None, 'p90': None, 'p95': None}`

## 5. Skip frequency

- total decisions: 0
- skipped decisions: 0
- skipped share across all completed grid rows: 0.0000

## 6. Grid results

| top1 | gap | Mean F1 | EM | F1 > 0.5 | MM-T | MM-F | latency | skip rate | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 0.05 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.7 | 0.1 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.75 | 0.1 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.75 | 0.15 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.8 | 0.1 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.8 | 0.15 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.85 | 0.1 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.85 | 0.2 |  |  |  |  |  |  |  | not_run_or_metrics_missing |
| 0.9 | 0.25 |  |  |  |  |  |  |  | not_run_or_metrics_missing |

## 7. Лучшие threshold

- Best Quality Threshold: `None`
- Best Balanced Threshold: `None`
- Best Fast Threshold: `None`

## 8. Baselines

| Baseline | Mean F1 | Latency |
|---|---:|---:|
| Best quality baseline | 0.7023 | 13.6441 |
| No-reranker full image+text baseline | 0.6784 | 3.4263 |
| Fast fusion baseline | 0.6575 | 2.508 |

## 9. Методологические ограничения

- Threshold подбирается на небольшом subset из 308 вопросов.
- Возможен overfitting под DocBench multimodal subset.
- Результат следует считать exploratory experiment.
- Основной baseline статьи не заменять без подтверждения на другом split.

## 10. Вывод для статьи и защиты

Если Threshold Skip сохраняет Mean F1 близко к лучшему baseline и снижает latency, его можно показывать на защите как простой quality/latency baseline перед Adaptive Reranking. В статью включать только как дополнительный exploratory результат, не как замену основного результата.

## Output

- results root: `results\threshold_skip_reranking_308_qwen3vl30b\grid_limit30`
- CSV: `reports\threshold_skip\threshold_skip_grid_results.csv`
- decisions: `reports\threshold_skip\threshold_skip_decisions.jsonl`
