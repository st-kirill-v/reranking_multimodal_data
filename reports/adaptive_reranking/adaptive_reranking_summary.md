# Adaptive Reranking Summary

Статус: реализация добавлена, полный 308-вопросный прогон еще не запускался в этом сеансе.

## 1. Что было реализовано

Добавлен отдельный экспериментальный режим `adaptive`, который выбирает стратегию реранкинга для каждого вопроса:

- `high_confidence`: пропуск дорогого VL reranker при уверенном retrieval;
- `table_or_text`: full image+text Nemotron VL reranker;
- `visual`: image-only Nemotron VL reranker;
- `unknown`: default best strategy, full image+text Nemotron VL reranker.

Существующие рабочие конфиги, старые results и реализации реранкеров не изменяются.

## 2. Routing strategies

Routing использует:

- `question_crop_intent()` из `src/cropping/layout_aware_eval.py`;
- metadata DocBench: `multimodal-t`, `multimodal-f`;
- retrieval confidence через `top1 - top2` score gap;
- fallback на full image+text reranker для ambiguous cases.

## 3. Сколько вопросов попало в каждый route

Будет заполнено после полного запуска:

```bash
python scripts/run_experiment.py --config configs/experiments/adaptive_reranking_308_qwen3vl30b.yaml
```

## 4. Где reranker был пропущен

Будет заполнено после полного запуска. Поля появятся в `results/adaptive_reranking_308_qwen3vl30b/predictions.jsonl`:

- `adaptive_route`;
- `adaptive_strategy`;
- `adaptive_skipped_reranker`;
- `adaptive_route_reason`;
- `adaptive_retrieval_top1_score`;
- `adaptive_retrieval_top1_gap`.

## 5. Итоговые метрики

Пока не рассчитаны. После запуска будут сохранены:

- `results/adaptive_reranking_308_qwen3vl30b/metrics.json`;
- `results/adaptive_reranking_308_qwen3vl30b/metrics_table.md`;
- `reports/adaptive_reranking/adaptive_reranking_summary.md`.

## 6. Сравнение с baseline

Baseline для сравнения:

| Baseline | Mean F1 | Latency |
|---|---:|---:|
| Best quality baseline | 0.7023 | 13.6441 |
| Fast baseline | 0.6575 | 2.5080 |
| No-reranker full image+text baseline | 0.6784 | 3.4263 |

## 7. Вывод

Финальный вывод о качестве и latency можно делать только после полного запуска. Если adaptive-режим сохранит Mean F1 близко к `0.7023` и снизит latency, его стоит включать в защиту как развитие алгоритма. Если качество заметно снизится, результат лучше использовать как дополнительную абляцию.
