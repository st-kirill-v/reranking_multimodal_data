# Отчёт об обновлении article_final.md: Adaptive Reranking

## 1. Обновлённые разделы

- **Аннотация**: добавлен результат Adaptive Reranking как balanced-конфигурации; основной best-quality результат `Mean F1 = 0.7023` сохранён.
- **Введение**: добавлено позиционирование Adaptive Reranking как способа снижения вычислительных затрат за счёт условного применения visual-language reranker.
- **Исследовательские вопросы и вклад работы**: уточнён RQ5 и добавлено упоминание Adaptive Reranking в контексте quality/latency trade-off.
- **Метод**: добавлено описание Adaptive Reranking, route selection и route `high_confidence`, где дорогой VL reranker может быть пропущен.
- **Экспериментальная постановка**: Adaptive Reranking добавлен как дополнительный экспериментальный режим, не подменяющий основную линию `No Reranker -> Text Reranker -> Multimodal Reranker`.
- **5.4 RQ1**: добавлен график сравнения Mean F1 по стратегиям reranking.
- **5.5 RQ2**: обновлена сводная таблица основных результатов, добавлен Adaptive Reranking как balanced режим.
- **5.7 RQ4 / Evidence**: добавлен подраздел `5.7.1 Дополнительный эксперимент: Adaptive route selection`.
- **5.8 RQ5**: Adaptive Reranking добавлен как главный balanced вариант quality/latency trade-off.
- **5.9 Discussion of Findings**: добавлена интерпретация, что reranker не обязательно запускать одинаково для всех вопросов.
- **7 Обсуждение результатов** и **Заключение**: добавлен вывод о перспективности условного применения мультимодального reranker.

## 2. Новые графики

Добавлены три PNG-файла:

| Figure | File |
| --- | --- |
| Answer quality comparison across reranking strategies | `reports/figures/reranking_quality_comparison.png` |
| Adaptive reranking route distribution | `reports/figures/adaptive_route_distribution.png` |
| Quality-latency trade-off | `reports/figures/quality_latency_tradeoff.png` |

Для воспроизводимости добавлен скрипт:

```text
scripts/generate_adaptive_article_figures.py
```

## 3. Использованные Adaptive metrics

Основные значения:

| Metric | Value |
| --- | ---: |
| Total | 308 |
| Mean F1 | 0.7009 |
| F1 > 0.5 | 0.7500 |
| Latency mean | 9.7882s |
| Latency p50 | 9.9154s |
| Latency p95 | 11.2595s |

По типам вопросов:

| Type | Mean F1 | F1 > 0.5 |
| --- | ---: | ---: |
| `multimodal-t` | 0.7181 | 0.7682 |
| `multimodal-f` | 0.6579 | 0.7045 |

Routing:

| Route | Count | Mean F1 | Latency mean |
| --- | ---: | ---: | ---: |
| `table_or_text` | 203 | 0.7185 | 9.8231s |
| `visual` | 88 | 0.6549 | 11.0222s |
| `high_confidence` | 17 | 0.7296 | 2.9837s |

Skipped reranker:

```text
skipped_reranker_count = 17
skipped_reranker_rate = 0.0552
```

Сравнение с best-quality baseline:

| Configuration | Mean F1 | Latency |
| --- | ---: | ---: |
| Best Multimodal | 0.7023 | 13.6441s |
| Adaptive Reranking | 0.7009 | 9.7882s |

Вывод: Adaptive Reranking почти сохраняет качество лучшей мультимодальной конфигурации и снижает среднюю latency примерно на 28%.

## 4. Что не было включено как основной результат

- **Fusion Weight Search** не включён как сильный результат статьи. Он упомянут только в Discussion как exploratory negative result, поскольку pilot subset не показал убедительного улучшения относительно существующего fast fusion baseline.
- **Exact Match** удалён из текста статьи и новых таблиц по текущему требованию. Основные метрики статьи теперь сфокусированы на `Mean F1`, `F1 > 0.5`, MM-T/MM-F F1 и latency.

## 5. Проверка соответствия главной цели статьи

Проверено, что обновлённый текст сохраняет основную научную позицию:

- работа не заявляет новый VLM;
- работа не заявляет новый retriever;
- работа не заявляет новую архитектуру reranker;
- основной результат статьи остаётся `Mean F1 = 0.7023`;
- Adaptive Reranking представлен как balanced дополнительный эксперимент, а не как новый best-quality результат;
- центральный вклад сохраняется как controlled evaluation мультимодального реранкинга как отдельного компонента Document QA pipeline.
