# Аудит возможностей улучшения мультимодального реранкинга

Репозиторий: `reranking_multimodal_data`
Тема: **Разработка алгоритма реранкинга мультимодальных данных**
Дата аудита: 23.06.2026

Аудит выполнен как аналитическая проверка текущего состояния проекта. Код, конфигурации экспериментов, результаты и архитектура не изменялись.

## 1. Current Reranking Architecture

Текущий пайплайн исследования соответствует постановке статьи:

```text
Question
  ↓
Retrieval / Candidate Generation
  ↓
Reranking
  ↓
Evidence Construction
  ↓
Qwen3-VL
  ↓
Answer
```

Основной объект исследования в коде представлен модулем `src/reranking/` и интеграцией реранкинга в `scripts/evaluate_full_pipeline_layout_aware_clean.py`.

### Реализованные типы реранкинга

| Компонент | Реализация | Назначение |
|---|---|---|
| No Reranker | `src/reranking/no_reranker.py` | Базовый режим без изменения порядка кандидатов |
| Text Reranker | `src/reranking/text_reranker.py` | Текстовый cross-encoder / lexical reranking |
| Nemotron VL Reranker | `src/mmrag/rerank.py`, `NemotronVLReranker` | Мультимодальный image-only реранкер |
| Nemotron VL Text+Image Reranker | `src/mmrag/rerank.py`, `NemotronVLTextImageReranker` | Мультимодальный реранкер с изображением страницы и извлеченным текстом |
| Score Fusion | `src/reranking/fusion.py` | Комбинация retrieval score, text rerank score, image rerank score и эвристических бонусов |

### Наличие score-компонентов

Возможность score fusion уже почти полностью реализована:

| Score | Наличие | Где используется |
|---|---:|---|
| `retrieval_score` | Да | `RetrievalCandidate.score` в `src/mmrag/schema.py` |
| `reranker_score` / image rerank score | Да | `RetrievalCandidate.rerank_score` после Nemotron VL reranker |
| `text_rerank_score` | Да | Добавляется в `CrossEncoderTextReranker` |
| `fusion_score` | Да | Добавляется в `fuse_candidates()` |

Формула в текущей реализации близка к требуемой идее:

```text
final_score =
  alpha * normalized_retrieval_score
+ beta  * normalized_text_rerank_score
+ gamma * normalized_image_rerank_score
+ heuristic_bonuses
```

Дополнительно уже реализованы бонусы:

- `number_match`;
- `keyword_match`;
- `exact_phrase_match`;
- `table_header_match`.

### Evidence construction

В финальный VLM-контекст уже могут попадать:

- изображение полной страницы;
- layout crop;
- OCR;
- `page_text`;
- `caption`;
- `table_text`.

Наиболее сильная стратегия по статье и отчетам: **full page + layout crop**.

### Layout-aware логика

В `src/cropping/layout_aware_eval.py` уже есть полезные элементы:

- определение `question_crop_intent`;
- разделение на `table`, `visual`, `unknown`;
- эвристики для table-heavy и visual-heavy вопросов;
- бонусы для caption/reference match;
- штрафы за несовпадение типа crop;
- адаптивные настройки context/crop policy.

Важно: эта логика сейчас влияет прежде всего на выбор evidence/crop, но не является полноценным adaptive reranking strategy.

## 2. Existing Experiments

Основная экспериментальная линия уже соответствует статье:

```text
No Reranker
  → Text Reranker
  → Multimodal Reranker
```

Ключевые конфигурации находятся в `configs/experiments/`.

### Главные протестированные режимы

| Режим | Пример конфигурации | Смысл |
|---|---|---|
| Nemotron image retrieval + no reranker | `image_text_full_308_nemotron_no_reranker_qwen3vl30b.yaml` | Сильный retrieval baseline без реранкинга |
| Nemotron image retrieval + VL reranker | `multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b.yaml` | Мультимодальный image-only reranking |
| Nemotron full image+text + VL reranker | `image_text_full_308_nemotron_qwen3vl30b.yaml` | Лучшая quality-конфигурация |
| Fusion Nemotron + VL reranker | `image_text_fusion_308_nemotron_qwen3vl30b.yaml` | Score fusion поверх image/text signals |
| Fusion Nemotron без image reranker | `image_text_fusion_308_nemotron_no_reranker_qwen3vl30b.yaml` | Быстрый вариант с хорошим качеством |
| BM25 + BGE reranker | `text_reranker_308_bge_large_qwen3vl30b.yaml` | Текстовый baseline |

### Ключевые результаты

| Конфигурация | Mean F1 | F1 > 0.5 | Exact Match | Latency, sec |
|---|---:|---:|---:|---:|
| Nemotron full image+text + VL reranker + Qwen3-VL-30B | **0.7023** | **0.7565** | 0.1201 | 13.6441 |
| Nemotron image+text input + VL reranker + Qwen3-VL-30B | 0.6979 | 0.7468 | 0.1169 | 12.9920 |
| Nemotron image + VL reranker + Qwen3-VL-30B | 0.6902 | 0.7500 | 0.1071 | 11.0702 |
| Fusion Nemotron + VL reranker + Qwen3-VL-30B | 0.6793 | 0.7175 | 0.1136 | 9.4122 |
| Nemotron full image+text no reranker + Qwen3-VL-30B | 0.6784 | 0.7143 | 0.1201 | 3.4263 |
| Fusion Nemotron no image reranker + Qwen3-VL-30B | 0.6575 | 0.6786 | 0.1136 | **2.5080** |
| BM25 + BGE-reranker-large + Qwen3-VL-30B | 0.5497 | 0.5714 | 0.0812 | 1.2341 |
| BM25 no reranker + Qwen3-VL-30B | 0.5144 | 0.5325 | 0.0909 | 0.7146 |

Вывод: лучший прирост качества дает мультимодальный реранкинг, но основной latency cost также связан именно с ним.

## 3. Existing Ablations

В проекте уже проведены важные абляции, достаточные для защиты основной научной позиции.

### Абляция реранкера

По `reports/experiment_summary/reranker_ablation.csv`:

| Группа | Mean F1 без reranker | Mean F1 с reranker | Прирост | Latency cost |
|---|---:|---:|---:|---:|
| ColPali + Qwen3-VL-8B | 0.5657 | 0.6170 | +0.0513 | +5.3667 sec |
| ColPali + Qwen3-VL-30B | 0.6539 | 0.6860 | +0.0321 | +8.8082 sec |
| Nemotron image + Qwen3-VL-30B | 0.6479 | 0.6902 | +0.0423 | +8.5167 sec |
| Nemotron fusion + Qwen3-VL-30B | 0.6575 | 0.6793 | +0.0219 | +6.9042 sec |
| Nemotron full image+text + Qwen3-VL-30B | 0.6784 | 0.7023 | +0.0240 | +10.2179 sec |
| BM25 + BGE-large + Qwen3-VL-30B | 0.5144 | 0.5497 | +0.0353 | +0.5198 sec |
| Text evidence + BGE-large + Qwen3-VL-30B | 0.4424 | 0.5354 | +0.0930 | +0.5149 sec |

Средний эффект реранкинга из статьи:

- `+0.0428` Mean F1;
- `+5.8355 sec` latency.

### Абляция retrieval

По агрегированным результатам:

| Retriever | Mean F1 | Max F1 | Комментарий |
|---|---:|---:|---|
| Nemotron image | 0.6791 | **0.7023** | Лучший candidate pool |
| ColPali / ColVision | 0.6402 | 0.6860 | Сильный мультимодальный baseline |
| BM25 | 0.5328 | 0.5497 | Полезен как быстрый текстовый baseline |
| BGE-base / BGE-large text encoders | 0.5205 / 0.4889 | 0.5354 | Уступают мультимодальному retrieval |

### Абляция evidence strategy

| Evidence strategy | Mean F1 | Max F1 | Комментарий |
|---|---:|---:|---|
| Full page + layout crop | **0.6629** | **0.7023** | Основная эффективная стратегия |
| Page text | 0.5328 | 0.5497 | Быстро, но ограниченно |
| OCR + page_text + caption + table_text | 0.4994 | 0.5354 | Расширение текста само по себе не гарантирует прирост |

## 4. Quick Wins (1–2 дня)

Ниже перечислены улучшения, которые можно завершить быстрее всего, потому что большая часть инфраструктуры уже есть.

### 4.1. Adaptive routing по типу вопроса

Идея:

```text
text/table-heavy question → full image+text + VL reranker
visual-heavy question     → image-only Nemotron VL reranker или layout-aware variant
high-confidence retrieval → no reranker / fast fusion
```

Что уже есть:

- типы DocBench: `multimodal-t`, `multimodal-f`;
- `question_crop_intent()` для `table`, `visual`, `unknown`;
- layout-aware признаки;
- несколько готовых конфигураций с разным quality/latency profile.

Ожидаемый эффект:

- качество: небольшой прирост или сохранение Mean F1 около текущего максимума;
- latency: потенциальное снижение средней задержки за счет отказа от дорогого text+image reranking на части вопросов;
- особенно интересно, что image-only Nemotron VL reranker дает немного лучший MM-F результат, чем full image+text конфигурация, при меньшей latency.

Сложность: низкая-средняя.
Риск: средний, потому что неправильная маршрутизация может ухудшить часть вопросов.

Вердикт: **лучший кандидат перед защитой**, если цель — показать развитие алгоритма без переобучения.

### 4.2. Threshold-based skip reranking

Идея:

```text
если top-1 retrieval score уверенно выше остальных,
то не запускать дорогой VL reranker
```

Что уже есть:

- `RetrievalCandidate.score`;
- логирование `retrieval_scores`;
- сильный no-reranker baseline: Mean F1 `0.6784` при latency `3.4263 sec`;
- лучший reranker baseline: Mean F1 `0.7023` при latency `13.6441 sec`.

Ожидаемый эффект:

- Mean F1: скорее сохранение с небольшим снижением или небольшой прирост при удачном threshold;
- Latency: заметное снижение средней задержки;
- F1 > 0.5: может сохраниться близко к текущему уровню, если reranker пропускать только для уверенных случаев.

Сложность: низкая.
Риск: низкий-средний.
Вердикт: **лучший latency quick win**.

### 4.3. Grid search для score fusion weights

Идея:

```text
final_score =
  alpha * retrieval_score
+ beta  * text_reranker_score
+ gamma * image_reranker_score
+ layout/text bonuses
```

Что уже есть:

- готовый `score_fusion` режим;
- параметры `alpha`, `beta`, `gamma`;
- текстовые бонусы;
- конфиги fusion-экспериментов;
- быстрый fusion без image reranker: Mean F1 `0.6575`, latency `2.5080 sec`.

Ожидаемый эффект:

- качество: вероятно `+0.005`-`+0.015` к текущим fusion-конфигурациям, но маловероятно обогнать `0.7023`;
- latency: можно сохранить низкой, если использовать no-image-reranker fusion;
- практическая ценность: сильный fast/balanced режим.

Сложность: низкая.
Риск: низкий для эксперимента, средний для научного вывода из-за риска подгонки под 308 вопросов.
Вердикт: **хороший быстрый эксперимент**, но не главный кандидат для нового лучшего качества.

### 4.4. Multi-stage reranking cascade

Идея:

```text
Retriever Top-K
  ↓
Text Reranker / Fusion prefilter
  ↓
Multimodal Reranker
  ↓
VLM
```

Что уже есть:

- текстовые реранкеры BGE/Jina/MiniLM;
- Nemotron VL reranker;
- score fusion;
- CLI-аргументы для text reranker внутри fusion.

Что нужно добавить:

- явный режим cascade, где text reranker сужает Top-K перед дорогим VL reranker;
- логирование числа кандидатов до/после каждого этапа.

Ожидаемый эффект:

- качество: неопределенное, особенно на visual-heavy вопросах;
- latency: может заметно снизиться, если Nemotron VL reranker обрабатывает меньше кандидатов;
- F1 > 0.5: риск просадки, если текстовый этап отфильтрует визуально важную страницу.

Сложность: средняя.
Риск: средний.
Вердикт: перспективно, но лучше делать после threshold/adaptive routing.

### 4.5. Layout bonus в final score

Идея: использовать уже вычисляемые layout/crop признаки не только для evidence selection, но и как легкий бонус в ранжировании.

Что уже есть:

- `question_crop_intent`;
- `score_layout_crop_v2`;
- `table_header_match` в fusion;
- caption/reference heuristics.

Ожидаемый эффект:

- качество: небольшой прирост на table-heavy и figure-heavy вопросах;
- latency: почти без изменений;
- риск: возможна переоценка страниц с визуально подходящим, но семантически неверным crop.

Сложность: низкая-средняя.
Риск: средний.
Вердикт: полезно как аккуратное развитие existing fusion, но требует контрольной абляции.

## 5. Medium Improvements

### 5.1. Более строгий adaptive reranking classifier

Сейчас в проекте есть эвристики для question intent, но нет отдельного полноценного классификатора стратегии реранкинга.

Реалистичный вариант:

- использовать `multimodal-t` / `multimodal-f`;
- добавить lexical rules для table/figure/chart/formula;
- выбирать стратегию из уже протестированных конфигураций.

Сложность: средняя.
Оценка срока: 2-4 дня.
Риск: средний.

### 5.2. Validation-tuned fusion

Можно подобрать веса fusion на части DocBench и проверить на оставшейся части.

Плюсы:

- использует уже существующую инфраструктуру;
- может дать более убедительный balanced baseline;
- хорошо оформляется как ablation.

Минусы:

- 308 вопросов — маленький набор;
- высок риск overfitting;
- потребуется аккуратное разделение на validation/test.

Сложность: средняя.
Оценка срока: 2-4 дня.
Риск: средний.

### 5.3. Error-case reranking

Можно отдельно проанализировать вопросы, где:

- retrieval top-1 неверен;
- reranker ухудшил порядок;
- evidence crop выбран неправильно;
- VLM ошибся при правильном evidence.

Это может дать точечные улучшения без смены архитектуры.

Сложность: средняя.
Оценка срока: 2-3 дня.
Риск: низкий-средний.

### 5.4. Оптимизация text+image reranker input

В лучшем режиме Nemotron VL Text+Image получает текст страницы до `4096` символов. Можно проверить:

- `2048` vs `4096` vs `8192`;
- порядок полей `table_text`, `caption`, `page_text`, `ocr`;
- исключение шумного OCR для части вопросов.

Сложность: низкая.
Оценка срока: 1-3 дня.
Риск: низкий-средний.
Главный эффект: возможно снижение latency и небольшое улучшение качества.

## 6. High-Risk Ideas

### 6.1. Qwen3-VL-Reranker

Проверка репозитория показывает, что Qwen3-VL уже используется как VLM answer generator, но отдельный `Qwen3-VL-Reranker` в коде не реализован.

Что потребуется:

- определить scoring prompt или model endpoint для ранжирования;
- стабилизировать формат входа `question + page image + page text`;
- обеспечить числовой score, пригодный для сортировки;
- добавить batch processing;
- провести полную абляцию против Nemotron VL reranker.

Оценка реалистичности за 1-2 дня: низкая.
Риск: высокий.
Потенциальный эффект: высокий, но неопределенный.

Вывод: для защиты лучше оставить как future work, если нет уже готового локального/remote reranking endpoint.

### 6.2. Graph-based evidence

Сравнение с идеями RAGAnything / MAGE-RAG показывает, что графовая evidence architecture в текущем проекте не реализована.

Чтобы внедрить graph-based evidence, нужны:

- схема узлов: page, text block, table, figure, caption, crop;
- связи: belongs-to, references, caption-of, same-page, semantic-neighbor;
- graph retrieval;
- graph reranking или evidence traversal;
- адаптация VLM-контекста;
- новая оценка и абляция.

Оценка реалистичности за 2 дня: очень низкая.
Риск: очень высокий.
Причина: это не точечное улучшение реранкинга, а новая архитектура evidence representation.

Вывод: не внедрять перед защитой. Уместно описать как направление развития.

### 6.3. Fine-tuning reranker

В проекте сейчас используются готовые reranker-модели без дообучения под DocBench.

Fine-tuning потребует:

- разметки положительных/отрицательных кандидатов;
- train/validation/test split;
- контроля leakage по документам;
- GPU-ресурсов;
- отдельной методологической секции.

Оценка реалистичности за 1-2 дня: низкая.
Риск: высокий.
Вывод: не делать перед защитой.

## 7. Recommended Plan Before Defense

Рекомендуемый план должен сохранять научную позицию статьи: проект исследует **мультимодальный реранкинг**, а retrieval, evidence construction и VLM рассматриваются как факторы, влияющие на его эффективность.

### Минимальный безопасный план

1. Зафиксировать текущий лучший результат как основной:
   - Mean F1 `0.7023`;
   - F1 > 0.5 `0.7565`;
   - latency `13.6441 sec`;
   - конфигурация `Nemotron full image+text + VL reranker + Qwen3-VL-30B`.

2. Не менять статью и результаты без нового полного прогона.

3. Для презентации дополнительно подчеркнуть:
   - реранкинг стабильно улучшает качество;
   - мультимодальный реранкинг превосходит текстовый;
   - latency является главным ограничением;
   - fast configuration уже дает Mean F1 `0.6575` при `2.5080 sec`.

### Если есть 1 день

Сделать только аналитическую проверку threshold/adaptive policy на уже сохраненных логах, без переписывания пайплайна:

- посчитать, где no-reranker уже выбирает правильную страницу;
- оценить, на какой доле вопросов можно пропустить VL reranker;
- подготовить график quality/latency trade-off.

Это безопасно для защиты, потому что не меняет основной результат.

### Если есть 2 дня

Наиболее разумный эксперимент:

```text
Adaptive Reranking:
  high-confidence retrieval → no reranker / fast fusion
  visual-heavy             → Nemotron image VL reranker
  text/table-heavy          → Nemotron full image+text VL reranker
```

Почему именно он:

- использует уже реализованные компоненты;
- хорошо связан с темой "алгоритм реранкинга";
- может улучшить latency без отказа от мультимодального реранкинга;
- легко объясняется на защите.

Что обязательно проверить:

- Mean F1;
- F1 > 0.5;
- latency;
- результаты отдельно для `multimodal-t` и `multimodal-f`;
- сравнение с текущей лучшей конфигурацией.

## ТОП-5 улучшений по соотношению

Оценка дана по критерию:

```text
ожидаемый прирост качества или качества/latency
/
время реализации
```

| Rank | Улучшение | Ожидаемый эффект | Время | Риск | Итог |
|---:|---|---|---|---|---|
| 1 | Threshold-based skip reranking | Сильное снижение latency при близком F1 | 1 день | Низкий-средний | Лучший быстрый latency win |
| 2 | Adaptive routing по типу вопроса | Потенциально сохраняет/слегка улучшает F1 и снижает latency | 1-2 дня | Средний | Лучшее развитие алгоритма перед защитой |
| 3 | Grid search score fusion weights | Может улучшить fast/balanced fusion на `+0.005`-`+0.015` F1 | 1 день | Низкий-средний | Хороший эксперимент без смены архитектуры |
| 4 | Оптимизация text+image input для Nemotron VL reranker | Может снизить latency и шум текста | 1-2 дня | Низкий-средний | Безопасная локальная настройка |
| 5 | Multi-stage reranking cascade | Может снизить стоимость VL reranking | 2 дня | Средний | Перспективно, но требует аккуратной абляции |

## Итоговый вывод

В проекте уже реализованы почти все базовые элементы для исследования улучшений мультимодального реранкинга: retrieval score, text rerank score, image rerank score, score fusion, layout-aware evidence и несколько сильных экспериментальных конфигураций.

Самые реалистичные улучшения перед защитой не требуют нового обучения модели. Они находятся в области **адаптивного выбора стратегии реранкинга**, **условного пропуска дорогого VL reranker** и **настройки score fusion**. Замена Nemotron на Qwen3-VL-Reranker и внедрение graph-based evidence являются потенциально интересными, но слишком рискованными для короткого срока и должны оставаться future work.
