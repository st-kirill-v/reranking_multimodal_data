# Сравнение с литературой для защиты

## 1. Чем работа отличается от существующих?

Краткий ответ:

> Наша работа не предлагает новый VLM, retriever или архитектуру реранкера. Отличие состоит в том, что мы изолируем этап мультимодального реранкинга внутри Document QA pipeline и контролируемо оцениваем его вклад в качество ответа и latency.

Большинство близких работ решают более широкую задачу. DocBench вводит benchmark для document reading systems. M3DocRAG, ViDoRe V3, ColPali и MM-Embed развивают retrieval или multimodal retrieval. DocDancer и MARDoc строят agentic long-document QA systems с search/read tools, memory и reflection. В нашей работе исследуется более узкий компонент:

```text
Retrieval -> Reranking -> Evidence Construction -> VLM Answer
```

и сравниваются:

```text
No Reranker -> Text Reranker -> Multimodal Reranker -> Adaptive Reranking
```

Главная идея: показать, когда реранкинг действительно улучшает answer quality, какой тип реранкинга полезнее и какую latency cost он добавляет.

## 2. Почему это не просто очередной Multimodal RAG?

Краткий ответ:

> Multimodal RAG обычно оценивает систему целиком: retriever, evidence construction, reasoning и генерацию ответа. В нашей работе центральный объект исследования - не весь RAG pipeline, а reranking stage как отдельный компонент.

Retrieval и Qwen3-VL в проекте используются как фиксированное экспериментальное окружение. Они нужны, чтобы измерить вклад реранкинга на уровне итогового ответа. Поэтому работа отвечает не на вопрос "какая MRAG-система лучшая?", а на более компонентный вопрос:

```text
Что меняется, если после retrieval добавить text reranker,
multimodal reranker или adaptive routing?
```

Такое исследование полезно инженерно: в реальной системе реранкер может быть главным источником как прироста качества, так и задержки.

## 3. Почему нельзя напрямую сравнивать Mean F1 с DocDancer / MARDoc?

Краткий ответ:

> Потому что это разные evaluation protocols, разные метрики и разные типы систем.

DocDancer и MARDoc являются agentic long-document QA systems. Они используют search/read tools, memory, reflection, иногда fine-tuning или synthetic trajectories, и отчитываются по LasJ / judge-based метрикам на DocBench и MMLongBench-Doc. Наша работа использует мультимодальное подмножество DocBench и оценивает controlled reranking pipeline через Mean F1, F1 > 0.5, Exact Match и latency.

Поэтому некорректно говорить:

```text
Наша работа лучше / хуже DocDancer или MARDoc.
```

Корректно говорить:

```text
DocDancer и MARDoc показывают возможности agentic long-document QA,
а наша работа исследует другой вопрос: вклад reranking stage
при фиксированном Document QA pipeline.
```

Именно поэтому в статье они используются как related work и внешние ориентиры, но не как прямые численные baseline.

## 4. В чём актуальность работы?

Актуальность связана с тем, что в document QA по PDF-документам ошибка часто возникает не только на этапе генерации ответа, но и раньше: система может выбрать не ту страницу, не тот фрагмент таблицы, не тот crop или не связать текстовую и визуальную evidence. Реранкинг находится между retrieval и VLM generation и напрямую влияет на то, какие данные получит модель ответа.

В наших экспериментах лучший мультимодальный реранкер достигает `Mean F1 = 0.7023` при latency `13.6441s`. Adaptive Reranking сохраняет почти то же качество (`0.7009`) и снижает latency до `9.7882s`. Это показывает, что реранкинг является не второстепенной деталью, а самостоятельным проектным компонентом с измеримым quality/latency trade-off.

## 5. Какие работы ближе всего?

Самые близкие группы работ:

| Группа | Примеры | Чем близки | Чем отличаются |
| --- | --- | --- | --- |
| Benchmark-и Document QA | DOCBENCH | Дают датасет и задачу document reading | Не изолируют reranking |
| Document VQA RAG | Enhancing Document VQA via RAG | Анализируют retrieval/reranking gains | Не делают DocBench controlled No/Text/Multimodal reranking |
| Multimodal retrieval | ColPali, ViDoRe V3, MM-Embed | Показывают важность visual retrieval/reranking | Обычно оценивают retrieval, а не final answer quality на DocBench |
| Multimodal RAG frameworks | M3DocRAG, MMRAG-DocQA | Используют retrieval, evidence selection, VLM | Фокус на framework/retrieval, а не на изоляции reranking stage |
| Agentic long-document QA | DocDancer, MARDoc | Работают с DocBench и long-document QA | Используют agentic search, memory, reflection; не standalone reranking study |
| Graph-based MRAG | MAGE-RAG, RAG-Anything | Развивают graph / hybrid evidence representation | Полезны как future work, но не являются controlled reranking-stage evaluation |

## 6. Формулировка для комиссии

Короткая версия для ответа:

> Современные работы по Document QA и Multimodal RAG часто фокусируются на retrieval, visual document retrieval, agentic search, memory/reflection или end-to-end системах. Моя работа занимает более узкую нишу: я изолирую этап мультимодального реранкинга и показываю его вклад в итоговое качество ответа и latency. Поэтому вклад работы методологический и экспериментальный: controlled comparison No Reranker, Text Reranker, Multimodal Reranker и Adaptive Reranking на мультимодальном подмножестве DocBench.

## 7. Что важно не утверждать

Не стоит говорить:

- "Мы превосходим DocDancer / MARDoc".
- "Мы создали самостоятельную новую модель реранкинга".
- "Мы предлагаем новую архитектуру Multimodal RAG".
- "Наш Mean F1 напрямую сравним с LasJ из agentic работ".

Лучше говорить:

- "Мы исследуем вклад reranking stage как отдельного компонента".
- "Мы показываем quality/latency trade-off".
- "Мы дополняем agentic и retrieval-oriented работы компонентным анализом".
- "Наш результат полезен для проектирования Document QA pipeline, где нужно выбрать между качеством и стоимостью inference".

## 8. Итоговый вывод

Наша работа актуальна, потому что современные Document QA / Multimodal RAG исследования часто фокусируются на retrieval, agentic search, memory, reasoning или end-to-end системах, тогда как вклад мультимодального реранкинга как отдельного компонента остаётся недостаточно изолированным. Наша работа закрывает эту нишу через controlled comparison `No Reranker`, `Text Reranker`, `Multimodal Reranker` и `Adaptive Reranking` с answer-level метриками и latency analysis.
