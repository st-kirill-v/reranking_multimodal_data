# Novelty-check: мультимодальный реранкинг как controlled component в Document QA / Multimodal RAG

Дата проверки: 23.06.2026
Цель: проверить, существуют ли работы, которые уже делают то же самое, что наша статья: controlled study отдельного этапа мультимодального реранкинга в Document QA / Multimodal RAG pipeline.

Проверяемая формулировка:

> Современные работы по Document QA и Multimodal RAG часто фокусируются на retrieval, agentic evidence search, memory/reflection или end-to-end systems. Однако мультимодальный reranking как отдельный controlled component с сравнением No Reranker, Text Reranker и Multimodal Reranker, answer-level metrics и latency analysis остаётся недостаточно изученным.

## 1. Executive Summary

Короткий вывод: **формулировка в целом корректна, но ее нужно писать осторожно**. В найденных работах действительно активно исследуются multimodal retrieval, visual document retrieval, agentic evidence search, graph evidence, memory/reflection и end-to-end Document QA systems. При этом не найдена работа, которая одновременно закрывает все признаки нашей постановки:

1. Document QA / PDF QA / Multimodal RAG.
2. Retrieval по страницам или фрагментам документа.
3. Отдельный этап reranking.
4. Multimodal / visual-language reranking.
5. Сравнение с text-only reranking.
6. Сравнение с no-reranker baseline.
7. Answer-level evaluation.
8. Latency / compute-cost analysis.
9. Ablation именно по reranking stage.

Самая близкая работа: **Enhancing Document VQA Models via Retrieval-Augmented Generation**. Она систематически оценивает RAG для Document VQA, сравнивает text-based и visual retrieval variants, использует answer-level metric ANLS и прямо указывает, что retrieval/reranking components дают основной вклад. Поэтому она **частично угрожает новизне**, но не закрывает нашу точную постановку: controlled comparison `No Reranker -> Text Reranker -> Multimodal Reranker` на DocBench с Mean F1 / F1 > 0.5 / Exact Match / latency не является ее центральным объектом.

Итоговая позиция статьи должна быть не "первое исследование мультимодального реранкинга", а:

> Работа дополняет существующие исследования Document QA и Multimodal RAG controlled-анализом именно reranking stage: сравниваются отсутствие реранкинга, текстовый реранкинг и мультимодальный реранкинг при фиксированном Document QA pipeline, с оценкой answer quality и latency trade-off.

## 2. Closest Works

### Сводная таблица novelty-check

| Work | Dataset | Retrieval | Text Reranker | Multimodal Reranker | No-Reranker Baseline | Answer Metrics | Latency | Threat to Novelty |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [Enhancing Document VQA Models via RAG](https://arxiv.org/abs/2508.18984) | MP-DocVQA, DUDE, InfographicVQA | Да | Частично / reranking component | Частично, через visual RAG variant | Да, concatenate-all-pages baseline | Да, ANLS | Compute/memory motivation, latency не центральна | **Частично** |
| [ViDoRe V3](https://arxiv.org/abs/2601.08620) | ViDoRe v3, 10 datasets | Да | Да, textual reranking | Скорее visual retrieval / hybrid context, не standalone multimodal reranker | Да / pipeline comparisons | Да | Частично через RAG evaluation complexity | **Частично** |
| [Qwen3-VL-Embedding / Qwen3-VL-Reranker](https://arxiv.org/abs/2601.04720) | MMEB-V2 и multimodal retrieval/ranking benchmarks | Да | Не основной фокус | Да, cross-encoder multimodal reranker | Нет в Document QA pipeline | Нет как DocQA answer generation | Deployment trade-offs, но не DocQA latency study | **Частично** |
| [MM-Embed](https://arxiv.org/abs/2411.02571) | M-BEIR, MTEB, CIRCO | Да | Не как Document QA baseline | Да, zero-shot MLLM reranking | Нет | Нет, retrieval metrics | Нет | Частично |
| [M3DocRAG](https://arxiv.org/abs/2411.04952) | M3DocVQA, MMLongBench-Doc, MP-DocVQA | Да | Нет отдельной controlled text reranker линии | Нет standalone reranker | Baselines есть, но не reranker-focused | Да | Efficiency discussed, latency не главный controlled axis | Нет / частично |
| [MHier-RAG / MMRAG-DocQA](https://arxiv.org/abs/2508.00579) | MMLongBench-Doc, LongDocURL | Да | LLM-based reranking, не text-vs-multimodal triad | Joint similarity + LLM reranking | Baselines есть | Да | Не главный axis | Частично |
| [MARDoc](https://arxiv.org/abs/2606.05749) | MMLongBench-Doc, DocBench | Да | Нет standalone text reranker | Нет standalone multimodal reranker | Agentic baselines | Да | Не central | Нет |
| [DocDancer](https://arxiv.org/abs/2601.05163) | MMLongBench-Doc, DocBench | Да, tools | Нет standalone text reranker | Нет standalone reranker | Agentic / RAG baselines | Да | Tool design analysis, не reranker latency | Нет |
| [MAGE-RAG](https://arxiv.org/abs/2606.15906) | LongDocURL, MMLongBench-Doc | Да | Нет | Evidence graph controller, не reranker | Да: Direct MLLM, Text RAG, Page Visual RAG, Graph/Agentic RAG | Да | Да, budget-performance curves | Нет / частично |
| [RAG-Anything](https://arxiv.org/abs/2510.12323) | Multimodal benchmarks | Да | Нет как isolated reranker | Graph/hybrid retrieval, не reranker | Baselines есть | Да | Не главный axis | Нет |
| [VisRAG](https://arxiv.org/abs/2410.10594) | Multi-modality document datasets | Да, vision-based retrieval | Нет | Нет standalone reranker | Text RAG baseline | Да | Efficiency discussed | Нет |
| [CMRAG](https://arxiv.org/abs/2509.02123) | Visual document QA benchmarks | Да, co-modality retrieval | Нет | Нет standalone reranker | Single-modality baselines | Да | Не central | Нет |
| [ColPali](https://arxiv.org/abs/2407.01449) | ViDoRe | Да, visual page retrieval | Нет | Late interaction retrieval, not QA reranking | Text retrieval baselines | Нет / retrieval only | Да, retrieval efficiency | Нет |
| [DOCBENCH](https://arxiv.org/abs/2407.10701) | DocBench | Не изолирован | Нет | Нет | Parse-then-read / document systems | Да, accuracy | Нет | Нет |
| [MMLongBench-Doc](https://arxiv.org/abs/2407.01523) | MMLongBench-Doc | Нет controlled RAG pipeline | Нет | Нет | LVLM/LLM baselines | Да, F1 | Нет | Нет |

### 2.1. Enhancing Document VQA Models via Retrieval-Augmented Generation

**Полное название:** Enhancing Document VQA Models via Retrieval-Augmented Generation
**Авторы:** Eric López, Artemis Llabrés, Ernest Valveny
**Год:** 2025
**Ссылка:** https://arxiv.org/abs/2508.18984
**Датасеты:** MP-DocVQA, DUDE, InfographicVQA.

Что совпадает с нашей работой:

- Document VQA / RAG pipeline.
- Retrieval перед answer generation.
- Есть reranking components.
- Есть answer-level evaluation через ANLS.
- Есть ablation, где retrieval и reranking названы основными источниками прироста.
- Есть сравнение с baseline, который обрабатывает все страницы.

Что отличается:

- Работа не делает центральным объектом именно мультимодальный reranking stage.
- Не найдена триада `No Reranker -> Text Reranker -> Multimodal Reranker` как основная экспериментальная линия.
- Датасеты отличаются от DocBench.
- Основной вывод сформулирован шире: польза RAG для Document VQA, а не controlled-анализ разных типов реранкинга.
- Latency / compute cost мотивируется, но не является таким же явно выделенным сравнением качества и задержки, как в нашей статье.

Угроза новизне: **частично**.
Как учитывать: обязательно включить в Related Work как ближайшую работу и подчеркнуть, что наша статья сужает фокус до контролируемого сравнения reranking modes.

### 2.2. ViDoRe V3

**Полное название:** ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios
**Авторы:** António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, Gautier Viaud
**Год:** 2026
**Ссылка:** https://arxiv.org/abs/2601.08620
**Датасет:** ViDoRe v3, 10 datasets, около 26k pages и 3,099 queries.

Что совпадает:

- Multimodal RAG over visually rich documents.
- Есть retrieval relevance, grounding и verified reference answers.
- Оцениваются retrieval and answer generation.
- В выводах указано, что visual retrievers сильнее textual, а late-interaction models и textual reranking улучшают performance.

Что отличается:

- Это benchmark / comprehensive evaluation, а не исследование конкретного reranking component.
- Reranking в найденном описании в основном textual reranking / late interaction, а не отдельная линия multimodal reranker vs text reranker vs no reranker.
- Latency как отдельный controlled trade-off не является центральной осью.

Угроза новизне: **частично**.
Как учитывать: использовать как сильное подтверждение мотивации: visual retrieval/reranking важны для multimodal RAG, но остается пространство для controlled reranker-stage study.

### 2.3. Qwen3-VL-Embedding and Qwen3-VL-Reranker

**Полное название:** Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking
**Авторы:** Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, Junyang Lin
**Год:** 2026
**Ссылка:** https://arxiv.org/abs/2601.04720
**Датасеты:** MMEB-V2 и другие multimodal retrieval/ranking benchmarks.

Что совпадает:

- Есть именно multimodal reranker.
- Qwen3-VL-Reranker выполняет fine-grained relevance estimation for query-document pairs.
- Поддерживаются text, images, document images, video.

Что отличается:

- Это model/report paper про embedding/reranker series, а не Document QA pipeline.
- Нет controlled Document QA comparison `No Reranker -> Text Reranker -> Multimodal Reranker`.
- Нет answer-level evaluation после VLM generation на DocBench.
- Не анализируется latency trade-off именно в Document QA pipeline.

Угроза новизне: **частично**.
Как учитывать: нельзя утверждать, что мультимодальные реранкеры как класс не изучены. Корректнее: они изучены на retrieval/ranking benchmarks, но их controlled-вклад в Document QA answer quality и latency исследован недостаточно.

### 2.4. MHier-RAG / MMRAG-DocQA

**Полное название:** MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning
**Также встречалось как:** MMRAG-DocQA / hierarchical multi-granularity retrieval line
**Авторы:** Ziyu Gong, Chengcheng Mai, Yihua Huang
**Год:** 2025
**Ссылка:** https://arxiv.org/abs/2508.00579
**Датасеты:** MMLongBench-Doc, LongDocURL.

Что совпадает:

- Multimodal long-context Document QA.
- Retrieval по multi-granularity evidence.
- Есть LLM-based re-ranking.
- Есть answer-level evaluation.

Что отличается:

- Главный вклад: hierarchical indexing, multi-granularity reasoning, cross-page dependencies.
- Reranking не изолирован как главный объект controlled study.
- Не проводится сравнение `No Reranker -> Text Reranker -> Multimodal Reranker`.
- Не DocBench.

Угроза новизне: **частично**.

### 2.5. MM-Embed

**Полное название:** MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs
**Авторы:** Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, Wei Ping
**Год:** 2024 / ICLR 2025
**Ссылка:** https://arxiv.org/abs/2411.02571
**Датасеты:** M-BEIR, MTEB, CIRCO.

Что совпадает:

- Multimodal retrieval.
- Zero-shot MLLM reranking.
- Есть идея prompt-and-reranking для refinement candidates.

Что отличается:

- Не Document QA over PDFs.
- Нет VLM answer generation после evidence selection.
- Нет no/text/multimodal reranker triad.
- Нет DocBench / MP-DocVQA style answer-level pipeline evaluation.

Угроза новизне: **частично**, только на уровне общей идеи multimodal reranking.

## 3. Works That Threaten Novelty

### Сильная частичная угроза

1. **Enhancing Document VQA Models via RAG**
   Угроза в том, что работа уже показывает: retrieval and reranking components могут давать основной прирост в Document VQA. Это близко к нашей мотивации. Отличие: наша работа делает controlled comparison типов реранкинга центральной линией и добавляет DocBench + latency trade-off.

2. **ViDoRe V3**
   Угроза в том, что benchmark оценивает современные RAG pipelines, verified answers, retrieval relevance, grounding, textual reranking и hybrid/visual contexts. Отличие: это не узкое исследование мультимодального reranking stage.

3. **Qwen3-VL-Embedding / Qwen3-VL-Reranker**
   Угроза в том, что существует специализированный multimodal reranker. Отличие: работа не про вклад reranker stage в Document QA answer quality.

### Средняя / слабая частичная угроза

4. **MHier-RAG / MMRAG-DocQA**
   Есть LLM-based re-ranking, но центральный вклад в hierarchical indexing и multi-granularity reasoning.

5. **MM-Embed**
   Есть MLLM reranking, но это universal multimodal retrieval, не Document QA.

### Работ, полностью закрывающих нашу постановку, не найдено

Не найдена работа, где одновременно есть:

- Document QA / PDF QA;
- page/fragment retrieval;
- no-reranker baseline;
- text reranker baseline;
- multimodal reranker baseline;
- answer-level metrics;
- latency analysis;
- ablation именно reranking stage.

## 4. Works That Support Our Motivation

### DOCBENCH

**DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems** вводит сам benchmark: 229 real documents, 1,102 questions, пять доменов и четыре типа вопросов. Работа показывает, что document reading systems остаются сложной задачей даже для сильных LLM-based systems. Она не исследует reranking, но поддерживает выбор датасета и важность Document QA.

Источник: https://arxiv.org/abs/2407.10701

### M3DocRAG

**M3DocRAG** показывает, что multi-modal retrieval важен для multi-page / multi-document DocVQA, особенно когда evidence находится в charts, figures и других визуальных элементах. Это поддерживает использование мультимодального retrieval и VLM generation в нашей постановке.

Источник: https://arxiv.org/abs/2411.04952

### VisRAG

**VisRAG** показывает пользу vision-based RAG: документные страницы индексируются как изображения, что снижает потери от OCR/parsing. Это подтверждает мотивацию использовать page images и мультимодальные candidate pools.

Источник: https://arxiv.org/abs/2410.10594

### ColPali

**ColPali** вводит visual document retrieval через VLM embeddings и late interaction на page-level tasks. Это поддерживает выбор визуальных retriever-ов как first-stage candidate generation, но не закрывает вопрос reranking-stage ablation.

Источник: https://arxiv.org/abs/2407.01449

### CMRAG

**CMRAG** подтверждает, что совместное использование text и image signals в visual document QA лучше, чем single-modality approaches. Это поддерживает нашу идею сравнивать text-only и multimodal reranking.

Источник: https://arxiv.org/abs/2509.02123

## 5. Works That Are Stronger but Solve a Different Problem

### DocDancer

**DocDancer** формулирует DocQA как agentic information-seeking problem и обучает open-source agent на synthetic trajectories. Он использует MMLongBench-Doc и DocBench, но не изолирует reranking как отдельный компонент. Это более сильная и масштабная agentic system, но другая научная задача.

Источник: https://arxiv.org/abs/2601.05163

### MARDoc

**MARDoc** строит memory-aware refinement agent framework с Explorer, Refiner и Reflector. Он работает на MMLongBench-Doc и DocBench и демонстрирует сильные результаты, но исследует structured memory / reflection, а не controlled reranking stage.

Источник: https://arxiv.org/abs/2606.05749

### MAGE-RAG

**MAGE-RAG** строит multigranular adaptive graph evidence framework. Работа явно анализирует budgets, graph/agentic evidence and context-noise control, но не является исследованием reranking stage. Это сильнее по архитектурной амбиции, но решает другую задачу.

Источник: https://arxiv.org/abs/2606.15906

### RAG-Anything

**RAG-Anything** предлагает dual-graph construction and cross-modal hybrid retrieval для all-in-one multimodal RAG. Это общая graph-based MRAG framework, а не Document QA reranking ablation.

Источник: https://arxiv.org/abs/2510.12323

### ViDoRAG

**ViDoRAG** использует GMM-based hybrid multimodal retrieval и iterative agent workflow with exploration, summarization and reflection. Это agentic retrieval/reasoning system, не controlled comparison reranking modes.

Источник: https://arxiv.org/abs/2502.18017

## 6. Final Novelty Verdict

### Вердикт

**Полного совпадения с нашей работой не найдено.**

Научная новизна в строгом смысле должна формулироваться как **новизна постановки controlled evaluation**, а не как изобретение нового типа реранкера или первой в мире multimodal reranking system.

Корректная формулировка вклада:

- не "мы предлагаем новый multimodal reranker";
- не "multimodal reranking ранее не исследовался";
- а "мы изолируем reranking stage в Document QA / Multimodal RAG pipeline и проводим контролируемое сравнение no-reranker, text reranker и multimodal reranker с answer-level metrics и latency analysis".

### Проверка исходного утверждения

Утверждение:

> Современные работы по Document QA и Multimodal RAG часто фокусируются на retrieval, agentic evidence search, memory/reflection или end-to-end systems. Однако мультимодальный reranking как отдельный controlled component с сравнением No Reranker, Text Reranker и Multimodal Reranker, answer-level metrics и latency analysis остаётся недостаточно изученным.

Оценка: **корректно, если добавить осторожность "насколько показывает проведенный обзор" и признать близкие работы**.

Рекомендуемая версия:

> Современные исследования Document QA и Multimodal RAG в основном развивают visual retrieval, multimodal indexing, agentic evidence search, structured memory или end-to-end frameworks. При этом, насколько показывает проведенный обзор, вклад именно reranking stage в Document QA pipeline реже анализируется как самостоятельный controlled component. Наша работа дополняет это направление сравнением `No Reranker`, `Text Reranker` и `Multimodal Reranker` при фиксированном pipeline, оценивая не только качество ответа, но и latency trade-off.

## 7. Recommended Wording for the Paper

### Для Introduction

```text
В отличие от работ, ориентированных на построение end-to-end Document QA систем,
visual retrieval или agentic evidence search, в данной работе основной объект
исследования намеренно сужен до этапа реранкинга. Мы рассматриваем retrieval,
evidence construction и VLM generation как компоненты окружения, позволяющие
измерить вклад реранкинга в итоговое качество ответа.
```

### Для Related Work

```text
Наиболее близкими к нашей постановке являются исследования Document VQA with RAG,
где анализируется влияние retrieval и reranking components на качество ответа.
Однако существующие работы чаще рассматривают RAG pipeline целиком, visual
retrieval, late interaction, graph evidence или agentic search. В нашей работе
реранкинг выделяется как самостоятельный controlled component: сравниваются
режимы без реранкинга, текстовый реранкинг и мультимодальный реранкинг при
одинаковом downstream VLM answer generation.
```

### Для Contributions

```text
Основной вклад работы состоит в контролируемой экспериментальной оценке
мультимодального реранкинга в Document QA pipeline. В отличие от end-to-end
agentic систем и retrieval-centric подходов, мы изолируем reranking stage и
оцениваем его влияние на answer-level metrics и latency.
```

### Чего лучше избегать

Не писать:

- "впервые предложен мультимодальный реранкер";
- "мультимодальный реранкинг ранее не изучался";
- "наша система превосходит SOTA на DocBench";
- "мы предлагаем новую архитектуру VLM/RAG".

Лучше писать:

- "controlled study";
- "reranking-stage ablation";
- "answer-level and latency trade-off";
- "multimodal reranking as a component of Document QA pipeline".

## 8. Recommended Wording for the Defense

### Короткая устная формулировка

```text
Моя работа не конкурирует напрямую с agentic системами вроде DocDancer или MARDoc.
Они решают задачу построения сильной end-to-end Document QA системы. Я исследую
более узкий компонент этого пайплайна: что дает этап реранкинга, если сравнить
отсутствие реранкинга, текстовый реранкинг и мультимодальный реранкинг при
одинаковой downstream генерации ответа.
```

### Если спросят про новизну

```text
Новизна не в создании нового VLM или нового класса реранкеров. Новизна в
контролируемой постановке эксперимента для Document QA: реранкинг рассматривается
как отдельный компонент, а не как скрытая часть retrieval или agentic search.
Мы показываем его вклад в итоговое качество ответа и отдельно анализируем цену
этого улучшения по latency.
```

### Если спросят про близкие работы

```text
Ближайшая работа - Enhancing Document VQA Models via RAG. Она показывает пользу
retrieval и reranking для Document VQA. Отличие моей работы в том, что я строю
экспериментальную линию именно вокруг реранкинга: No Reranker -> Text Reranker
-> Multimodal Reranker, на DocBench и с явным quality-latency анализом.
```

### Если спросят, почему результаты ниже agentic SOTA

```text
Agentic системы решают более широкую задачу и используют search/read tools,
memory, reflection или обучение trajectories. Цель моей работы другая: не
получить максимальный SOTA любой ценой, а понять вклад мультимодального
реранкинга как отдельного этапа pipeline и показать компромисс между качеством
и задержкой.
```

## Проверенные источники

- DOCBENCH: https://arxiv.org/abs/2407.10701
- Enhancing Document VQA Models via RAG: https://arxiv.org/abs/2508.18984
- ViDoRe V3: https://arxiv.org/abs/2601.08620
- Qwen3-VL-Embedding / Qwen3-VL-Reranker: https://arxiv.org/abs/2601.04720
- M3DocRAG: https://arxiv.org/abs/2411.04952
- MHier-RAG / MMRAG-DocQA line: https://arxiv.org/abs/2508.00579
- DocDancer: https://arxiv.org/abs/2601.05163
- MARDoc: https://arxiv.org/abs/2606.05749
- MAGE-RAG: https://arxiv.org/abs/2606.15906
- RAG-Anything: https://arxiv.org/abs/2510.12323
- ViDoRAG: https://arxiv.org/abs/2502.18017
- VisRAG: https://arxiv.org/abs/2410.10594
- CMRAG: https://arxiv.org/abs/2509.02123
- MM-Embed: https://arxiv.org/abs/2411.02571
- ColPali: https://arxiv.org/abs/2407.01449
- MMLongBench-Doc: https://arxiv.org/abs/2407.01523
