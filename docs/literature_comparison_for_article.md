# Сравнение с существующими работами

## 1. Позиционирование относительно существующих работ

Современные исследования Document QA и Multimodal RAG развиваются в нескольких направлениях. Одна линия работ строит benchmark-и для чтения сложных PDF-документов и оценки LLM-based document reading systems. Другая линия развивает visual document retrieval, мультимодальное извлечение evidence и retrieval-augmented generation для многостраничных документов. Более новые работы переходят к agentic document QA, где система использует инструменты поиска, чтения, memory, reflection и iterative evidence refinement.

Настоящая работа занимает более узкую и контролируемую позицию. Она не предлагает новый VLM, новый retriever или новую архитектуру reranker. Основной вклад состоит в controlled evaluation мультимодального реранкинга как отдельного компонента Document QA pipeline. В отличие от end-to-end и agentic систем, где вклад отдельных компонентов часто трудно изолировать, в данной работе фиксируется общий контур:

```text
Question -> Retrieval -> Reranking -> Evidence Construction -> Qwen3-VL -> Answer
```

и сравниваются режимы:

```text
No Reranker -> Text Reranker -> Multimodal Reranker -> Adaptive Reranking
```

Такое позиционирование позволяет оценить не только итоговое качество ответа, но и цену реранкинга по latency. Поэтому результаты работы следует рассматривать не как прямую заявку на SOTA относительно DocDancer, MARDoc или других agentic systems, а как компонентное исследование reranking stage внутри воспроизводимого Document QA pipeline.

## 2. Сравнение с DocBench

Работа **DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems** вводит датасет DocBench: 229 PDF-документов и 1,102 вопроса из нескольких доменов и типов задач. Эта работа формирует основу для оценки document reading systems, включая proprietary systems и parse-then-read pipelines. В ней не изолируется retrieval или reranking stage, а основной акцент сделан на сложности чтения документов и сравнении систем на уровне итоговой корректности ответа.

В нашей работе DocBench используется иначе. Мы берём мультимодальное подмножество из 308 вопросов и рассматриваем его как площадку для контролируемых ablation studies по реранкингу. Это меняет исследовательский вопрос: вместо оценки document reading system в целом анализируется, как разные варианты реранкинга влияют на answer-level качество и latency при фиксированном downstream VLM и сопоставимых стратегиях evidence construction.

Таким образом, DocBench является не конкурирующей работой, а источником benchmark-а и мотивацией. Он показывает, что document QA по реальным PDF остаётся сложной задачей; наша работа уточняет, какую роль в этом pipeline играет reranking stage.

## 3. Сравнение с Document VQA RAG работами

Ближайшей по постановке является работа **Enhancing Document VQA Models via Retrieval-Augmented Generation**. Она исследует retrieval-augmented generation для Document VQA, сравнивает text-centric и visual retrieval variants, использует answer-level метрику ANLS и показывает, что retrieval/reranking components могут давать основной вклад в качество. Эта работа частично пересекается с нашей мотивацией: обе работы рассматривают retrieval и reranking как важные элементы Document VQA pipeline, а не только саму генеративную модель.

Отличие состоит в том, что наша работа делает реранкинг центральным объектом исследования. Мы явно строим controlled comparison `No Reranker -> Text Reranker -> Multimodal Reranker` на мультимодальном подмножестве DocBench и дополнительно анализируем latency. В работе Enhancing Document VQA via RAG реранкинг важен как часть RAG pipeline, но не выделяется в виде самостоятельной линии сравнения с text-only и multimodal reranking на DocBench.

**M3DocRAG** также близок тематически: он исследует multi-modal retrieval для multi-page и multi-document DocVQA и показывает, что мультимодальный retrieval важен для документов с charts, figures и визуальными элементами. Однако основной фокус M3DocRAG находится на multimodal retrieval and RAG framework, а не на controlled ablation reranking stage. Наша работа использует похожую мотивацию, но задаёт более узкий вопрос: когда нужен reranker и какой тип reranking strategy даёт выигрыш в answer-level метриках.

**MMRAG-DocQA / MHier-RAG** использует hierarchical indexing, multi-granularity retrieval и LLM-based reranking для long-document QA. Эта линия показывает важность многоуровневого поиска evidence, но не проводит прямую триаду `No Reranker / Text Reranker / Multimodal Reranker` как основной объект исследования.

## 4. Сравнение с visual retrieval и multimodal retrieval работами

Работы **ColPali** и **ViDoRe V3** важны как основа visual document retrieval. ColPali показывает, что страницы документов можно эффективно индексировать как изображения с использованием vision-language embeddings и late interaction. ViDoRe V3 расширяет evaluation visual document retrieval и multimodal RAG, показывая, что visual retrievers и reranking могут улучшать работу в сложных real-world scenarios.

Эти работы поддерживают выбор визуального retrieval как первого этапа candidate generation. В наших экспериментах это согласуется с наблюдением, что Nemotron image retrieval и ColPali/ColVision дают сильные candidate pools для мультимодальных вопросов. Однако ColPali и ViDoRe V3 в первую очередь оценивают retrieval или benchmark-level RAG behavior, тогда как наша работа оценивает влияние reranking stage на финальный ответ после evidence construction и VLM generation.

Работа **MM-Embed** исследует universal multimodal retrieval и zero-shot MLLM reranking. Она показывает, что мультимодальный reranking может быть полезен в retrieval tasks, но не является Document QA исследованием на DocBench и не оценивает final answer generation. Поэтому MM-Embed полезен как обоснование model-level идеи multimodal reranking, но не закрывает нашу DocBench-based постановку.

**Qwen3-VL-Embedding / Qwen3-VL-Reranker** является прямой model-level работой по multimodal retrieval and ranking. Она релевантна как потенциально сильная future baseline для замены или дополнения Nemotron VL reranker. Однако эта работа не отвечает на наш основной вопрос: как multimodal reranker влияет на answer-level качество и latency в Document QA pipeline на DocBench.

## 5. Сравнение с agentic long-document QA системами

**DocDancer** и **MARDoc** являются наиболее сильными современными работами на DocBench-подобной long-document QA постановке. DocDancer формулирует document-grounded information seeking как agentic process с search/read tools и обучением на synthetic trajectories. MARDoc строит memory-aware refinement framework с Explorer, Refiner и Reflector, поддерживая структурированную память и iterative evidence refinement. Обе работы используют DocBench и демонстрируют сильные результаты по LasJ / judge-based evaluation.

Эти системы нельзя напрямую сравнивать с нашим Mean F1, поскольку различаются метрики, evaluation protocol, agentic setup, training regime и масштаб системы. Корректная формулировка состоит в следующем: в отличие от agentic systems, таких как DocDancer и MARDoc, наша работа изолирует reranking stage и анализирует его влияние при фиксированном Document QA pipeline. Их результаты показывают перспективность tool-augmented и memory-aware long-document QA, а наша работа дополняет эту линию компонентным анализом реранкинга.

**MAGE-RAG** и **RAG-Anything** также относятся к более общим graph-based или multimodal RAG frameworks. Они важны как future work: graph-based evidence representation может быть перспективна для связи страниц, таблиц, фигур и текстовых фрагментов. Однако эти работы решают задачу построения более общей evidence framework, а не controlled reranking-stage evaluation.

## 6. Расширенная сравнительная таблица

| Work | Task / Dataset | Main Focus | Retrieval | Reranking | Answer Metrics | Latency | Fine-Tuning | Relation to Our Work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DOCBENCH | DocBench, document reading | Benchmark для LLM-based document reading | Не изолирован | Нет | Да, accuracy / judge-based correctness | Нет | Нет | Источник датасета; не исследует reranking как компонент |
| Enhancing Document VQA via RAG | MP-DocVQA, DUDE, InfographicVQA | RAG для Document VQA | Да | Да, как часть RAG pipeline | Да, ANLS | Compute/memory discussed, latency не центральна | Нет / not training-centric | Ближайшая работа по retrieval/reranking gains, но не DocBench controlled reranking study |
| M3DocRAG | M3DocVQA, MMLongBench-Doc, MP-DocVQA | Multimodal retrieval для multi-page / multi-document DocVQA | Да | Evidence selection через retrieval | Да | Не основной axis | Использует pretrained components | Поддерживает важность multimodal retrieval, но не изолирует reranking |
| MMRAG-DocQA / MHier-RAG | MMLongBench-Doc, LongDocURL | Hierarchical indexing и multi-granularity retrieval | Да | Да, LLM-based reranking | Да | Не главный axis | Не central | Близко по pipeline, но focus на hierarchical retrieval, а не на triad reranking ablation |
| ViDoRe V3 | ViDoRe v3, 10 datasets | Benchmark для visual retrieval и multimodal RAG | Да | Да, textual reranking / late interaction | Да | Частично | Benchmark paper | Показывает пользу visual retrieval/reranking; не DocBench reranking-stage study |
| ColPali | ViDoRe | Visual document retrieval | Да | Late interaction как retrieval scoring | Нет final DocQA answer metrics | Retrieval efficiency | Да, VLM-based retriever training | Основа visual candidate generation; не Document QA reranking study |
| MM-Embed | M-BEIR, MTEB, CIRCO | Universal multimodal retrieval | Да | Да, zero-shot MLLM reranking | Нет Document QA answer metrics | Нет | Да | Поддерживает идею multimodal reranking, но вне DocBench / PDF QA |
| Qwen3-VL-Embedding / Reranker | MMEB-V2 and multimodal ranking benchmarks | Model-level multimodal retrieval/ranking | Да | Да, cross-encoder multimodal reranker | Нет DocBench answer generation | Deployment trade-offs | Да | Потенциальная future baseline для reranker replacement |
| DocDancer | MMLongBench-Doc, DocBench | Agentic document-grounded information seeking | Да, tools | Agentic evidence selection, not standalone reranker | Да, LasJ / Acc / F1 | Tool design analysis, не reranker latency | Да, 5k trajectories | Сильная agentic baseline; другая задача, не controlled reranking |
| MARDoc | MMLongBench-Doc, DocBench | Memory-aware refinement agent framework | Да, Explorer | Refiner/Reflector memory, not standalone reranker | Да, LasJ / Acc / F1 | Не центрально | Нет task-specific training | Близко по DocBench/Qwen3-VL, но исследует memory/reflection, не reranking |
| MAGE-RAG / RAG-Anything | LongDocURL, MMLongBench-Doc, multimodal benchmarks | Graph-based / all-in-one multimodal RAG | Да | Evidence graph / hybrid retrieval, not isolated reranker | Да | Частично | Не central | Future work для graph evidence; не закрывает reranking-stage ablation |

## 7. Компактная таблица для статьи

| Работа | Dataset | Основной фокус | Реранкинг как объект исследования | Анализ latency | Отличие нашей работы |
| --- | --- | --- | --- | --- | --- |
| DOCBENCH | DocBench | Benchmark для document reading systems | Нет | Нет | Используем мультимодальное подмножество DocBench для controlled reranking ablations |
| Enhancing Document VQA via RAG | MP-DocVQA, DUDE, InfographicVQA | RAG для Document VQA | Частично | Не центрально | У нас reranking stage является главным объектом и оценивается на DocBench |
| M3DocRAG | M3DocVQA, MMLongBench-Doc, MP-DocVQA | Multimodal retrieval для multi-page DocVQA | Нет standalone reranking | Не центрально | У нас фокус не на новом MRAG framework, а на сравнении режимов реранкинга |
| MMRAG-DocQA / MHier-RAG | MMLongBench-Doc, LongDocURL | Hierarchical retrieval и multi-granularity evidence | Частично, LLM-based reranking | Не центрально | У нас сравниваются No/Text/Multimodal reranking в фиксированном pipeline |
| ViDoRe V3 | ViDoRe v3 | Visual document retrieval и multimodal RAG evaluation | Частично | Частично | У нас answer-level DocBench evaluation и latency trade-off для reranking stage |
| DocDancer | MMLongBench-Doc, DocBench | Agentic search/read document QA | Нет standalone reranker | Не как reranker analysis | Agentic system; наша работа изолирует reranking stage |
| MARDoc | MMLongBench-Doc, DocBench | Memory/reflection для long-document QA | Нет standalone reranker | Не центрально | Memory-aware agentic framework; наша работа - controlled reranking study |
| Our Work | DocBench multimodal subset | Controlled evaluation мультимодального реранкинга | Да | Да | Сравнивает No Reranker, Text Reranker, Multimodal Reranker и Adaptive Reranking |

## 8. Внешние baseline и related work для статьи

Для обсуждения в статье наиболее полезно использовать следующие работы:

1. **DOCBENCH** - источник датасета и baseline document reading systems.
2. **Enhancing Document VQA Models via RAG** - ближайшая работа по Document VQA, retrieval и reranking gains.
3. **M3DocRAG** - сильная линия multimodal RAG для multi-page / multi-document DocVQA.
4. **MMRAG-DocQA / MHier-RAG** - hierarchical retrieval, multi-granularity evidence и LLM-based reranking.
5. **ViDoRe V3** - benchmark-level evidence, что visual retrievers и reranking важны для multimodal RAG.
6. **ColPali** - базовая работа по visual document retrieval.
7. **MM-Embed** - multimodal retrieval и MLLM reranking вне Document QA.
8. **Qwen3-VL-Embedding / Qwen3-VL-Reranker** - model-level multimodal reranking reference.
9. **DocDancer** - сильная agentic DocBench baseline, полезная для объяснения того, чем наша работа не является.
10. **MARDoc** - близкая DocBench + Qwen3-VL agentic evidence refinement baseline.

## 9. Чем отличается наша работа

Наша работа отличается от существующих исследований по четырём пунктам.

Во-первых, объект исследования ограничен reranking stage. Retrieval, evidence construction и VLM рассматриваются как окружение, через которое проявляется вклад реранкинга, а не как самостоятельные заявленные инновации.

Во-вторых, сравнение построено контролируемо: `No Reranker`, `Text Reranker`, `Multimodal Reranker` и `Adaptive Reranking` оцениваются в одном Document QA pipeline. Это позволяет интерпретировать различия как эффект reranking strategy, а не только как результат смены всей системы.

В-третьих, оценка проводится на уровне итогового ответа. Используются Mean F1, F1 > 0.5, Exact Match, анализ по типам `multimodal-t` и `multimodal-f`, а также latency. Это важно, потому что улучшение retrieval metrics само по себе не гарантирует улучшение answer quality.

В-четвёртых, работа явно анализирует quality/latency trade-off. Лучший мультимодальный реранкер достигает `Mean F1 = 0.7023` при latency `13.6441s`, а Adaptive Reranking сохраняет близкое качество (`0.7009`) при latency `9.7882s`. Fast Fusion baseline даёт `Mean F1 = 0.6575` при latency `2.5080s`. Дополнительный fusion pilot (`Mean F1 = 0.6587`, latency `7.84s`) следует рассматривать только как exploratory experiment без убедительного quality/latency преимущества.

## 10. Итоговый вывод

Наша работа актуальна, потому что современные Document QA / Multimodal RAG исследования часто фокусируются на retrieval, agentic search, memory, reasoning или end-to-end системах, тогда как вклад мультимодального реранкинга как отдельного компонента остаётся недостаточно изолированным. Работа закрывает эту нишу через controlled comparison `No Reranker`, `Text Reranker`, `Multimodal Reranker` и `Adaptive Reranking` с answer-level метриками и latency analysis на мультимодальном подмножестве DocBench.

Корректная формулировка вклада:

> В отличие от agentic systems, таких как DocDancer и MARDoc, наша работа изолирует reranking stage и анализирует его влияние при фиксированном Document QA pipeline.

Некорректная линия аргументации, которой следует избегать:

> Прямое численное сопоставление с DocDancer или MARDoc как с тем же самым типом системы.

Такое сравнение некорректно, поскольку у этих работ другие метрики, evaluation protocol, agentic setup и training regime.
