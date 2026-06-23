# Обзор научных работ, связанных с DocBench

Дата обзора: 2026-06-23.

## Критерии включения

В обзор включены только реально найденные источники, где **DocBench**:

- вводится как датасет;
- используется в экспериментах;
- либо явно упоминается как связанный benchmark, но не используется в собственной экспериментальной части.

Важно: `DocBench`, `Dr. DocBench` и `MMDocBench` - разные benchmark-и. Работы с похожими названиями вынесены отдельно и не считаются использованием исходного DocBench, если в статье не найдено прямого использования датасета Zou et al.

## Найденные работы, реально использующие DocBench

### 1. DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems

| Поле | Значение |
| --- | --- |
| Полное название | DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems |
| Авторы | Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, Dong Yu |
| Год | 2024 |
| Ссылки | arXiv: https://arxiv.org/abs/2407.10701; GitHub: https://github.com/Anni-Zou/DocBench |
| Использование DocBench | Оригинальная работа, вводит полный датасет |
| Полный или частичный DocBench | Полный: 229 PDF-документов, 1,102 вопроса, 5 доменов, 4 типа вопросов |
| RAG / MRAG | Не формулируется как MRAG-работа; оцениваются LLM-based document reading systems и parse-then-read pipeline |
| Reranking | Нет |
| Retrieval | Не выделен как отдельный controlled retrieval-компонент; некоторые сравниваемые системы могут иметь внутренние RAG/long-context механизмы |
| Модели | GPT-4, GPT-4o, Claude-3, KimiChat, GLM-4, Qwen-2.5, ERNIE-3.5; parse-then-read pipeline с GPT-4, GPT-3.5, ChatGLM3, Gemma, Mixtral, InternLM2, Llama, Yi, Phi-3, Command-R и др. |
| Fine-tuning | Нет, benchmark/evaluation paper |
| Основные метрики | Accuracy через GPT-4 evaluator; также проверяется согласие evaluator-а с human assessment |
| Основные результаты | Human baseline: 81.2 overall accuracy. Среди систем: KimiChat 70.9, GPT-4 API document system 69.8, parse-then-read GPT-4 67.9, Claude-3 67.6, GPT-4o 63.1. GPT-4 evaluator показал 98% agreement с human оценкой на проверочной выборке. |
| Идеи для нашего проекта | Использовать типы вопросов DocBench как ось анализа; явно разделять text-only, multimodal, metadata и unanswerable; показывать, что ошибки multimodal-вопросов часто связаны с location/extraction/calculation; сохранять оценку не только по общему score, но и по типам вопросов. |

Ключевые факты источника: DocBench принимает raw PDF и вопрос на вход, содержит 229 документов и 1,102 вопроса; dataset включает text-only, multimodal, metadata и unanswerable вопросы. В официальной оценке используется GPT-based binary correctness / Accuracy. GitHub-репозиторий содержит `evaluate.py`, `run.py`, prompts и download scripts.

### 2. DocDancer: Towards Agentic Document-Grounded Information Seeking

| Поле | Значение |
| --- | --- |
| Полное название | DocDancer: Towards Agentic Document-Grounded Information Seeking |
| Авторы | Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, Wentao Zhang |
| Год | 2026 |
| Ссылка | arXiv: https://arxiv.org/abs/2601.05163 |
| Использование DocBench | Используется как один из двух long-context document understanding benchmarks вместе с MMLongBench-Doc |
| Полный или частичный DocBench | В статье указано 229 real-world documents и 1,082 questions; это отличается от исходных 1,102 вопросов, поэтому вероятно используется фильтрованная/доступная версия или авторская нормализация split-а |
| RAG / MRAG | Да, в широком смысле agentic information-seeking DocQA; есть search/read tools и сравнение с RAG-based baselines |
| Reranking | Явного reranking-модуля нет; упор на agentic search/read и synthesis |
| Retrieval | Да: search tool для глобального поиска и read tool для локального чтения sections; используется document outline с текстом, изображениями и таблицами |
| Модели | Qwen3-30B-A3B-Thinking-2507, Qwen3-4B-Thinking-2507, GPT-4o, Gemini-2.5-Pro, GPT-5.2; baseline-ы: Naive VL, OCR-based GPT-4, VisRAG, ColPali, M3DocRAG, RAGAnything, Doc-React, MDocAgent, MACT, SimpleDoc, DocLens, DocAgent |
| Fine-tuning | Да. Fine-tune Qwen3-30B-A3B-Thinking-2507 и Qwen3-4B-Thinking-2507 на 5,000 synthetic agent trajectories |
| Основные метрики | Для MMLongBench-Doc: Accuracy, F1, LLM-as-Judge. Для DocBench: official evaluation procedure с GPT-4.1 judge; в таблице результат обозначен как LasJ |
| Основные результаты | На DocBench: DocDancer + GPT-5.2 достигает 85.5 LasJ; DocDancer + Qwen3-30B-A3B fine-tuned достигает 81.2; DocDancer + Qwen3-4B fine-tuned достигает 79.8; Human baseline указан как 81.2. |
| Идеи для нашего проекта | Использовать идею agentic evidence access как направление future work; добавить более явное различение search/read vs reranking; можно сравнить наш deterministic reranking pipeline с agentic search pipeline как более дорогим, но потенциально сильным подходом. |

DocDancer важен для позиционирования: он показывает, что свежие SOTA-подходы на DocBench смещаются от простого document reading к tool-augmented агентам и обучению поведения поиска. При этом работа не изолирует multimodal reranking как самостоятельный компонент, поэтому она не закрывает нашу исследовательскую нишу.

### 3. MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA

| Поле | Значение |
| --- | --- |
| Полное название | MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA |
| Авторы | Kaifeng Chen, Hongtao Liu, Qiyao Peng, Jian Yang, Yongqiang Liu, Xiaochen Zhang, Qing Yang |
| Год | 2026 |
| Ссылка | arXiv: https://arxiv.org/abs/2606.05749 |
| Использование DocBench | Используется как benchmark для multimodal long-document QA вместе с MMLongBench-Doc |
| Полный или частичный DocBench | Полный по описанию статьи: 229 documents и 1,102 questions |
| RAG / MRAG | Да, agentic multimodal long-document QA с iterative retrieval-reasoning |
| Reranking | Явного standalone reranker нет; есть Explorer, Refiner и Reflector, которые итеративно направляют retrieval и сжимают evidence |
| Retrieval | Да: Explorer retrieves multi-granularity evidence; outline служит unified index; используются textual и visual retrieval tools |
| Модели | Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-8B-Instruct; baseline-ы включают Vanilla VLM, DocSeeker, Docopilot, SV-RAG, VisRAG, BookRAG, MoLoRAG, M3DocRAG, RAGAnything, Doc-React, MDocAgent, Chain-of-Agent, SLEUTH, MACT, DocAgent, DocDancer |
| Fine-tuning | Нет task-specific training; авторы подчёркивают performance без task-specific training |
| Основные метрики | Для MMLongBench-Doc: Accuracy и F1. Для DocBench: official prompts with GPT-4o evaluator; в таблице DocBench metric обозначена как LasJ |
| Основные результаты | На DocBench: MARDoc + Qwen3-VL-30B-A3B-Instruct достигает 82.1 LasJ; MARDoc + Qwen3-VL-8B-Instruct достигает 72.3. Human baseline указан как 81.2. |
| Идеи для нашего проекта | Взять как аргумент, что retrieval сам по себе недостаточен: требуется компактное и точное evidence state. Для future work можно предложить memory-aware reranking/evidence selection: reranker не только сортирует страницы, но и поддерживает structured evidence memory. |

MARDoc особенно близок по тематике к нашему проекту, потому что работает с multimodal long-document QA, Qwen3-VL и DocBench. Главное отличие: MARDoc - agentic framework с memory/reflection loop, а наша работа - controlled ablation study реранкинга `No Reranker -> Text Reranker -> Multimodal Reranker` с явным анализом latency.

## Работы, связанные с DocBench косвенно или упомянутые как baseline

Следующие методы появляются в таблицах DocDancer/MARDoc как baseline-ы для long-document QA или RAG, но по найденным источникам не подтверждено, что их оригинальные статьи сами проводили эксперименты на DocBench:

| Работа / метод | Статус относительно DocBench | Как учитывать |
| --- | --- | --- |
| VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents | В оригинальной arXiv-аннотации говорится о vision-based RAG для multimodality documents и 20-40% end-to-end gain over traditional text-based RAG, но прямое использование DocBench в найденном источнике не подтверждено | Упоминать как релевантный MRAG baseline / visual retrieval-RAG направление, но не как DocBench paper |
| ColPali: Efficient Document Retrieval with Vision Language Models | Оригинальная работа вводит visual document retrieval и ViDoRe; прямое использование DocBench не найдено | Использовать как источник visual retrieval идеи для candidate generation, но не как DocBench evaluation |
| SV-RAG: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding | Релевантная long-document RAG работа; прямое использование DocBench в найденном источнике не подтверждено | Указывать как RAG baseline, если ссылаемся через MARDoc table, но не как самостоятельную DocBench-работу |
| RAGAnything / RAG-Anything | Появляется как baseline в DocDancer/MARDoc; прямое использование DocBench в найденном источнике не подтверждено | Упоминать как general multimodal RAG framework, а не как DocBench-specific study |
| BookRAG | В MARDoc указан как RAG-based baseline; прямое использование DocBench в оригинальной найденной аннотации не подтверждено | Учитывать только как baseline, reported in MARDoc |
| M3DocRAG, MoLoRAG, Doc-React, MDocAgent, MACT, SLEUTH, DocAgent, DocLens | Появляются в таблицах DocDancer/MARDoc; отдельное подтверждение прямого использования DocBench в оригинальных источниках не найдено в рамках текущего поиска | Не использовать как доказательство SOTA on DocBench без ссылки на DocDancer/MARDoc |

## Похожие по названию benchmark-и, но не исходный DocBench

| Работа | Почему не включена как DocBench-use |
| --- | --- |
| Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing | Это отдельный benchmark для expert-level document parsing; он содержит 4,514 annotated pages и 65k annotations. Это не DocBench Zou et al. |
| MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding | Это отдельный benchmark для fine-grained visual document understanding с 4,338 QA pairs и supporting regions. Это не DocBench Zou et al. |

## State of the Art on DocBench

### Какие подходы доминируют

На раннем этапе DocBench использовался для оценки **LLM-based document reading systems**: proprietary systems, document upload interfaces и parse-then-read pipelines. В более новых работах 2026 года заметен сдвиг к **agentic document QA**: система не просто получает весь документ или извлечённый текст, а использует инструменты поиска, чтения, refinement и reflection.

Три доминирующих направления:

1. **Document reading systems**: загрузка PDF, parsing, long-context LLM/VLM inference.
2. **RAG / multimodal RAG baselines**: retrieval relevant pages/chunks before generation.
3. **Agentic DocQA**: search/read tools, iterative exploration, structured memory, reflection.

### Какие модели используются чаще всего

В DocBench paper используются GPT-4, GPT-4o, Claude-3, KimiChat, GLM-4, Qwen-2.5, ERNIE-3.5 и open-source LLMs в parse-then-read pipeline.

В более новых DocBench-экспериментах появляются:

- Qwen3-VL-30B-A3B-Instruct;
- Qwen3-VL-8B-Instruct;
- Qwen3-30B-A3B-Thinking;
- Qwen3-4B-Thinking;
- GPT-4o, Gemini-2.5-Pro, GPT-5.2;
- agentic baselines типа DocAgent / DocDancer / MARDoc.

### Используют ли мультимодальный реранкинг

Найденные DocBench-работы **не делают мультимодальный реранкинг центральным объектом исследования**.

- DocBench paper оценивает systems end-to-end, без отдельного reranker.
- DocDancer использует agentic search/read и training trajectories, но не выделяет reranking как отдельную controlled component.
- MARDoc использует Explorer/Refiner/Reflector и structured memory, но это не standalone reranking evaluation.

Это важно для нашей статьи: существующие DocBench-работы показывают важность поиска evidence и agentic reasoning, но не дают систематического сравнения `No Reranker`, `Text Reranker`, `Multimodal Reranker`.

### Чем наша работа отличается

Наша работа отличается от найденных DocBench-исследований тем, что:

- использует мультимодальное подмножество DocBench как площадку для **controlled reranking ablations**;
- сравнивает `No Reranker`, `Text Reranker` и `Multimodal Reranker` в одном pipeline;
- анализирует не только answer quality, но и latency;
- рассматривает retrieval и VLM не как самостоятельный benchmark, а как окружение, влияющее на эффективность реранкинга;
- показывает, когда multimodal reranking оправдывает вычислительную стоимость.

Практическое позиционирование:

> DocBench и агентные работы показывают, что long-document multimodal QA требует точного evidence localization. Наша работа уточняет этот вывод на уровне компонента: именно multimodal reranking между retrieval и VLM generation является ключевой точкой precision-oriented evidence selection.

## Источники

1. Anni Zou et al. **DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems**. arXiv:2407.10701. https://arxiv.org/abs/2407.10701
2. Official DocBench GitHub repository. https://github.com/Anni-Zou/DocBench
3. Qintong Zhang et al. **DocDancer: Towards Agentic Document-Grounded Information Seeking**. arXiv:2601.05163. https://arxiv.org/abs/2601.05163
4. Kaifeng Chen et al. **MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA**. arXiv:2606.05749. https://arxiv.org/abs/2606.05749
5. Shi Yu et al. **VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents**. arXiv:2410.10594. https://arxiv.org/abs/2410.10594
6. Manuel Faysse et al. **ColPali: Efficient Document Retrieval with Vision Language Models**. arXiv:2407.01449. https://arxiv.org/abs/2407.01449
7. Minglai Yang et al. **Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing**. arXiv:2606.01393. https://arxiv.org/abs/2606.01393
8. Fengbin Zhu et al. **MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding**. arXiv:2410.21311. https://arxiv.org/abs/2410.21311
