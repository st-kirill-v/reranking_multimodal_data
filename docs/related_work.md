# 3 Обзор литературы

В этом разделе рассматриваются работы, связанные с мультимодальным вопросно-ответным анализом документов, визуальным поиском по документам, мультимодальным RAG и reranking. В обзор включены только существующие статьи и официальные источники. Если конкретный результат или ограничение не указаны напрямую в статье или на странице источника, такое ограничение формулируется как отличие относительно нашей постановки задачи, а не как заявленная авторами слабая сторона работы.

## 3.1 Document Question Answering

Document Question Answering (DocQA) расширяет классическое reading comprehension на визуально структурированные документы, где текст встроен в макеты, таблицы, формы, рисунки и сканированные страницы. Ранние benchmark-и для document VQA, такие как **DocVQA**, сформулировали задачу ответа на вопросы на естественном языке по изображениям документов и показали, что модели должны понимать структуру документа, а не только читать плоский OCR-текст. Mathew et al. сообщают о большом разрыве между модельными baseline-ами и человеком: human accuracy составляет 94.36%, что показывает сложность layout- и структурного reasoning даже тогда, когда ответ визуально присутствует на странице [1].

Вторая линия работ сосредоточена на архитектурах для document understanding. **LayoutLMv3** объединяет text masking и image masking objectives для Document AI и применяется как к text-centric задачам, например form understanding и document VQA, так и к image-centric задачам, включая document classification и layout analysis [2]. **Donut** устраняет зависимость от OCR, формулируя document understanding как OCR-free image-to-sequence задачу и тем самым снижая стоимость OCR, его негибкость и распространение OCR-ошибок [3]. **Pix2Struct** обобщает visual language understanding за счёт pretraining image-to-text модели на screenshot parsing и показывает, что единая модель может переноситься между документами, иллюстрациями, UI-скриншотами и natural images [4]. Эти методы демонстрируют важность визуального моделирования документов, но в основном остаются model-centric и сами по себе не решают large-corpus retrieval, page selection и production-style RAG по множеству PDF-страниц.

Более новые benchmark-и выходят за рамки single-page или task-specific document QA. **MMLongBench-Doc** оценивает long-context document understanding на 130 длинных PDF и 1,062 экспертно размеченных вопросах, где evidence может находиться в тексте, изображениях, графиках, таблицах, layout-структурах и номерах страниц. Результаты показывают, что даже сильные LVLM испытывают трудности: GPT-4o достигает только 42.7 F1 в опубликованной оценке, а многие LVLM уступают LLM, которым подают lossy OCR text [5]. **DocBench** особенно близок к нашей экспериментальной постановке: он содержит 229 реальных PDF-документов и 1,102 вопроса из пяти доменов и четырёх типов вопросов, включая text-only, multimodal, metadata и unanswerable questions [6]. В нашей работе используется мультимодальное подмножество DocBench, а основной фокус сделан на retrieval, reranking, layout-aware выборе evidence и генерации ответа для table/text-heavy и figure/visual-heavy вопросов.

## 3.2 Мультимодальный retrieval

Multimodal retrieval изучает поиск релевантного evidence в условиях, когда запросы и документы могут включать текст, изображения или смешанный мультимодальный контент. В текстовом retrieval benchmark **BEIR** стал важной точкой отсчёта для zero-shot retrieval и показал, что BM25 остаётся сильным baseline-ом, тогда как reranking и late-interaction архитектуры дают высокое качество ценой большей вычислительной стоимости [7]. Это мотивирует включение BM25 и text reranker baseline-ов в нашу работу, однако text-only retrieval недостаточен для визуально насыщенных PDF-страниц.

Visual document retrieval напрямую индексирует отрендеренные страницы документов. **ColPali** является одним из ключевых современных подходов в этом направлении: он кодирует изображения страниц с помощью vision-language model и использует late interaction для retrieval. В работе также представлен benchmark ViDoRe и подчёркивается, что визуально богатые документы содержат информацию в layout, таблицах, рисунках, шрифтах и других признаках, которые не полностью сохраняются в text extraction pipelines [8]. Наши эксперименты с ColPali/ColVision следуют именно этой линии визуального retrieval.

За пределами document-specific retrieval работа **MM-Embed** изучает universal multimodal retrieval с использованием multimodal LLM. Авторы fine-tune MLLM retriever на 10 датасетах и 16 retrieval tasks, сообщают о state-of-the-art результате на M-BEIR и исследуют prompting MLLM как zero-shot reranker [9]. **Omni-Embed-Nemotron** расширяет multimodal retrieval на text, image, audio и video, исходя из того, что реальные RAG-корпуса включают PDF, презентации, видео и другой смешанный контент [10]. **ViDoRe v3** дополнительно расширяет оценку visual document retrieval на multilingual и multi-type multimodal RAG scenarios с retrieval relevance, bounding boxes и reference answers; его анализ показывает, что visual retrievers превосходят textual retrievers, а late-interaction models и textual reranking улучшают performance [11]. Эти выводы согласуются с нашим эмпирическим результатом: группа `Nemotron image` является самой сильной группой retriever-ов в наших экспериментах.

## 3.3 Multimodal RAG

Retrieval-Augmented Generation (RAG) связывает внешний retrieval с generation, чтобы снизить зависимость от параметрической памяти модели и улучшить grounding. Обзоры Gao et al. и Fan et al. структурируют RAG вокруг retrieval, augmentation и generation компонентов и подчёркивают ограничения, связанные с качеством retrieval, выбором context, hallucination и evaluation [12, 13]. В мультимодальной постановке эти проблемы становятся сложнее, поскольку релевантный evidence может быть распределён между page images, OCR, tables, captions и figures.

**MuRAG** является ранним мультимодальным retrieval-augmented generator для open QA по изображениям и тексту. Он использует multimodal memory и объединяет contrastive и generative training. На WebQA и MultimodalQA работа сообщает о state-of-the-art accuracy с абсолютным улучшением на 10-20% относительно предыдущих систем [14]. Однако MuRAG решает open QA по изображениям и тексту, а не visual PDF page retrieval с layout-aware crop selection.

Более новые multimodal RAG системы фокусируются на документах. **M2RAG** вводит benchmark для multimodal RAG в open-domain contexts, включая image captioning, multimodal QA, fact verification и image reranking, а также предлагает multimodal retrieval-augmented instruction tuning (MM-RAIT) для улучшения использования context [15]. **MHier-RAG** предлагает hierarchical indexing и multi-granularity retrieval для visual-rich document QA, мотивируя это cross-page fragmentation и inter-modal disconnection в мультимодальных длинных документах [16]. **RAG-Anything** предлагает all-in-one multimodal RAG framework, рассматривающий мультимодальный контент как взаимосвязанные knowledge entities и использующий dual-graph construction и cross-modal hybrid retrieval [17]. Наша работа отличается тем, что фокусируется не на новом общем framework, а на контролируемом экспериментальном сравнении retrieval, reranking, evidence packaging и VLM choices на мультимодальных вопросах DocBench.

## 3.4 Реранкинг в retrieval-системах

Reranking обычно используется как второй этап после эффективного retrieval. Cross-encoders и sequence-to-sequence rerankers могут моделировать богатые query-document interactions, но требуют больше вычислений. **MonoT5** адаптирует pretrained sequence-to-sequence model к document ranking, генерируя relevance labels и интерпретируя logits как relevance probabilities; модель показывает сильные результаты на MS MARCO и Robust04 [18]. **ColBERT** предлагает более эффективную альтернативу через late interaction: query и document representations кодируются раздельно и сравниваются через fine-grained token-level interaction, что снижает query-time cost при сохранении значительной части выразительности interaction-based ranking [19].

Text embedding и reranking models также являются важными baseline-ами. **BGE M3** вводит multilingual, multi-functionality, multi-granularity embedding model, поддерживающую dense, sparse и multi-vector retrieval, а также длинные входы до 8192 токенов [20]. Наши text-only ветки используют BM25 и BGE rerankers для построения сильного текстового baseline. В мультимодальной постановке reranking может быть visual-language based: **MM-Embed** явно исследует MLLM как zero-shot rerankers для multimodal retrieval и сообщает, что такой reranking улучшает сложные composed-image retrieval tasks [9]. **Qwen3-VL-Embedding and Qwen3-VL-Reranker** демонстрирует unified multimodal retrieval and ranking framework для text, image, document image и video inputs, включая cross-encoder reranking для fine-grained relevance estimation [21]. В наших экспериментах используются Nemotron VL reranking и text+image reranking, чтобы проверить этот же принцип второго этапа на DocBench.

**Исследовательский разрыв.** Существующие работы по реранкингу показывают, что второй этап ранжирования улучшает качество retrieval, однако остаётся недостаточно изученным вопрос о том, как именно этот этап влияет на итоговое качество генерации ответов в мультимодальном document QA. Многие исследования оценивают retrieval отдельно от генерации ответов, тогда как другие предлагают end-to-end framework-и, где вклад отдельных компонентов трудно изолировать. В результате в литературе редко проводится контролируемый анализ на уровне компонентов, который одновременно рассматривает retriever, reranker, формирование evidence, VLM-модель и latency constraints в рамках единого воспроизводимого DocBench pipeline. Именно этот разрыв мотивирует данное исследование: retrieval, reranking, layout-aware выбор evidence и генерация VLM рассматриваются как взаимосвязанные компоненты одной системы, влияющие как на качество ответов, так и на практическую стоимость inference.

## 3.5 Позиционирование работы

Несмотря на значительный прогресс в multimodal document understanding и multimodal RAG, сравнительно небольшое число работ рассматривает мультимодальный реранкинг как самостоятельный центральный объект исследования внутри document question answering framework. Большинство предыдущих исследований либо вводит новую архитектуру, либо улучшает retrieval, generation или evidence representation как отдельный компонент pipeline.

В отличие от большинства предыдущих работ, сосредоточенных преимущественно на архитектурах retrieval или end-to-end multimodal RAG frameworks, данное исследование рассматривает мультимодальный реранкинг как основной механизм выбора evidence. Retrieval, формирование evidence и генерация VLM анализируются прежде всего как факторы, влияющие на эффективность реранкинга.

Предыдущие работы устанавливают несколько важных фактов: document QA требует layout и visual reasoning; visual document retrieval может превосходить OCR-only retrieval; multimodal RAG должен объединять evidence из разных модальностей; reranking повышает качество retrieval ценой дополнительных вычислений. Данная работа находится на пересечении этих направлений, но её акцент уже: она исследует, как reranking меняет качество выбора evidence в мультимодальном document QA. Retriever, формирование evidence и VLM-модель рассматриваются не как самостоятельные цели статьи, а как окружение, в котором измеряется эффективность текстового и мультимодального реранкинга.

По сравнению с DocVQA, LayoutLMv3, Donut и Pix2Struct, данная работа не обучает новую модель document understanding. Вместо этого исследуется этап реранкинга, который соединяет кандидаты retrieval с генерацией ответа на основе VLM. По сравнению с ColPali, ViDoRe и MM-Embed, первичный retrieval используется не как самостоятельная цель сравнения, а как источник кандидатов для последующего анализа реранкинга. По сравнению с MuRAG, M2RAG, MHier-RAG и RAG-Anything, работа не предлагает новый general multimodal RAG framework, а фокусируется на контролируемых ablations линии No Reranker -> Text Reranker -> Multimodal Reranker.

Ближайший практический вопрос исследования можно сформулировать так: при фиксированном мультимодальном evaluation set DocBench когда достаточно обходиться без reranker, когда полезен text reranker и когда необходим multimodal reranker, использующий визуальные и текстовые признаки страницы? Поэтому работа дополняет benchmark-и и model papers именно через анализ этапа реранкинга: она связывает выбор reranker с качеством ответа, grounding и latency в одном pipeline.

Практическая значимость такой постановки заключается в явном анализе компромисса между качеством и latency именно для reranking. В реальных document QA системах мультимодальный reranker может улучшать выбор evidence, но становиться главным источником задержки. Поэтому наш анализ не ограничивается одним лучшим score: он сравнивает варианты без реранкинга, text rerankers и visual-language rerankers, чтобы определить, когда дополнительный мультимодальный реранкинг оправдывает свою вычислительную стоимость.

Основная цель исследования состоит не в предложении новой архитектуры retrieval, а в анализе условий, при которых мультимодальный реранкинг даёт измеримые преимущества по сравнению с текстовым реранкингом и baseline-ами без реранкинга.

Основные вклады работы состоят в следующем:

1. Масштабное эмпирическое исследование стратегий реранкинга для multimodal document question answering на DocBench.
2. Контролируемое сравнение режимов No Reranker, Text Reranker и Multimodal Reranker в единой схеме оценки.
3. Анализ того, как выбор retriever, формирование evidence и VLM-модель влияют на эффективность реранкинга.
4. Совместная оценка качества ответа и latency для анализа компромисса между качеством и скоростью мультимодального реранкинга.


## 3.6 Сравнение с существующими работами

| Работа | Основной фокус | Авторы / год | Статус публикации | Задача | Модели / метод | Датасеты | Основные результаты по источнику | Отличие от нашего подхода | Источник |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DocBench | Benchmark | Zou et al., 2024 | arXiv preprint / benchmark paper | LLM-based document reading systems | Proprietary systems и parse-then-read open LLM pipeline | 229 реальных PDF, 1,102 вопроса, пять доменов, четыре типа вопросов | Показывает заметный разрыв между существующими document reading systems и человеком | Мы используем мультимодальное подмножество для controlled reranking ablations | arXiv:2407.10701 |
| ColPali | Retrieval | Faysse et al., 2024; ICLR 2025 conference paper | Conference paper / arXiv source | Visual document retrieval | VLM page-image embeddings + late interaction | ViDoRe visual document retrieval benchmark | Превосходит современные document retrieval pipelines при упрощении indexing | Retrieval используется как генерация кандидатов для анализа reranking | arXiv:2407.01449 |
| MM-Embed | Retrieval + Reranking | Lin et al., 2024 | arXiv preprint | Universal multimodal retrieval | MLLM bi-encoder, modality-aware hard negatives, MLLM reranking | M-BEIR; MTEB; composed image retrieval | SOTA на M-BEIR; MLLM reranking улучшает сложный composed image retrieval более чем на 7 mAP@5 | Общий multimodal retrieval; мы изучаем reranking в DocBench PDF QA с layout-aware evidence и VLM-генерацией ответа | arXiv:2411.02571 |
| MuRAG | End-to-End MRAG | Chen et al., 2022 | arXiv preprint | Open QA по изображениям и тексту | Multimodal retrieval-augmented Transformer | WebQA, MultimodalQA | Сообщает о SOTA с абсолютными улучшениями 10-20% | Не предназначен для анализа reranking в PDF page retrieval, layout-aware crops или DocBench document reading | arXiv:2210.02928 |
| RAG-Anything | MRAG Framework | Guo et al., 2025 | arXiv preprint | General multimodal RAG framework | Dual-graph construction; cross-modal hybrid retrieval | Multimodal benchmarks | Сообщает о значительных улучшениях относительно SOTA methods | Framework-level работа; данная работа выполняет контролируемое сравнение вариантов реранкинга | arXiv:2510.12323 |
| MHier-RAG | Document QA Framework | Gong, Mai, Huang, 2025 | arXiv preprint | Visual-rich document QA | Hierarchical indexing, multi-granularity retrieval, LLM-based reranking | MMLongBench-Doc, LongDocURL | Сообщает о превосходстве над baseline-ами на public long-document QA datasets | Другая document QA постановка; наш фокус — reranking на мультимодальном подмножестве DocBench | arXiv:2508.00579 |
| BGE M3 | Text Embedding | Chen et al., 2024 | arXiv preprint | Multilingual, multifunctional text embeddings | Dense, sparse, multi-vector retrieval с self-knowledge distillation | Multilingual/cross-lingual retrieval tasks | SOTA на multilingual и cross-lingual retrieval tasks; поддержка длинных документов до 8192 токенов | Используется как контекст для текстовым реранкингом baselines; центральный объект — мультимодальный реранкинг | arXiv:2402.03216 |
| Qwen3-VL-Embedding / Qwen3-VL-Reranker | Multimodal Reranking | Li et al., 2026 | arXiv preprint | Multimodal retrieval and ranking | Qwen3-VL embedding и cross-encoder reranker для text/images/document images/video | Multimodal embedding benchmarks, включая MMEB-V2 | Сообщает о SOTA multimodal retrieval/ranking results; 8B занимает первое место на MMEB-V2 в отчёте | Опубликовано после наших основных экспериментов; релевантное направление для будущего развития мультимодального реранкинга | arXiv:2601.04720 |
| Наша работа | Multimodal Reranking | Этот репозиторий, 2026 | Experimental study | Разработка и оценка стратегии мультимодального реранкинга для document QA | No Reranker, Text Reranker, Multimodal Reranker в едином pipeline | Мультимодальное подмножество DocBench | Оценивает влияние реранкинга на итоговое качество ответа и latency | Дополняет model/benchmark papers, изолируя роль реранкинга при разных retriever, evidence и VLM settings | Repository results and configs |

Дополнительно, для быстрого позиционирования относительно ближайших направлений, таблица ниже показывает, какие аспекты покрываются каждой работой.

| Работа | Visual Retrieval | Reranking | Multimodal QA | DocBench | Анализ latency |
| ---- | ---------------- | --------- | ------------- | -------- | ---------------- |
| ColPali | Да | Late interaction retrieval; не основной reranking-анализ | Частично, через visual document retrieval | Нет | Частично |
| MM-Embed | Да | Да, MLLM reranking | Да | Нет | Ограниченно |
| MuRAG | Частично | Нет явного component-level reranking анализа | Да | Нет | Нет |
| RAG-Anything | Да | В рамках framework | Да | Нет | Ограниченно |
| MHier-RAG | Да | Да, LLM-based reranking | Да | Нет | Ограниченно |
| Наша работа | Да | Да, с ablation | Да | Да | Да |

## Список литературы

Годы публикации и доступность источников проверялись по указанным arXiv или официальным страницам. Работы, помеченные в разделе 3.6 как arXiv preprint, следует рассматривать как препринты, если в цитируемом источнике явно не указан peer-reviewed venue.

[1] Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar. *DocVQA: A Dataset for VQA on Document Images*. arXiv:2007.00398. https://arxiv.org/abs/2007.00398

[2] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking*. arXiv:2204.08387. https://arxiv.org/abs/2204.08387

[3] Geewook Kim et al. *OCR-free Document Understanding Transformer*. arXiv:2111.15664. https://arxiv.org/abs/2111.15664

[4] Kenton Lee et al. *Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding*. arXiv:2210.03347. https://arxiv.org/abs/2210.03347

[5] Yubo Ma et al. *MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations*. arXiv:2407.01523. https://arxiv.org/abs/2407.01523

[6] Anni Zou et al. *DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems*. arXiv:2407.10701. https://arxiv.org/abs/2407.10701

[7] Nandan Thakur et al. *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. arXiv:2104.08663. https://arxiv.org/abs/2104.08663

[8] Manuel Faysse et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. arXiv:2407.01449. https://arxiv.org/abs/2407.01449

[9] Sheng-Chieh Lin et al. *MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs*. arXiv:2411.02571. https://arxiv.org/abs/2411.02571

[10] Mengyao Xu et al. *Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video*. arXiv:2510.03458. https://arxiv.org/abs/2510.03458

[11] António Loison et al. *ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios*. arXiv:2601.08620. https://arxiv.org/abs/2601.08620

[12] Yunfan Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. https://arxiv.org/abs/2312.10997

[13] Wenqi Fan et al. *A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models*. arXiv:2405.06211. https://arxiv.org/abs/2405.06211

[14] Wenhu Chen et al. *MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text*. arXiv:2210.02928. https://arxiv.org/abs/2210.02928

[15] Zhenghao Liu et al. *Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts*. arXiv:2502.17297. https://arxiv.org/abs/2502.17297

[16] Ziyu Gong, Chengcheng Mai, Yihua Huang. *MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning*. arXiv:2508.00579. https://arxiv.org/abs/2508.00579

[17] Zirui Guo et al. *RAG-Anything: All-in-One RAG Framework*. arXiv:2510.12323. https://arxiv.org/abs/2510.12323

[18] Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin. *Document Ranking with a Pretrained Sequence-to-Sequence Model*. arXiv:2003.06713. https://arxiv.org/abs/2003.06713

[19] Omar Khattab, Matei Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. arXiv:2004.12832. https://arxiv.org/abs/2004.12832

[20] Jianlv Chen et al. *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216. https://arxiv.org/abs/2402.03216

[21] Mingxin Li et al. *Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking*. arXiv:2601.04720. https://arxiv.org/abs/2601.04720
