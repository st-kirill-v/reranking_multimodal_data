# Experiments Section Notes

This note documents the sources and interpretation used to prepare `docs/experiments.md`.

## Tables and Artifacts Used

- `reports/experiment_summary/main_table.csv`
- `reports/experiment_summary/component_aggregation.csv`
- `reports/experiment_summary/reranker_ablation.csv`
- `reports/experiment_summary/recommendations.csv`
- `reports/experiment_summary/paper_sections/paper_section_findings.md`
- `reports/experiment_summary/paper_sections/5_4_reranker_ablation_summary.csv`
- `reports/tables/`
- `results/`

The section uses only reported experiment results from these artifacts. No new metrics, models, datasets, or experiments were introduced.

## Key Results Used

- Best overall quality: `image_text_full_308_nemotron_qwen3vl30b`, reported as Nemotron full image+text + VL reranker + Qwen3-VL-30B, with Mean F1 0.7023, F1 > 0.5 0.7565, Exact Match 0.1201, and latency 13.6441 seconds.
- Best fast option: `image_text_fusion_308_nemotron_no_reranker_qwen3vl30b`, reported as Fusion Nemotron no image reranker + Qwen3-VL-30B, with Mean F1 0.6575 and latency 2.5080 seconds.
- Best text-only baseline: `text_reranker_308_bge_large_qwen3vl30b`, reported as BM25 + BGE-reranker-large + Qwen3-VL-30B, with Mean F1 0.5497 and latency 1.2344 seconds.
- Average paired reranker gain: +0.0428 Mean F1.
- Average paired reranker latency cost: +5.8355 seconds.
- Best average retriever group: Nemotron image retrieval, with average Mean F1 0.6791.
- Best average reranker group: Nemotron VL reranker, with average Mean F1 0.6787.
- Best evidence group: full page + layout crop, with average Mean F1 0.6629.

## Main Paper-Level Conclusion

The experiments support the paper's central argument that multimodal reranking improves document QA quality under controlled candidate-generation, evidence-construction, and VLM settings. The strongest results are obtained when visual candidate generation, multimodal reranking, layout-aware evidence selection, and Qwen3-VL-30B are combined.

The section also emphasizes that multimodal reranking has a meaningful latency cost. Therefore, the practical conclusion is not that multimodal reranking should always be used, but that it is most valuable when answer quality is prioritized and when the question depends on visual layout, tables, figures, or visually ambiguous document pages.

## Relation to the Project Topic

The section is structured around the main comparison line:

```text
No Reranker
-> Text Reranker
-> Multimodal Reranker
```

This directly supports the project topic, "Development of a Multimodal Reranking Algorithm for Document QA." Candidate generation, evidence construction, and VLM backbone are treated as factors that influence reranking effectiveness, not as independent benchmarks.
