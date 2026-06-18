# Section 5 Experiments: Paper-Facing Analysis

This file is generated from `reports/tables/paper_multimodal_308.csv` and is intended as direct material for the Experiments section.

## 5.3 Retriever Analysis

The strongest retriever group is **Nemotron image**, with average Mean F1=0.6791 and best single-run Mean F1=0.7023 from `image_text_full_308_nemotron_qwen3vl30b`. This supports using visually grounded retrieval as the first stage for multimodal DocBench questions.

## 5.4 Reranker Analysis

Aggregating by reranker, **Nemotron VL reranker** has the highest average Mean F1 (0.6787) among reranker groups. Across paired ablations, reranking improves Mean F1 by 0.0428 on average, while adding 5.8355 seconds of latency on average.
The largest quality gain is observed for **Text evidence BGE-large reranker** (Delta F1=0.0930), whereas the largest latency cost is **Nemotron full image+text Qwen30B text-image reranker** (Delta latency=10.2179s).

## 5.5 VLM Analysis

The best VLM group is **Qwen3-VL-30B**, with average Mean F1=0.6074 and average latency=5.3796s. Its best experiment is `image_text_full_308_nemotron_qwen3vl30b`.

## 5.6 Modality Analysis

The best MM-F result is obtained by `multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b` with MM-F F1=0.6620. The smallest MM-T/MM-F gap is `text_reranker_308_bge_large_qwen3vl30b` (gap=0.0156), while the largest gap is `text_evidence_encoder_308_bge_base_en_v1_5_bge_reranker_large_qwen3vl30b` (gap=0.1107).
Overall, MM-T scores are generally higher than MM-F scores, indicating that table/text-heavy questions benefit more from text evidence and layout-aware crops than figure/visual-heavy questions.

## 5.7 Quality-Speed Trade-off

The best-quality configuration is `image_text_full_308_nemotron_qwen3vl30b` (Mean F1=0.7023, latency=13.6441s). The fastest configuration above Mean F1 >= 0.65 is `image_text_fusion_308_nemotron_no_reranker_qwen3vl30b` (Mean F1=0.6575, latency=2.5080s). The balanced score criterion selects `text_reranker_308_no_reranker_qwen3vl30b`.

## Generated Tables

- `5_3_retriever_analysis.csv`
- `5_4_reranker_aggregation.csv`
- `5_4_reranker_ablation.csv`
- `5_4_reranker_ablation_summary.csv`
- `5_5_vlm_analysis.csv`
- `5_6_modality_analysis.csv`
- `5_6_modality_by_vlm.csv`
- `5_7_quality_speed_recommendations.csv`

## Generated Figures

- `figures/5_3_retriever_mean_f1.png`
- `figures/5_4_reranker_delta_f1.png`
- `figures/5_4_reranker_delta_latency.png`
- `figures/5_5_vlm_mean_f1.png`
- `figures/5_6_modality_top10_mmf.png`
- `figures/5_7_quality_speed_tradeoff.png`
