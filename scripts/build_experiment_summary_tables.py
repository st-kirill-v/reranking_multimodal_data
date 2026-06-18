from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # Plotting is optional for thin local environments.
    plt = None
    sns = None

PLOT_WARNING_EMITTED = False


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "reports" / "tables" / "paper_multimodal_308.csv"
OUT_DIR = ROOT / "reports" / "tables"
SUMMARY_DIR = ROOT / "reports" / "experiment_summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
PAPER_SECTIONS_DIR = SUMMARY_DIR / "paper_sections"
PAPER_FIGURES_DIR = PAPER_SECTIONS_DIR / "figures"


METRIC_COLUMNS = [
    "mean_f1",
    "f1_gt_0_5",
    "exact_match",
    "multimodal_t_mean_f1",
    "multimodal_f_mean_f1",
    "doc_hit_at_1",
    "page_hit_at_1",
    "latency_mean",
]


DISPLAY_COLUMNS = [
    ("rank", "Rank"),
    ("short_name", "Experiment"),
    ("family", "Family"),
    ("retriever_short", "Retriever"),
    ("reranker_short", "Reranker"),
    ("evidence_short", "Evidence"),
    ("vlm_short", "VLM"),
    ("mean_f1", "Mean F1"),
    ("f1_gt_0_5", "F1>0.5"),
    ("multimodal_t_mean_f1", "MM-T F1"),
    ("multimodal_f_mean_f1", "MM-F F1"),
    ("latency_mean", "Latency, s"),
    ("comment", "Comment"),
    ("results_dir", "Results"),
]


ABLATION_COLUMNS = [
    ("comparison", "Comparison"),
    ("no_reranker", "No reranker"),
    ("with_reranker", "With reranker"),
    ("mean_f1_no", "Mean F1 no"),
    ("mean_f1_with", "Mean F1 with"),
    ("delta_mean_f1", "Delta F1"),
    ("latency_no", "Latency no"),
    ("latency_with", "Latency with"),
    ("delta_latency", "Delta latency"),
    ("verdict", "Verdict"),
]


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for col in METRIC_COLUMNS:
                try:
                    row[col] = float(row[col])
                except (TypeError, ValueError):
                    row[col] = None
            rows.append(row)
    return rows


def clean_name(experiment: str) -> str:
    replacements = {
        "docbench_hybrid_bm25_tools_qwen3vl30b": "ColPali + VL reranker + Qwen30B",
        "docbench_qwen3vl30b_no_reranker": "ColPali no reranker + Qwen30B",
        "multimodal_308_with_reranker": "ColPali + VL reranker + Qwen8B",
        "multimodal_308_no_reranker": "ColPali no reranker + Qwen8B",
        "multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b": "Nemotron image + VL reranker + Qwen30B",
        "multimodal_308_nemotron_image_retriever_qwen3vl30b": "Nemotron image no reranker + Qwen30B",
        "image_text_fusion_308_colpali_qwen3vl30b": "Fusion ColPali + VL reranker + Qwen30B",
        "image_text_fusion_308_nemotron_qwen3vl30b": "Fusion Nemotron + VL reranker + Qwen30B",
        "image_text_fusion_308_nemotron_no_reranker_qwen3vl30b": "Fusion Nemotron no image reranker + Qwen30B",
        "image_text_input_308_nemotron_qwen3vl30b": "Nemotron image+text input + VL reranker + Qwen30B",
        "image_text_full_308_nemotron_qwen3vl30b": "Nemotron full image+text + VL reranker + Qwen30B",
        "image_text_full_308_nemotron_no_reranker_qwen3vl30b": "Nemotron full image+text no reranker + Qwen30B",
        "text_reranker_308_bge_base_qwen3vl30b": "BM25 + BGE-reranker-base + Qwen30B",
        "text_reranker_308_bge_large_qwen3vl30b": "BM25 + BGE-reranker-large + Qwen30B",
        "text_reranker_308_jina_qwen3vl30b": "BM25 + Jina reranker + Qwen30B",
        "text_reranker_308_minilm_qwen3vl30b": "BM25 + MiniLM reranker + Qwen30B",
        "text_reranker_308_no_reranker_qwen3vl30b": "BM25 no reranker + Qwen30B",
        "text_evidence_encoder_308_bge_base_en_v1_5_bge_reranker_large_qwen3vl30b": "BGE-base encoder + BGE-large reranker + Qwen30B",
        "text_evidence_encoder_308_bge_large_en_v1_5_bge_reranker_large_qwen3vl30b": "BGE-large encoder + BGE-large reranker + Qwen30B",
        "text_evidence_encoder_308_bge_large_en_v1_5_no_reranker_qwen3vl30b": "BGE-large encoder no reranker + Qwen30B",
    }
    return replacements.get(experiment, experiment)


def family(row: dict[str, object]) -> str:
    exp = str(row["experiment"])
    if exp.startswith("image_text_full"):
        return "Full image+text"
    if exp.startswith("image_text_input"):
        return "Image+text input"
    if exp.startswith("image_text_fusion"):
        return "Image+text fusion"
    if exp.startswith("text_evidence_encoder"):
        return "Text evidence encoder"
    if exp.startswith("text_reranker"):
        return "BM25 text reranking"
    if "nemotron_image_retriever" in exp:
        return "Nemotron image retriever"
    if "qwen3vl30b" in exp or exp.startswith("docbench"):
        return "ColPali visual Qwen30B"
    return "ColPali visual Qwen8B"


def shorten_retriever(value: str) -> str:
    if "nemotron_image" in value:
        return "Nemotron image"
    if "colpali" in value or "colvision" in value:
        return "ColPali/ColVision"
    if "text_encoder" in value:
        if "bge-large" in value:
            return "BGE-large text encoder"
        if "bge-base" in value:
            return "BGE-base text encoder"
        return "Text encoder"
    if "text_page_bm25" in value:
        return "BM25 page_text"
    return value


def shorten_reranker(value: str) -> str:
    if value.startswith("none") or value == "none":
        return "None"
    if "llama-nemotron-rerank" in value:
        return "Nemotron VL reranker"
    if "bge-reranker-large" in value:
        return "BGE-reranker-large"
    if "bge-reranker-base" in value:
        return "BGE-reranker-base"
    if "jina-reranker" in value:
        return "Jina reranker"
    if "MiniLM" in value:
        return "MiniLM reranker"
    return value


def shorten_evidence(value: str) -> str:
    if "ocr+page_text+caption+table_text" in value:
        return "OCR + page_text + captions + table_text"
    if "layout_aware_v2" in value:
        return "full page + layout crop"
    if value == "page_text":
        return "page_text"
    return value


def shorten_vlm(value: str) -> str:
    if "qwen3-vl-30b" in value:
        return "Qwen3-VL-30B"
    if "Qwen3-VL-8B" in value:
        return "Qwen3-VL-8B"
    return value


def comment(row: dict[str, object]) -> str:
    exp = str(row["experiment"])
    if exp == "multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b":
        return "Former best image-only result; strong baseline."
    if exp == "image_text_input_308_nemotron_qwen3vl30b":
        return "Previous best; text is added to VLM input."
    if exp == "image_text_full_308_nemotron_qwen3vl30b":
        return "Best overall result; text is used in reranker and VLM input."
    if exp == "image_text_full_308_nemotron_no_reranker_qwen3vl30b":
        return "Fast full image+text ablation; text remains in VLM input, reranker removed."
    if exp == "image_text_fusion_308_nemotron_no_reranker_qwen3vl30b":
        return "Very fast; fusion partly compensates for removing image reranker."
    if exp == "image_text_fusion_308_nemotron_qwen3vl30b":
        return "Best fusion result; faster than strongest visual baseline but lower F1."
    if exp.startswith("text_evidence_encoder"):
        return "Advisor text-evidence branch; useful ablation, not main method."
    if exp.startswith("text_reranker"):
        return "Text-only baseline branch."
    if exp.endswith("no_reranker") or "no_reranker" in exp:
        return "Speed ablation."
    return "Reference baseline."


def enrich(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda r: r["mean_f1"] or -1, reverse=True)
    enriched = []
    for rank, row in enumerate(ranked, 1):
        new_row = dict(row)
        new_row["rank"] = rank
        new_row["short_name"] = clean_name(str(row["experiment"]))
        new_row["family"] = family(row)
        new_row["retriever_short"] = shorten_retriever(str(row["retriever"]))
        new_row["reranker_short"] = shorten_reranker(str(row["reranker"]))
        new_row["evidence_short"] = shorten_evidence(str(row["evidence"]))
        new_row["vlm_short"] = shorten_vlm(str(row["vlm"]))
        new_row["comment"] = comment(row)
        enriched.append(new_row)
    return enriched


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return value if isinstance(value, float) else default


def rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    return value


def numeric_mean(values: list[object]) -> float | None:
    nums = [value for value in values if isinstance(value, float)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def enrich_analysis_fields(rows: list[dict[str, object]]) -> None:
    pareto = set(compute_pareto_experiments(rows))
    for row in rows:
        row["mm_gap"] = metric(row, "multimodal_t_mean_f1") - metric(row, "multimodal_f_mean_f1")
        row["balanced_score"] = metric(row, "mean_f1") / math.log1p(
            max(metric(row, "latency_mean"), 0.0)
        )
        row["pareto_frontier"] = str(row["experiment"]) in pareto


def compute_pareto_experiments(rows: list[dict[str, object]]) -> list[str]:
    frontier: list[str] = []
    for row in rows:
        exp = str(row.get("experiment", ""))
        f1 = metric(row, "mean_f1", -1.0)
        latency = metric(row, "latency_mean", 999999.0)
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_f1 = metric(other, "mean_f1", -1.0)
            other_latency = metric(other, "latency_mean", 999999.0)
            if (
                other_f1 >= f1
                and other_latency <= latency
                and (other_f1 > f1 or other_latency < latency)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(exp)
    return frontier


MAIN_ANALYSIS_COLUMNS = [
    ("rank", "rank"),
    ("experiment", "experiment"),
    ("short_name", "short_name"),
    ("family", "family"),
    ("retriever_short", "retriever"),
    ("reranker_short", "reranker"),
    ("evidence_short", "evidence"),
    ("vlm_short", "vlm"),
    ("mean_f1", "mean_f1"),
    ("f1_gt_0_5", "f1_gt_0_5"),
    ("exact_match", "exact_match"),
    ("multimodal_t_mean_f1", "multimodal_t_mean_f1"),
    ("multimodal_f_mean_f1", "multimodal_f_mean_f1"),
    ("mm_gap", "mm_gap"),
    ("doc_hit_at_1", "doc_hit_at_1"),
    ("page_hit_at_1", "page_hit_at_1"),
    ("latency_mean", "latency_mean"),
    ("balanced_score", "balanced_score"),
    ("pareto_frontier", "pareto_frontier"),
    ("results_dir", "results_dir"),
]


COMPONENT_COLUMNS = [
    ("component", "component"),
    ("value", "value"),
    ("count_experiments", "count_experiments"),
    ("mean_mean_f1", "mean_mean_f1"),
    ("max_mean_f1", "max_mean_f1"),
    ("mean_f1_gt_0_5", "mean_f1_gt_0_5"),
    ("mean_mm_t_f1", "mean_mm_t_f1"),
    ("mean_mm_f_f1", "mean_mm_f_f1"),
    ("mean_latency", "mean_latency"),
    ("best_experiment", "best_experiment"),
]


RECOMMENDATION_COLUMNS = [
    ("category", "category"),
    ("experiment", "experiment"),
    ("short_name", "short_name"),
    ("mean_f1", "mean_f1"),
    ("f1_gt_0_5", "f1_gt_0_5"),
    ("multimodal_t_mean_f1", "multimodal_t_mean_f1"),
    ("multimodal_f_mean_f1", "multimodal_f_mean_f1"),
    ("latency_mean", "latency_mean"),
    ("reason", "reason"),
]

SECTION_RETRIEVER_COLUMNS = [
    ("retriever", "retriever"),
    ("count_experiments", "count_experiments"),
    ("mean_mean_f1", "mean_mean_f1"),
    ("max_mean_f1", "max_mean_f1"),
    ("mean_f1_gt_0_5", "mean_f1_gt_0_5"),
    ("mean_latency", "mean_latency"),
    ("best_experiment", "best_experiment"),
]

SECTION_RERANKER_COLUMNS = [
    ("reranker", "reranker"),
    ("count_experiments", "count_experiments"),
    ("mean_mean_f1", "mean_mean_f1"),
    ("mean_latency", "mean_latency"),
    ("best_experiment", "best_experiment"),
]

SECTION_VLM_COLUMNS = [
    ("vlm", "vlm"),
    ("count_experiments", "count_experiments"),
    ("mean_mean_f1", "mean_mean_f1"),
    ("mean_latency", "mean_latency"),
    ("best_experiment", "best_experiment"),
]

SECTION_MODALITY_COLUMNS = [
    ("rank_by_mmf", "rank_by_mmf"),
    ("experiment", "experiment"),
    ("short_name", "short_name"),
    ("family", "family"),
    ("vlm", "vlm"),
    ("mean_f1", "mean_f1"),
    ("multimodal_t_mean_f1", "multimodal_t_mean_f1"),
    ("multimodal_f_mean_f1", "multimodal_f_mean_f1"),
    ("mm_gap", "mm_gap"),
    ("latency_mean", "latency_mean"),
]

SECTION_MODALITY_BY_VLM_COLUMNS = [
    ("vlm", "vlm"),
    ("count_experiments", "count_experiments"),
    ("mean_mm_t_f1", "mean_mm_t_f1"),
    ("mean_mm_f_f1", "mean_mm_f_f1"),
    ("mean_mm_gap", "mean_mm_gap"),
    ("best_mmf_experiment", "best_mmf_experiment"),
]

SECTION_ABLATION_SUMMARY_COLUMNS = [
    ("metric", "metric"),
    ("value", "value"),
    ("comparison", "comparison"),
]


def build_component_aggregation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    component_specs = [
        ("Family", "family"),
        ("Retriever", "retriever_short"),
        ("Reranker", "reranker_short"),
        ("VLM", "vlm_short"),
        ("Evidence", "evidence_short"),
    ]
    out: list[dict[str, object]] = []
    for component_name, key in component_specs:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(key, "unknown"))].append(row)
        for value, items in sorted(groups.items()):
            best = max(items, key=lambda r: metric(r, "mean_f1", -1.0))
            out.append(
                {
                    "component": component_name,
                    "value": value,
                    "count_experiments": len(items),
                    "mean_mean_f1": numeric_mean([item.get("mean_f1") for item in items]),
                    "max_mean_f1": metric(best, "mean_f1"),
                    "mean_f1_gt_0_5": numeric_mean([item.get("f1_gt_0_5") for item in items]),
                    "mean_mm_t_f1": numeric_mean(
                        [item.get("multimodal_t_mean_f1") for item in items]
                    ),
                    "mean_mm_f_f1": numeric_mean(
                        [item.get("multimodal_f_mean_f1") for item in items]
                    ),
                    "mean_latency": numeric_mean([item.get("latency_mean") for item in items]),
                    "best_experiment": best.get("experiment", ""),
                }
            )
    return out


def build_recommendations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def rec(category: str, row: dict[str, object], reason: str) -> dict[str, object]:
        return {
            "category": category,
            "experiment": row.get("experiment", ""),
            "short_name": row.get("short_name", ""),
            "mean_f1": row.get("mean_f1"),
            "f1_gt_0_5": row.get("f1_gt_0_5"),
            "multimodal_t_mean_f1": row.get("multimodal_t_mean_f1"),
            "multimodal_f_mean_f1": row.get("multimodal_f_mean_f1"),
            "latency_mean": row.get("latency_mean"),
            "reason": reason,
        }

    best_quality = max(rows, key=lambda r: metric(r, "mean_f1", -1.0))
    speed_candidates = [row for row in rows if metric(row, "mean_f1") >= 0.65]
    best_speed = min(speed_candidates or rows, key=lambda r: metric(r, "latency_mean", 999999.0))
    best_balanced = max(rows, key=lambda r: metric(r, "balanced_score", -1.0))
    text_candidates = [row for row in rows if str(row.get("family")) == "BM25 text reranking"]
    if not text_candidates:
        text_candidates = [row for row in rows if str(row.get("family")) == "Text evidence encoder"]
    best_text = max(text_candidates or rows, key=lambda r: metric(r, "mean_f1", -1.0))
    no_reranker_candidates = [row for row in rows if str(row.get("reranker_short")) == "None"]
    best_no_reranker = max(no_reranker_candidates or rows, key=lambda r: metric(r, "mean_f1", -1.0))

    return [
        rec("Best quality", best_quality, "Maximum Mean F1."),
        rec("Best speed", best_speed, "Lowest latency among experiments with Mean F1 >= 0.65."),
        rec("Best balanced", best_balanced, "Maximum Mean F1 / log(1 + latency)."),
        rec(
            "Best text-only baseline",
            best_text,
            "Best experiment in the BM25 text-reranking family.",
        ),
        rec(
            "Best no-reranker variant",
            best_no_reranker,
            "Best Mean F1 among rows without reranker.",
        ),
    ]


def group_rows(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return groups


def build_retriever_section(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for retriever, items in group_rows(rows, "retriever_short").items():
        best = max(items, key=lambda r: metric(r, "mean_f1", -1.0))
        out.append(
            {
                "retriever": retriever,
                "count_experiments": len(items),
                "mean_mean_f1": numeric_mean([item.get("mean_f1") for item in items]),
                "max_mean_f1": metric(best, "mean_f1"),
                "mean_f1_gt_0_5": numeric_mean([item.get("f1_gt_0_5") for item in items]),
                "mean_latency": numeric_mean([item.get("latency_mean") for item in items]),
                "best_experiment": best.get("experiment", ""),
            }
        )
    return sorted(out, key=lambda r: metric(r, "mean_mean_f1", -1.0), reverse=True)


def build_reranker_section(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for reranker, items in group_rows(rows, "reranker_short").items():
        best = max(items, key=lambda r: metric(r, "mean_f1", -1.0))
        out.append(
            {
                "reranker": reranker,
                "count_experiments": len(items),
                "mean_mean_f1": numeric_mean([item.get("mean_f1") for item in items]),
                "mean_latency": numeric_mean([item.get("latency_mean") for item in items]),
                "best_experiment": best.get("experiment", ""),
            }
        )
    return sorted(out, key=lambda r: metric(r, "mean_mean_f1", -1.0), reverse=True)


def build_vlm_section(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for vlm, items in group_rows(rows, "vlm_short").items():
        best = max(items, key=lambda r: metric(r, "mean_f1", -1.0))
        out.append(
            {
                "vlm": vlm,
                "count_experiments": len(items),
                "mean_mean_f1": numeric_mean([item.get("mean_f1") for item in items]),
                "mean_latency": numeric_mean([item.get("latency_mean") for item in items]),
                "best_experiment": best.get("experiment", ""),
            }
        )
    return sorted(out, key=lambda r: metric(r, "mean_mean_f1", -1.0), reverse=True)


def build_modality_section(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda r: metric(r, "multimodal_f_mean_f1", -1.0), reverse=True)
    out: list[dict[str, object]] = []
    for rank, row in enumerate(ranked, 1):
        out.append(
            {
                "rank_by_mmf": rank,
                "experiment": row.get("experiment", ""),
                "short_name": row.get("short_name", ""),
                "family": row.get("family", ""),
                "vlm": row.get("vlm_short", ""),
                "mean_f1": row.get("mean_f1"),
                "multimodal_t_mean_f1": row.get("multimodal_t_mean_f1"),
                "multimodal_f_mean_f1": row.get("multimodal_f_mean_f1"),
                "mm_gap": row.get("mm_gap"),
                "latency_mean": row.get("latency_mean"),
            }
        )
    return out


def build_modality_by_vlm(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for vlm, items in group_rows(rows, "vlm_short").items():
        best_mmf = max(items, key=lambda r: metric(r, "multimodal_f_mean_f1", -1.0))
        out.append(
            {
                "vlm": vlm,
                "count_experiments": len(items),
                "mean_mm_t_f1": numeric_mean([item.get("multimodal_t_mean_f1") for item in items]),
                "mean_mm_f_f1": numeric_mean([item.get("multimodal_f_mean_f1") for item in items]),
                "mean_mm_gap": numeric_mean([item.get("mm_gap") for item in items]),
                "best_mmf_experiment": best_mmf.get("experiment", ""),
            }
        )
    return sorted(out, key=lambda r: metric(r, "mean_mm_f_f1", -1.0), reverse=True)


def build_ablation_summary(ablation: list[dict[str, object]]) -> list[dict[str, object]]:
    if not ablation:
        return []
    avg_delta_f1 = numeric_mean([row.get("delta_mean_f1") for row in ablation]) or 0.0
    avg_delta_latency = numeric_mean([row.get("delta_latency") for row in ablation]) or 0.0
    max_delta_f1 = max(ablation, key=lambda r: metric(r, "delta_mean_f1", -1.0))
    max_delta_latency = max(ablation, key=lambda r: metric(r, "delta_latency", -1.0))
    return [
        {"metric": "mean_delta_f1", "value": avg_delta_f1, "comparison": "all reranker ablations"},
        {
            "metric": "mean_delta_latency",
            "value": avg_delta_latency,
            "comparison": "all reranker ablations",
        },
        {
            "metric": "max_delta_f1",
            "value": max_delta_f1.get("delta_mean_f1"),
            "comparison": max_delta_f1.get("comparison", ""),
        },
        {
            "metric": "max_delta_latency",
            "value": max_delta_latency.get("delta_latency"),
            "comparison": max_delta_latency.get("comparison", ""),
        },
    ]


def paper_setup_plotting() -> bool:
    if not setup_plotting():
        return False
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return True


def save_simple_barplot(
    rows: list[dict[str, object]],
    label_key: str,
    value_key: str,
    title: str,
    xlabel: str,
    output_name: str,
    color: str = "#4C78A8",
) -> None:
    if not paper_setup_plotting() or not rows:
        return
    labels = [str(row[label_key]) for row in rows]
    y = list(range(len(labels)))
    plt.figure(figsize=(12, max(4, 0.45 * len(labels))))
    plt.barh(y, [metric(row, value_key) for row in rows], color=color)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / output_name, dpi=180)
    plt.close()


def save_paper_quality_speed_plot(
    rows: list[dict[str, object]], recommendations: list[dict[str, object]]
) -> None:
    if not paper_setup_plotting():
        return
    colors = family_colors(rows)
    plt.figure(figsize=(12, 7))
    for family_name in sorted(colors):
        items = [row for row in rows if row.get("family") == family_name]
        plt.scatter(
            [metric(row, "latency_mean") for row in items],
            [metric(row, "mean_f1") for row in items],
            label=family_name,
            color=colors[family_name],
            s=70,
            alpha=0.85,
        )
    rec_by_category = {str(row["category"]): row for row in recommendations}
    markers = {
        "Best quality": ("*", 220, "#000000"),
        "Best speed": ("X", 130, "#D62728"),
        "Best balanced": ("P", 130, "#9467BD"),
    }
    by_exp = {str(row["experiment"]): row for row in rows}
    for category, (marker, size, color) in markers.items():
        rec = rec_by_category.get(category)
        if not rec:
            continue
        row = by_exp.get(str(rec.get("experiment")))
        if not row:
            continue
        plt.scatter(
            [metric(row, "latency_mean")],
            [metric(row, "mean_f1")],
            marker=marker,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=category,
            zorder=5,
        )
        plt.annotate(
            category,
            (metric(row, "latency_mean"), metric(row, "mean_f1")),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=8,
        )
    frontier = sorted(
        [row for row in rows if row.get("pareto_frontier")], key=lambda r: metric(r, "latency_mean")
    )
    if frontier:
        plt.plot(
            [metric(row, "latency_mean") for row in frontier],
            [metric(row, "mean_f1") for row in frontier],
            color="black",
            linewidth=2,
            linestyle="--",
            label="Pareto frontier",
        )
    plt.xlabel("Latency, seconds")
    plt.ylabel("Mean F1")
    plt.title("Section 5.7 Quality-Speed Trade-off")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / "5_7_quality_speed_tradeoff.png", dpi=180)
    plt.close()


def save_paper_modality_plot(modality_rows: list[dict[str, object]]) -> None:
    if not paper_setup_plotting():
        return
    top10 = modality_rows[:10]
    labels = [str(row.get("short_name", ""))[:32] for row in top10]
    x = list(range(len(top10)))
    width = 0.38
    plt.figure(figsize=(14, 7))
    plt.bar(
        [i - width / 2 for i in x],
        [metric(row, "multimodal_t_mean_f1") for row in top10],
        width=width,
        label="MM-T F1",
        color="#54A24B",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [metric(row, "multimodal_f_mean_f1") for row in top10],
        width=width,
        label="MM-F F1",
        color="#E45756",
    )
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("F1")
    plt.title("Section 5.6 Modality Analysis: Top-10 by MM-F F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_FIGURES_DIR / "5_6_modality_top10_mmf.png", dpi=180)
    plt.close()


def write_paper_section_findings(
    path: Path,
    retriever_rows: list[dict[str, object]],
    reranker_rows: list[dict[str, object]],
    ablation: list[dict[str, object]],
    ablation_summary: list[dict[str, object]],
    vlm_rows: list[dict[str, object]],
    modality_rows: list[dict[str, object]],
    modality_by_vlm: list[dict[str, object]],
    recommendations: list[dict[str, object]],
) -> None:
    best_retriever = retriever_rows[0]
    best_reranker = reranker_rows[0]
    best_vlm = vlm_rows[0]
    best_mmf = modality_rows[0]
    max_gap = max(modality_rows, key=lambda r: abs(metric(r, "mm_gap", 0.0)))
    min_gap = min(modality_rows, key=lambda r: abs(metric(r, "mm_gap", 0.0)))
    recs = {str(row["category"]): row for row in recommendations}
    best_quality = recs["Best quality"]
    best_speed = recs["Best speed"]
    best_balanced = recs["Best balanced"]
    summary = {str(row["metric"]): row for row in ablation_summary}
    max_gain = summary.get("max_delta_f1", {})
    max_latency = summary.get("max_delta_latency", {})
    avg_delta_f1 = summary.get("mean_delta_f1", {}).get("value", 0.0)
    avg_delta_latency = summary.get("mean_delta_latency", {}).get("value", 0.0)

    lines = [
        "# Section 5 Experiments: Paper-Facing Analysis",
        "",
        "This file is generated from `reports/tables/paper_multimodal_308.csv` and is intended as direct material for the Experiments section.",
        "",
        "## 5.3 Retriever Analysis",
        "",
        (
            f"The strongest retriever group is **{best_retriever['retriever']}**, "
            f"with average Mean F1={format_value(best_retriever['mean_mean_f1'])} "
            f"and best single-run Mean F1={format_value(best_retriever['max_mean_f1'])} "
            f"from `{best_retriever['best_experiment']}`. "
            "This supports using visually grounded retrieval as the first stage for multimodal DocBench questions."
        ),
        "",
        "## 5.4 Reranker Analysis",
        "",
        (
            f"Aggregating by reranker, **{best_reranker['reranker']}** has the highest average Mean F1 "
            f"({format_value(best_reranker['mean_mean_f1'])}) among reranker groups. "
            f"Across paired ablations, reranking improves Mean F1 by {format_value(avg_delta_f1)} on average, "
            f"while adding {format_value(avg_delta_latency)} seconds of latency on average."
        ),
        (
            f"The largest quality gain is observed for **{max_gain.get('comparison', '')}** "
            f"(Delta F1={format_value(max_gain.get('value', 0.0))}), whereas the largest latency cost is "
            f"**{max_latency.get('comparison', '')}** (Delta latency={format_value(max_latency.get('value', 0.0))}s)."
        ),
        "",
        "## 5.5 VLM Analysis",
        "",
        (
            f"The best VLM group is **{best_vlm['vlm']}**, with average Mean F1="
            f"{format_value(best_vlm['mean_mean_f1'])} and average latency="
            f"{format_value(best_vlm['mean_latency'])}s. "
            f"Its best experiment is `{best_vlm['best_experiment']}`."
        ),
        "",
        "## 5.6 Modality Analysis",
        "",
        (
            f"The best MM-F result is obtained by `{best_mmf['experiment']}` "
            f"with MM-F F1={format_value(best_mmf['multimodal_f_mean_f1'])}. "
            f"The smallest MM-T/MM-F gap is `{min_gap['experiment']}` "
            f"(gap={format_value(min_gap['mm_gap'])}), while the largest gap is "
            f"`{max_gap['experiment']}` (gap={format_value(max_gap['mm_gap'])})."
        ),
        (
            "Overall, MM-T scores are generally higher than MM-F scores, indicating that table/text-heavy "
            "questions benefit more from text evidence and layout-aware crops than figure/visual-heavy questions."
        ),
        "",
        "## 5.7 Quality-Speed Trade-off",
        "",
        (
            f"The best-quality configuration is `{best_quality['experiment']}` "
            f"(Mean F1={format_value(best_quality['mean_f1'])}, latency={format_value(best_quality['latency_mean'])}s). "
            f"The fastest configuration above Mean F1 >= 0.65 is `{best_speed['experiment']}` "
            f"(Mean F1={format_value(best_speed['mean_f1'])}, latency={format_value(best_speed['latency_mean'])}s). "
            f"The balanced score criterion selects `{best_balanced['experiment']}`."
        ),
        "",
        "## Generated Tables",
        "",
        "- `5_3_retriever_analysis.csv`",
        "- `5_4_reranker_aggregation.csv`",
        "- `5_4_reranker_ablation.csv`",
        "- `5_4_reranker_ablation_summary.csv`",
        "- `5_5_vlm_analysis.csv`",
        "- `5_6_modality_analysis.csv`",
        "- `5_6_modality_by_vlm.csv`",
        "- `5_7_quality_speed_recommendations.csv`",
        "",
        "## Generated Figures",
        "",
        "- `figures/5_3_retriever_mean_f1.png`",
        "- `figures/5_4_reranker_delta_f1.png`",
        "- `figures/5_4_reranker_delta_latency.png`",
        "- `figures/5_5_vlm_mean_f1.png`",
        "- `figures/5_6_modality_top10_mmf.png`",
        "- `figures/5_7_quality_speed_tradeoff.png`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_section_outputs(
    rows: list[dict[str, object]],
    ablation: list[dict[str, object]],
    recommendations: list[dict[str, object]],
) -> None:
    PAPER_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    retriever_rows = build_retriever_section(rows)
    reranker_rows = build_reranker_section(rows)
    vlm_rows = build_vlm_section(rows)
    modality_rows = build_modality_section(rows)
    modality_by_vlm = build_modality_by_vlm(rows)
    ablation_summary = build_ablation_summary(ablation)

    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_3_retriever_analysis.csv", retriever_rows, SECTION_RETRIEVER_COLUMNS
    )
    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_4_reranker_aggregation.csv", reranker_rows, SECTION_RERANKER_COLUMNS
    )
    write_labeled_csv(PAPER_SECTIONS_DIR / "5_4_reranker_ablation.csv", ablation, ABLATION_COLUMNS)
    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_4_reranker_ablation_summary.csv",
        ablation_summary,
        SECTION_ABLATION_SUMMARY_COLUMNS,
    )
    write_labeled_csv(PAPER_SECTIONS_DIR / "5_5_vlm_analysis.csv", vlm_rows, SECTION_VLM_COLUMNS)
    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_6_modality_analysis.csv", modality_rows, SECTION_MODALITY_COLUMNS
    )
    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_6_modality_by_vlm.csv",
        modality_by_vlm,
        SECTION_MODALITY_BY_VLM_COLUMNS,
    )
    write_labeled_csv(
        PAPER_SECTIONS_DIR / "5_7_quality_speed_recommendations.csv",
        recommendations,
        RECOMMENDATION_COLUMNS,
    )

    save_simple_barplot(
        retriever_rows,
        "retriever",
        "mean_mean_f1",
        "Section 5.3 Retriever Analysis",
        "Mean of Mean F1",
        "5_3_retriever_mean_f1.png",
        color="#4C78A8",
    )
    save_simple_barplot(
        ablation,
        "comparison",
        "delta_mean_f1",
        "Section 5.4 Reranker Delta F1",
        "Delta Mean F1",
        "5_4_reranker_delta_f1.png",
        color="#4C78A8",
    )
    save_simple_barplot(
        ablation,
        "comparison",
        "delta_latency",
        "Section 5.4 Reranker Delta Latency",
        "Delta latency, seconds",
        "5_4_reranker_delta_latency.png",
        color="#F58518",
    )
    save_simple_barplot(
        vlm_rows,
        "vlm",
        "mean_mean_f1",
        "Section 5.5 VLM Analysis",
        "Mean of Mean F1",
        "5_5_vlm_mean_f1.png",
        color="#72B7B2",
    )
    save_paper_modality_plot(modality_rows)
    save_paper_quality_speed_plot(rows, recommendations)

    write_paper_section_findings(
        PAPER_SECTIONS_DIR / "paper_section_findings.md",
        retriever_rows,
        reranker_rows,
        ablation,
        ablation_summary,
        vlm_rows,
        modality_rows,
        modality_by_vlm,
        recommendations,
    )


def write_labeled_csv(
    path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in columns])
        for row in rows:
            writer.writerow([format_value(row.get(key, "")) for key, _ in columns])


def setup_plotting() -> bool:
    global PLOT_WARNING_EMITTED
    if plt is None:
        if not PLOT_WARNING_EMITTED:
            print("Plotting skipped: matplotlib/seaborn is not installed.")
            PLOT_WARNING_EMITTED = True
        return False
    if sns is not None:
        sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return True


def family_colors(rows: list[dict[str, object]]) -> dict[str, object]:
    families = sorted({str(row.get("family", "unknown")) for row in rows})
    if sns is not None:
        palette = sns.color_palette("tab10", n_colors=max(len(families), 1))
    else:
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i % 10) for i in range(len(families))]
    return {family_name: palette[i] for i, family_name in enumerate(families)}


def save_quality_speed_plot(rows: list[dict[str, object]]) -> None:
    if not setup_plotting():
        return
    colors = family_colors(rows)
    plt.figure(figsize=(12, 7))
    for family_name in sorted(colors):
        items = [row for row in rows if row.get("family") == family_name]
        plt.scatter(
            [metric(row, "latency_mean") for row in items],
            [metric(row, "mean_f1") for row in items],
            label=family_name,
            color=colors[family_name],
            s=70,
            alpha=0.85,
        )

    top5 = sorted(rows, key=lambda r: metric(r, "mean_f1"), reverse=True)[:5]
    for row in top5:
        plt.annotate(
            str(row.get("short_name", ""))[:28],
            (metric(row, "latency_mean"), metric(row, "mean_f1")),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    frontier = [row for row in rows if row.get("pareto_frontier")]
    frontier = sorted(frontier, key=lambda r: metric(r, "latency_mean"))
    if frontier:
        plt.plot(
            [metric(row, "latency_mean") for row in frontier],
            [metric(row, "mean_f1") for row in frontier],
            color="black",
            linewidth=2,
            linestyle="--",
            label="Pareto frontier",
        )

    plt.xlabel("Latency, seconds")
    plt.ylabel("Mean F1")
    plt.title("Quality / Speed Trade-off")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quality_speed_tradeoff.png", dpi=180)
    plt.close()


def save_ablation_plots(ablation: list[dict[str, object]]) -> None:
    if not setup_plotting() or not ablation:
        return
    labels = [str(row["comparison"]) for row in ablation]
    y = list(range(len(labels)))

    plt.figure(figsize=(12, 6))
    plt.barh(y, [metric(row, "delta_mean_f1") for row in ablation], color="#4C78A8")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Delta Mean F1")
    plt.title("Reranker Quality Gain")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reranker_delta_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.barh(y, [metric(row, "delta_latency") for row in ablation], color="#F58518")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Delta latency, seconds")
    plt.title("Reranker Latency Cost")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reranker_delta_latency.png", dpi=180)
    plt.close()


def save_modality_plot(rows: list[dict[str, object]]) -> None:
    if not setup_plotting():
        return
    top10 = sorted(rows, key=lambda r: metric(r, "mean_f1"), reverse=True)[:10]
    labels = [str(row.get("short_name", ""))[:32] for row in top10]
    x = list(range(len(top10)))
    width = 0.38

    plt.figure(figsize=(14, 7))
    plt.bar(
        [i - width / 2 for i in x],
        [metric(row, "multimodal_t_mean_f1") for row in top10],
        width=width,
        label="MM-T F1",
        color="#54A24B",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [metric(row, "multimodal_f_mean_f1") for row in top10],
        width=width,
        label="MM-F F1",
        color="#E45756",
    )
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("F1")
    plt.title("MM-T vs MM-F for Top-10 Experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modality_top10_grouped.png", dpi=180)
    plt.close()


def print_extended_analysis(
    rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    ablation: list[dict[str, object]],
    recommendations: list[dict[str, object]],
) -> None:
    best_quality = recommendations[0]
    best_speed = recommendations[1]
    avg_delta_f1 = numeric_mean([row.get("delta_mean_f1") for row in ablation]) or 0.0
    avg_delta_latency = numeric_mean([row.get("delta_latency") for row in ablation]) or 0.0
    best_retriever = max(
        [row for row in component_rows if row["component"] == "Retriever"],
        key=lambda r: metric(r, "mean_mean_f1", -1.0),
    )
    best_vlm = max(
        [row for row in component_rows if row["component"] == "VLM"],
        key=lambda r: metric(r, "mean_mean_f1", -1.0),
    )
    best_rerank_gain = (
        max(ablation, key=lambda r: metric(r, "delta_mean_f1", -1.0)) if ablation else None
    )
    most_expensive_rerank = (
        max(ablation, key=lambda r: metric(r, "delta_latency", -1.0)) if ablation else None
    )
    min_gap = min(rows, key=lambda r: abs(metric(r, "mm_gap", 0.0)))
    max_gap = max(rows, key=lambda r: abs(metric(r, "mm_gap", 0.0)))

    print()
    print("=" * 100)
    print("EXTENDED ANALYSIS")
    print("=" * 100)
    print(
        f"Best quality: {best_quality['short_name']} "
        f"(Mean F1={format_value(best_quality['mean_f1'])}, latency={format_value(best_quality['latency_mean'])}s)."
    )
    print(
        f"Best fast option with Mean F1 >= 0.65: {best_speed['short_name']} "
        f"(Mean F1={format_value(best_speed['mean_f1'])}, latency={format_value(best_speed['latency_mean'])}s)."
    )
    print(f"Average reranker gain: +{format_value(avg_delta_f1)} Mean F1.")
    print(f"Average reranker latency cost: +{format_value(avg_delta_latency)}s.")
    if best_rerank_gain:
        print(
            f"Best reranker quality gain: {best_rerank_gain['comparison']} "
            f"(+{format_value(best_rerank_gain['delta_mean_f1'])} Mean F1)."
        )
    if most_expensive_rerank:
        print(
            f"Most expensive reranker by latency: {most_expensive_rerank['comparison']} "
            f"(+{format_value(most_expensive_rerank['delta_latency'])}s)."
        )
    print(
        f"Best average retriever group: {best_retriever['value']} "
        f"(mean Mean F1={format_value(best_retriever['mean_mean_f1'])})."
    )
    print(
        f"Best average VLM group: {best_vlm['value']} "
        f"(mean Mean F1={format_value(best_vlm['mean_mean_f1'])})."
    )
    print(
        f"Smallest MM-T/MM-F gap: {min_gap['short_name']} "
        f"(gap={format_value(min_gap['mm_gap'])})."
    )
    print(
        f"Largest MM-T/MM-F gap: {max_gap['short_name']} "
        f"(gap={format_value(max_gap['mm_gap'])})."
    )
    print(
        "Paper/report conclusion: the best-quality method is full image+text Nemotron retrieval/reranking "
        "with Qwen3-VL-30B, while no-reranker Nemotron image+text is the practical speed/quality trade-off."
    )


def write_csv(
    path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]], delimiter: str = ","
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow([label for _, label in columns])
        for row in rows:
            writer.writerow([format_value(row.get(key, "")) for key, _ in columns])


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    labels = [label for _, label in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in rows:
        values = [format_value(row.get(key, "")).replace("|", "/") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def print_table(title: str, rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    headers = [label for _, label in columns]
    data = [[format_value(row.get(key, "")) for key, _ in columns] for row in rows]
    widths = [
        min(max(len(header), *(len(row[i]) for row in data)), 42)
        for i, header in enumerate(headers)
    ]
    print(" | ".join(header[: widths[i]].ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in data:
        print(" | ".join(row[i][: widths[i]].ljust(widths[i]) for i in range(len(widths))))


def build_ablation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_exp = {str(row["experiment"]): row for row in rows}

    def make(comparison: str, no_exp: str, with_exp: str) -> dict[str, object] | None:
        no = by_exp.get(no_exp)
        yes = by_exp.get(with_exp)
        if not no or not yes:
            return None
        delta_f1 = (yes["mean_f1"] or 0.0) - (no["mean_f1"] or 0.0)
        delta_latency = (yes["latency_mean"] or 0.0) - (no["latency_mean"] or 0.0)
        verdict = (
            "Quality improves, latency increases" if delta_f1 > 0 and delta_latency > 0 else "Mixed"
        )
        return {
            "comparison": comparison,
            "no_reranker": clean_name(no_exp),
            "with_reranker": clean_name(with_exp),
            "mean_f1_no": no["mean_f1"],
            "mean_f1_with": yes["mean_f1"],
            "delta_mean_f1": delta_f1,
            "latency_no": no["latency_mean"],
            "latency_with": yes["latency_mean"],
            "delta_latency": delta_latency,
            "verdict": verdict,
        }

    pairs = [
        make(
            "ColPali Qwen8B VL reranker",
            "multimodal_308_no_reranker",
            "multimodal_308_with_reranker",
        ),
        make(
            "ColPali Qwen30B VL reranker",
            "docbench_qwen3vl30b_no_reranker",
            "docbench_hybrid_bm25_tools_qwen3vl30b",
        ),
        make(
            "Nemotron image Qwen30B VL reranker",
            "multimodal_308_nemotron_image_retriever_qwen3vl30b",
            "multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b",
        ),
        make(
            "Nemotron fusion Qwen30B image reranker",
            "image_text_fusion_308_nemotron_no_reranker_qwen3vl30b",
            "image_text_fusion_308_nemotron_qwen3vl30b",
        ),
        make(
            "Nemotron full image+text Qwen30B text-image reranker",
            "image_text_full_308_nemotron_no_reranker_qwen3vl30b",
            "image_text_full_308_nemotron_qwen3vl30b",
        ),
        make(
            "BM25 text BGE-large reranker",
            "text_reranker_308_no_reranker_qwen3vl30b",
            "text_reranker_308_bge_large_qwen3vl30b",
        ),
        make(
            "Text evidence BGE-large reranker",
            "text_evidence_encoder_308_bge_large_en_v1_5_no_reranker_qwen3vl30b",
            "text_evidence_encoder_308_bge_large_en_v1_5_bge_reranker_large_qwen3vl30b",
        ),
    ]
    return [pair for pair in pairs if pair is not None]


def write_markdown(
    path: Path, rows: list[dict[str, object]], ablation: list[dict[str, object]]
) -> None:
    best = rows[0]
    fastest_visual = min(
        [row for row in rows if "text" not in str(row["family"]).lower()],
        key=lambda row: row["latency_mean"] or 999.0,
    )
    best_text = max(
        [row for row in rows if "text" in str(row["family"]).lower()],
        key=lambda row: row["mean_f1"] or -1.0,
    )
    lines = [
        "# Human Experiment Summary",
        "",
        "## Short Conclusions",
        "",
        f"- Best overall: **{best['short_name']}**, mean F1 = **{format_value(best['mean_f1'])}**, latency = **{format_value(best['latency_mean'])}s**.",
        f"- Fastest visual branch: **{fastest_visual['short_name']}**, mean F1 = **{format_value(fastest_visual['mean_f1'])}**, latency = **{format_value(fastest_visual['latency_mean'])}s**.",
        f"- Best text branch: **{best_text['short_name']}**, mean F1 = **{format_value(best_text['mean_f1'])}**.",
        "- Visual reranking remains important: removing it is much faster, but quality drops.",
        "- Text-only and text-evidence branches are useful advisor baselines, not the best final method.",
        "",
        "## Main Table",
        "",
        markdown_table(rows, DISPLAY_COLUMNS),
        "",
        "## Reranker Ablation",
        "",
        markdown_table(ablation, ABLATION_COLUMNS),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SOURCE_CSV}")

    rows = enrich(read_rows(SOURCE_CSV))
    enrich_analysis_fields(rows)
    ablation = build_ablation(rows)
    component_rows = build_component_aggregation(rows)
    recommendations = build_recommendations(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    main_csv = OUT_DIR / "human_experiment_summary.csv"
    main_tsv = OUT_DIR / "human_experiment_summary_for_excel.tsv"
    ablation_csv = OUT_DIR / "human_reranker_ablation.csv"
    ablation_tsv = OUT_DIR / "human_reranker_ablation_for_excel.tsv"
    markdown = OUT_DIR / "human_experiment_summary.md"

    write_csv(main_csv, rows, DISPLAY_COLUMNS)
    write_csv(main_tsv, rows, DISPLAY_COLUMNS, delimiter="\t")
    write_csv(ablation_csv, ablation, ABLATION_COLUMNS)
    write_csv(ablation_tsv, ablation, ABLATION_COLUMNS, delimiter="\t")
    write_markdown(markdown, rows, ablation)

    summary_main_csv = SUMMARY_DIR / "main_table.csv"
    summary_component_csv = SUMMARY_DIR / "component_aggregation.csv"
    summary_ablation_csv = SUMMARY_DIR / "reranker_ablation.csv"
    summary_recommendations_csv = SUMMARY_DIR / "recommendations.csv"

    write_labeled_csv(summary_main_csv, rows, MAIN_ANALYSIS_COLUMNS)
    write_labeled_csv(summary_component_csv, component_rows, COMPONENT_COLUMNS)
    write_labeled_csv(summary_ablation_csv, ablation, ABLATION_COLUMNS)
    write_labeled_csv(summary_recommendations_csv, recommendations, RECOMMENDATION_COLUMNS)

    save_quality_speed_plot(rows)
    save_ablation_plots(ablation)
    save_modality_plot(rows)
    write_paper_section_outputs(rows, ablation, recommendations)

    print_table("MAIN EXPERIMENT TABLE", rows, DISPLAY_COLUMNS)
    print_table("RERANKER ABLATION", ablation, ABLATION_COLUMNS)
    print_extended_analysis(rows, component_rows, ablation, recommendations)
    print()
    print("Saved files:")
    print(f"- {main_csv}")
    print(f"- {main_tsv}")
    print(f"- {ablation_csv}")
    print(f"- {ablation_tsv}")
    print(f"- {markdown}")
    print(f"- {summary_main_csv}")
    print(f"- {summary_component_csv}")
    print(f"- {summary_ablation_csv}")
    print(f"- {summary_recommendations_csv}")
    for figure_name in [
        "quality_speed_tradeoff.png",
        "reranker_delta_f1.png",
        "reranker_delta_latency.png",
        "modality_top10_grouped.png",
    ]:
        figure_path = FIGURES_DIR / figure_name
        status = (
            "generated" if figure_path.exists() else "not generated; install matplotlib/seaborn"
        )
        print(f"- {figure_path} ({status})")
    print(f"- {PAPER_SECTIONS_DIR / 'paper_section_findings.md'}")
    for csv_name in [
        "5_3_retriever_analysis.csv",
        "5_4_reranker_aggregation.csv",
        "5_4_reranker_ablation.csv",
        "5_4_reranker_ablation_summary.csv",
        "5_5_vlm_analysis.csv",
        "5_6_modality_analysis.csv",
        "5_6_modality_by_vlm.csv",
        "5_7_quality_speed_recommendations.csv",
    ]:
        print(f"- {PAPER_SECTIONS_DIR / csv_name}")
    for figure_name in [
        "5_3_retriever_mean_f1.png",
        "5_4_reranker_delta_f1.png",
        "5_4_reranker_delta_latency.png",
        "5_5_vlm_mean_f1.png",
        "5_6_modality_top10_mmf.png",
        "5_7_quality_speed_tradeoff.png",
    ]:
        figure_path = PAPER_FIGURES_DIR / figure_name
        status = (
            "generated" if figure_path.exists() else "not generated; install matplotlib/seaborn"
        )
        print(f"- {figure_path} ({status})")


if __name__ == "__main__":
    main()
