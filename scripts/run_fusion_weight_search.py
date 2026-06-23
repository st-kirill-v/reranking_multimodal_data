from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import dump_config, load_config


GRID: list[tuple[float, float, float]] = [
    (0.2, 0.2, 0.6),
    (0.2, 0.3, 0.5),
    (0.2, 0.4, 0.4),
    (0.3, 0.2, 0.5),
    (0.3, 0.3, 0.4),
    (0.3, 0.4, 0.3),
    (0.4, 0.2, 0.4),
    (0.4, 0.3, 0.3),
    (0.5, 0.2, 0.3),
]

BASELINES = [
    ("Best quality baseline", 0.7023, 13.6441),
    ("Fast fusion baseline", 0.6575, 2.5080),
    ("No-reranker full image+text baseline", 0.6784, 3.4263),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory Score Fusion weight search without touching old results."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/experiments/image_text_fusion_308_nemotron_qwen3vl30b.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/fusion_weight_search_308_qwen3vl30b"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/fusion_weight_search"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def experiment_suffix(alpha: float, beta: float, gamma: float) -> str:
    return f"a{int(alpha * 100):03d}_b{int(beta * 100):03d}_g{int(gamma * 100):03d}"


def make_config(
    base_config: dict[str, Any], alpha: float, beta: float, gamma: float, root: Path
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    suffix = experiment_suffix(alpha, beta, gamma)
    experiment_name = f"fusion_weight_search_308_qwen3vl30b_{suffix}"
    config["experiment_name"] = experiment_name
    config.setdefault("fusion", {})["mode"] = "score_fusion"
    config["fusion"]["alpha"] = alpha
    config["fusion"]["beta"] = beta
    config["fusion"]["gamma"] = gamma
    config.setdefault("output", {})["directory"] = str(root / suffix)
    config.setdefault("cropping", {})["debug_crop_dir"] = str(
        Path("data/debug_crops/fusion_weight_search_308_qwen3vl30b") / suffix
    )
    config.setdefault("metadata", {})["experiment_family"] = "fusion_weight_search_308_qwen3vl30b"
    config["metadata"]["methodological_note"] = (
        "Additional exploratory fusion experiment; weights are tuned on a small 308-question subset, "
        "so results must not replace the main paper result without a separate validation protocol."
    )
    return config


def run_config(config_path: Path, *, dry_run: bool) -> None:
    command = [sys.executable, "scripts/run_experiment.py", "--config", str(config_path)]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


def load_metrics(result_dir: Path) -> dict[str, Any] | None:
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def metric_row(alpha: float, beta: float, gamma: float, result_dir: Path) -> dict[str, Any]:
    metrics = load_metrics(result_dir)
    if metrics is None:
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "mean_f1": "",
            "exact_match": "",
            "f1_gt_0_5": "",
            "mm_t_f1": "",
            "mm_f_f1": "",
            "latency": "",
            "number_of_candidates": 10,
            "notes": "not_run_or_metrics_missing",
        }
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "mean_f1": metrics.get("mean_f1"),
        "exact_match": metrics.get("exact_match"),
        "f1_gt_0_5": metrics.get("accuracy_f1_gt_0_5"),
        "mm_t_f1": metrics.get("by_modality", {}).get("multimodal-t", {}).get("mean_f1"),
        "mm_f_f1": metrics.get("by_modality", {}).get("multimodal-f", {}).get("mean_f1"),
        "latency": metrics.get("latency_seconds", {}).get("mean"),
        "number_of_candidates": 10,
        "notes": "all_scores_available; exploratory_grid; overfitting_risk_308_questions",
    }


def _float_or_none(value: Any) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def write_reports(rows: list[dict[str, Any]], report_dir: Path, output_root: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "fusion_weight_search_results.csv"
    fieldnames = [
        "alpha",
        "beta",
        "gamma",
        "mean_f1",
        "exact_match",
        "f1_gt_0_5",
        "mm_t_f1",
        "mm_f_f1",
        "latency",
        "number_of_candidates",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if _float_or_none(row.get("mean_f1")) is not None]
    best_quality = max(completed, key=lambda row: float(row["mean_f1"])) if completed else None
    best_tradeoff = (
        max(completed, key=lambda row: float(row["mean_f1"]) / max(float(row["latency"]), 1e-9))
        if completed
        else None
    )
    comparison_lines: list[str] = []
    if best_quality:
        best_f1 = float(best_quality["mean_f1"])
        best_latency = float(best_quality["latency"])
        for name, baseline_f1, baseline_latency in BASELINES:
            comparison_lines.append(
                f"| {name} | {baseline_f1:.4f} | {baseline_latency:.4f} | "
                f"{best_f1 - baseline_f1:.4f} | {best_latency - baseline_latency:.4f} |"
            )
    else:
        comparison_lines.append("| not available |  |  |  |  |")

    table_lines = [
        "| alpha | beta | gamma | Mean F1 | EM | F1 > 0.5 | MM-T F1 | MM-F F1 | latency | notes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['alpha']} | {row['beta']} | {row['gamma']} | "
            f"{row['mean_f1']} | {row['exact_match']} | {row['f1_gt_0_5']} | "
            f"{row['mm_t_f1']} | {row['mm_f_f1']} | {row['latency']} | {row['notes']} |"
        )

    summary = [
        "# Fusion Weight Search Summary",
        "",
        "Статус: additional exploratory fusion experiment.",
        "",
        "## Что проверяется",
        "",
        "Подбор весов `alpha`, `beta`, `gamma` для Score Fusion:",
        "",
        "```text",
        "final_score = alpha * normalized_retrieval_score",
        "            + beta  * normalized_text_rerank_score",
        "            + gamma * normalized_image_rerank_score",
        "            + heuristic_bonuses",
        "```",
        "",
        "В этой сетке используется Nemotron fusion режим, где доступны все три score. Для режимов без image reranker вес `gamma` должен быть перенормирован на доступные score; такие результаты в данной сетке не смешиваются с основной таблицей.",
        "",
        "## Methodological note",
        "",
        "DocBench subset содержит 308 вопросов, поэтому подбор весов несет риск overfitting. Если найденный результат окажется лучше baseline, он не должен автоматически заменять основной результат статьи без отдельной валидации.",
        "",
        "## Results",
        "",
        *table_lines,
        "",
        "## Best candidates",
        "",
        f"- Best quality: {best_quality if best_quality else 'not available'}",
        f"- Best F1/latency trade-off: {best_tradeoff if best_tradeoff else 'not available'}",
        "",
        "## Baseline comparison",
        "",
        "| Baseline | Baseline Mean F1 | Baseline latency | Best fusion delta F1 | Best fusion delta latency |",
        "|---|---:|---:|---:|---:|",
        *comparison_lines,
        "",
        "## Output",
        "",
        f"- results root: `{output_root}`",
        f"- CSV: `{csv_path}`",
        "",
    ]
    (report_dir / "fusion_weight_search_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    base_config = load_config(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir = args.output_root / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha, beta, gamma in GRID:
        config = make_config(base_config, alpha, beta, gamma, args.output_root)
        if args.limit is not None:
            config.setdefault("evaluation", {})["limit"] = args.limit
        suffix = experiment_suffix(alpha, beta, gamma)
        result_dir = args.output_root / suffix
        config_path = generated_config_dir / f"{suffix}.json"
        dump_config(config, config_path)
        if not (args.skip_existing and (result_dir / "metrics.json").exists()):
            run_config(config_path, dry_run=args.dry_run)
        rows.append(metric_row(alpha, beta, gamma, result_dir))

    write_reports(rows, args.report_dir, args.output_root)
    print(f"Saved: {args.report_dir / 'fusion_weight_search_results.csv'}")
    print(f"Saved: {args.report_dir / 'fusion_weight_search_summary.md'}")


if __name__ == "__main__":
    main()
