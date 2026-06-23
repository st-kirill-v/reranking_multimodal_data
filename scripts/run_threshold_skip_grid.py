from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import dump_config, load_config


THRESHOLD_GRID: list[tuple[float, float]] = [
    (0.70, 0.05),
    (0.70, 0.10),
    (0.75, 0.10),
    (0.75, 0.15),
    (0.80, 0.10),
    (0.80, 0.15),
    (0.85, 0.10),
    (0.85, 0.20),
    (0.90, 0.25),
]

BEST_QUALITY_BASELINE = {"mean_f1": 0.7023, "latency": 13.6441}
NO_RERANKER_BASELINE = {"mean_f1": 0.6784, "latency": 3.4263}
FAST_FUSION_BASELINE = {"mean_f1": 0.6575, "latency": 2.5080}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Threshold Skip Reranking grid as an exploratory experiment."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/experiments/threshold_skip_reranking_308_qwen3vl30b.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/threshold_skip_reranking_308_qwen3vl30b/grid_limit30"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/threshold_skip"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of questions per threshold for preliminary search. Use --full to run all 308.",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run the grid on all selected questions."
    )
    return parser.parse_args()


def threshold_suffix(threshold_top1: float, threshold_gap: float) -> str:
    return f"top1_{int(threshold_top1 * 100):03d}_gap_{int(threshold_gap * 100):03d}"


def make_config(
    base_config: dict[str, Any],
    threshold_top1: float,
    threshold_gap: float,
    output_root: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    suffix = threshold_suffix(threshold_top1, threshold_gap)
    config["experiment_name"] = f"threshold_skip_reranking_308_qwen3vl30b_{suffix}"
    config.setdefault("reranking", {})["mode"] = "threshold_skip"
    config.setdefault("threshold_skip", {})["threshold_top1"] = threshold_top1
    config["threshold_skip"]["threshold_gap"] = threshold_gap
    config.setdefault("output", {})["directory"] = str(output_root / suffix)
    config.setdefault("cropping", {})["debug_crop_dir"] = str(
        Path("data/debug_crops/threshold_skip_reranking_308_qwen3vl30b") / suffix
    )
    config.setdefault("metadata", {})["experiment_family"] = "threshold_skip_grid"
    config["metadata"]["methodological_note"] = (
        "Exploratory threshold search on 308 DocBench questions; do not replace the main "
        "paper baseline without validation on another split."
    )
    return config


def run_config(config_path: Path, *, dry_run: bool) -> None:
    command = [sys.executable, "scripts/run_experiment.py", "--config", str(config_path)]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1 - frac) + values[high] * frac


def score_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def metrics_row(
    threshold_top1: float,
    threshold_gap: float,
    result_dir: Path,
) -> dict[str, Any]:
    metrics = load_json(result_dir / "metrics.json")
    predictions = load_predictions(result_dir / "predictions.jsonl")
    skipped = [row for row in predictions if row.get("threshold_skip_reranker") is True]
    reranked = [row for row in predictions if row.get("threshold_skip_reranker") is False]
    if metrics is None:
        return {
            "threshold_top1": threshold_top1,
            "threshold_gap": threshold_gap,
            "mean_f1": "",
            "exact_match": "",
            "f1_gt_0_5": "",
            "mm_t_f1": "",
            "mm_f_f1": "",
            "latency_mean": "",
            "latency_p50": "",
            "latency_p95": "",
            "skip_rate": "",
            "number_skipped": "",
            "number_reranked": "",
            "quality_delta_vs_best": "",
            "latency_delta_vs_best": "",
            "quality_delta_vs_no_reranker": "",
            "latency_delta_vs_no_reranker": "",
            "quality_delta_vs_fast_fusion": "",
            "latency_delta_vs_fast_fusion": "",
            "notes": "not_run_or_metrics_missing",
        }
    latency = metrics.get("latency_seconds", {})
    mean_f1 = float(metrics.get("mean_f1", 0.0))
    latency_mean = float(latency.get("mean", 0.0))
    total = max(len(predictions), 1)
    return {
        "threshold_top1": threshold_top1,
        "threshold_gap": threshold_gap,
        "mean_f1": mean_f1,
        "exact_match": metrics.get("exact_match"),
        "f1_gt_0_5": metrics.get("accuracy_f1_gt_0_5"),
        "mm_t_f1": metrics.get("by_modality", {}).get("multimodal-t", {}).get("mean_f1"),
        "mm_f_f1": metrics.get("by_modality", {}).get("multimodal-f", {}).get("mean_f1"),
        "latency_mean": latency_mean,
        "latency_p50": latency.get("p50"),
        "latency_p95": latency.get("p95"),
        "skip_rate": len(skipped) / total,
        "number_skipped": len(skipped),
        "number_reranked": len(reranked),
        "quality_delta_vs_best": mean_f1 - BEST_QUALITY_BASELINE["mean_f1"],
        "latency_delta_vs_best": latency_mean - BEST_QUALITY_BASELINE["latency"],
        "quality_delta_vs_no_reranker": mean_f1 - NO_RERANKER_BASELINE["mean_f1"],
        "latency_delta_vs_no_reranker": latency_mean - NO_RERANKER_BASELINE["latency"],
        "quality_delta_vs_fast_fusion": mean_f1 - FAST_FUSION_BASELINE["mean_f1"],
        "latency_delta_vs_fast_fusion": latency_mean - FAST_FUSION_BASELINE["latency"],
        "notes": "exploratory_threshold_grid; overfitting_risk_308_questions",
    }


def write_decisions(
    report_dir: Path,
    output_root: Path,
    threshold_pairs: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    decisions = []
    path = report_dir / "threshold_skip_decisions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for threshold_top1, threshold_gap in threshold_pairs:
            suffix = threshold_suffix(threshold_top1, threshold_gap)
            result_dir = output_root / suffix
            for row in load_predictions(result_dir / "predictions.jsonl"):
                item = {
                    "experiment": suffix,
                    "threshold_top1": threshold_top1,
                    "threshold_gap": threshold_gap,
                    "question_id": row.get("question_id"),
                    "question_type": row.get("type") or row.get("original_type"),
                    "question": row.get("question"),
                    "top1_score": row.get("threshold_skip_top1"),
                    "top2_score": row.get("threshold_skip_top2"),
                    "gap": row.get("threshold_skip_gap"),
                    "relative_gap": row.get("threshold_skip_relative_gap"),
                    "skip_reranker": row.get("threshold_skip_reranker"),
                    "route_used": row.get("threshold_skip_route_used"),
                    "answer": row.get("generated") or row.get("generated_answer"),
                    "ground_truth": row.get("expected") or row.get("expected_answer"),
                    "f1": row.get("f1"),
                    "latency": row.get("latency"),
                    "reranker_latency": row.get("rerank_latency") or row.get("latency_rerank"),
                }
                decisions.append(item)
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return decisions


def _float_or_none(value: Any) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if _float_or_none(row.get("mean_f1")) is not None]
    if not completed:
        return {"quality": None, "balanced": None, "fast": None}
    quality = max(completed, key=lambda row: float(row["mean_f1"]))
    balanced_candidates = [
        row for row in completed if float(row["mean_f1"]) >= BEST_QUALITY_BASELINE["mean_f1"] - 0.01
    ]
    balanced = (
        min(balanced_candidates, key=lambda row: float(row["latency_mean"]))
        if balanced_candidates
        else None
    )
    fast_candidates = [row for row in completed if float(row["mean_f1"]) >= 0.67]
    fast = (
        min(fast_candidates, key=lambda row: float(row["latency_mean"]))
        if fast_candidates
        else None
    )
    return {"quality": quality, "balanced": balanced, "fast": fast}


def write_reports(
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    report_dir: Path,
    output_root: Path,
    *,
    limit: int | None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "threshold_skip_grid_results.csv"
    fieldnames = [
        "threshold_top1",
        "threshold_gap",
        "mean_f1",
        "exact_match",
        "f1_gt_0_5",
        "mm_t_f1",
        "mm_f_f1",
        "latency_mean",
        "latency_p50",
        "latency_p95",
        "skip_rate",
        "number_skipped",
        "number_reranked",
        "quality_delta_vs_best",
        "latency_delta_vs_best",
        "quality_delta_vs_no_reranker",
        "latency_delta_vs_no_reranker",
        "quality_delta_vs_fast_fusion",
        "latency_delta_vs_fast_fusion",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = select_best(rows)
    top1_dist = score_distribution(decisions, "top1_score")
    gap_dist = score_distribution(decisions, "gap")
    skipped = [row for row in decisions if row.get("skip_reranker") is True]
    total = max(len(decisions), 1)
    grid_lines = [
        "| top1 | gap | Mean F1 | EM | F1 > 0.5 | MM-T | MM-F | latency | skip rate | notes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        grid_lines.append(
            f"| {row['threshold_top1']} | {row['threshold_gap']} | {row['mean_f1']} | "
            f"{row['exact_match']} | {row['f1_gt_0_5']} | {row['mm_t_f1']} | "
            f"{row['mm_f_f1']} | {row['latency_mean']} | {row['skip_rate']} | {row['notes']} |"
        )
    text = [
        "# Threshold Skip Reranking Summary",
        "",
        "Статус: additional exploratory experiment.",
        "",
        "## 1. Что реализовано",
        "",
        "Добавлен отдельный режим `threshold_skip`: перед запуском дорогого text+image Nemotron VL reranker проверяется уверенность первичного retrieval. Если `top1_score >= threshold_top1` и `gap >= threshold_gap`, реранкер пропускается.",
        "",
        "## 2. Проверенные threshold",
        "",
        f"- threshold pairs: `{THRESHOLD_GRID}`",
        f"- questions per threshold: `{limit if limit is not None else 'all'}`",
        "",
        "## 3. Используемый score",
        "",
        "Используется только `RetrievalCandidate.score` после первого retrieval. `fusion_score` и `rerank_score` не используются для принятия skip/run решения.",
        "",
        "## 4. Распределение retrieval scores",
        "",
        f"- top1 distribution: `{top1_dist}`",
        f"- gap distribution: `{gap_dist}`",
        "",
        "## 5. Skip frequency",
        "",
        f"- total decisions: {len(decisions)}",
        f"- skipped decisions: {len(skipped)}",
        f"- skipped share across all completed grid rows: {len(skipped) / total:.4f}",
        "",
        "## 6. Grid results",
        "",
        *grid_lines,
        "",
        "## 7. Лучшие threshold",
        "",
        f"- Best Quality Threshold: `{best['quality']}`",
        f"- Best Balanced Threshold: `{best['balanced']}`",
        f"- Best Fast Threshold: `{best['fast']}`",
        "",
        "## 8. Baselines",
        "",
        "| Baseline | Mean F1 | Latency |",
        "|---|---:|---:|",
        f"| Best quality baseline | {BEST_QUALITY_BASELINE['mean_f1']} | {BEST_QUALITY_BASELINE['latency']} |",
        f"| No-reranker full image+text baseline | {NO_RERANKER_BASELINE['mean_f1']} | {NO_RERANKER_BASELINE['latency']} |",
        f"| Fast fusion baseline | {FAST_FUSION_BASELINE['mean_f1']} | {FAST_FUSION_BASELINE['latency']} |",
        "",
        "## 9. Методологические ограничения",
        "",
        "- Threshold подбирается на небольшом subset из 308 вопросов.",
        "- Возможен overfitting под DocBench multimodal subset.",
        "- Результат следует считать exploratory experiment.",
        "- Основной baseline статьи не заменять без подтверждения на другом split.",
        "",
        "## 10. Вывод для статьи и защиты",
        "",
        "Если Threshold Skip сохраняет Mean F1 близко к лучшему baseline и снижает latency, его можно показывать на защите как простой quality/latency baseline перед Adaptive Reranking. В статью включать только как дополнительный exploratory результат, не как замену основного результата.",
        "",
        "## Output",
        "",
        f"- results root: `{output_root}`",
        f"- CSV: `{csv_path}`",
        f"- decisions: `{report_dir / 'threshold_skip_decisions.jsonl'}`",
        "",
    ]
    (report_dir / "threshold_skip_summary.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    effective_limit = None if args.full else args.limit
    base_config = load_config(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir = args.output_root / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    threshold_pairs = THRESHOLD_GRID
    rows = []
    for threshold_top1, threshold_gap in threshold_pairs:
        config = make_config(base_config, threshold_top1, threshold_gap, args.output_root)
        if effective_limit is not None:
            config.setdefault("evaluation", {})["limit"] = effective_limit
        suffix = threshold_suffix(threshold_top1, threshold_gap)
        config_path = generated_config_dir / f"{suffix}.json"
        result_dir = args.output_root / suffix
        dump_config(config, config_path)
        if not (args.skip_existing and (result_dir / "metrics.json").exists()):
            run_config(config_path, dry_run=args.dry_run)
        rows.append(metrics_row(threshold_top1, threshold_gap, result_dir))

    decisions = write_decisions(args.report_dir, args.output_root, threshold_pairs)
    write_reports(rows, decisions, args.report_dir, args.output_root, limit=effective_limit)
    print(f"Saved: {args.report_dir / 'threshold_skip_grid_results.csv'}")
    print(f"Saved: {args.report_dir / 'threshold_skip_decisions.jsonl'}")
    print(f"Saved: {args.report_dir / 'threshold_skip_summary.md'}")


if __name__ == "__main__":
    main()
