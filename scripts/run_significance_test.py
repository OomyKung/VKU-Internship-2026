"""Run paired significance tests on repeated FIM comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_METRICS = ("f_score", "total_spread", "extra_spread", "mf", "dcv")
DEFAULT_LOWER_IS_BETTER_METRICS = ("dcv",)
DEFAULT_OUTPUT_COLUMNS = [
    "metric",
    "baseline",
    "challenger",
    "status",
    "pair_count",
    "baseline_mean",
    "challenger_mean",
    "mean_delta",
    "median_delta",
    "wins",
    "losses",
    "ties",
    "win_rate",
    "wilcoxon_p",
    "ttest_p",
    "adjusted_p",
    "ci95_low",
    "ci95_high",
    "significant",
    "skip_reason",
]


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _seed_from_name(name: str) -> str | None:
    match = re.search(r"seed[-_](.+)$", name.strip(), re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _infer_seed_from_path(path: Path) -> str | None:
    current: Path | None = path
    while current is not None:
        seed = _seed_from_name(current.name)
        if seed is not None:
            return seed
        current = current.parent if current.parent != current else None
    return None


def _collect_comparison_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    csv_paths = sorted(input_path.rglob("*_permutation_comparison.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No '*_permutation_comparison.csv' files found under {input_path}")
    return csv_paths


def load_significance_input(input_path: Path | str) -> pd.DataFrame:
    """Load one aggregated comparison CSV or a directory of per-seed comparison CSVs."""

    resolved_input = Path(input_path)
    frames: list[pd.DataFrame] = []
    for file_index, csv_path in enumerate(_collect_comparison_csvs(resolved_input), start=1):
        frame = pd.read_csv(csv_path)
        if "seed" not in frame.columns:
            inferred_seed = _infer_seed_from_path(csv_path.parent) or str(file_index)
            frame["seed"] = inferred_seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if "stack_name" not in combined.columns and "permutation_name" in combined.columns:
        combined["stack_name"] = combined["permutation_name"]
    if "status" not in combined.columns:
        combined["status"] = "ok"
    return combined


def _bootstrap_ci(differences: np.ndarray, reps: int, random_seed: int) -> tuple[float, float]:
    if differences.size < 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(reps, dtype=float)
    for index in range(reps):
        sample = rng.choice(differences, size=differences.size, replace=True)
        bootstrap_means[index] = float(sample.mean())
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _adjust_p_values(p_values: Iterable[float], method: str) -> list[float]:
    values = list(float(value) for value in p_values)
    if method == "none":
        return values
    if method == "bonferroni":
        scale = len(values)
        return [float(min(1.0, value * scale)) for value in values]
    if method != "holm":
        raise ValueError(f"Unsupported p-value adjustment method '{method}'.")
    if not values:
        return []
    adjusted = [0.0] * len(values)
    ordered_indices = sorted(range(len(values)), key=lambda index: values[index])
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(ordered_indices):
        scaled = float((total - rank) * values[index])
        running_max = max(running_max, scaled)
        adjusted[index] = float(min(1.0, running_max))
    return adjusted


def _pair_key_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column_name for column_name in ("dataset", "protected_attribute", "seed") if column_name in frame.columns]
    if "seed" not in columns:
        raise ValueError("Significance testing requires repeated paired runs with a 'seed' column.")
    return columns


def _paired_metric_rows(
    frame: pd.DataFrame,
    *,
    baseline: str,
    challenger: str,
    metric: str,
) -> pd.DataFrame:
    pair_keys = _pair_key_columns(frame)
    baseline_frame = (
        frame.loc[frame["stack_name"].astype(str) == baseline, pair_keys + [metric]]
        .rename(columns={metric: "baseline_value"})
        .copy()
    )
    challenger_frame = (
        frame.loc[frame["stack_name"].astype(str) == challenger, pair_keys + [metric]]
        .rename(columns={metric: "challenger_value"})
        .copy()
    )
    return baseline_frame.merge(
        challenger_frame,
        on=pair_keys,
        how="inner",
        validate="one_to_one",
    )


def _paired_test_row(
    frame: pd.DataFrame,
    *,
    baseline: str,
    challenger: str,
    metric: str,
    lower_is_better: bool,
    min_pairs: int,
    bootstrap_reps: int,
    random_seed: int,
) -> dict[str, object]:
    paired = _paired_metric_rows(frame, baseline=baseline, challenger=challenger, metric=metric)
    if len(paired) < int(min_pairs):
        return {
            "metric": metric,
            "baseline": baseline,
            "challenger": challenger,
            "status": "skipped",
            "pair_count": int(len(paired)),
            "baseline_mean": pd.NA,
            "challenger_mean": pd.NA,
            "mean_delta": pd.NA,
            "median_delta": pd.NA,
            "wins": pd.NA,
            "losses": pd.NA,
            "ties": pd.NA,
            "win_rate": pd.NA,
            "wilcoxon_p": pd.NA,
            "ttest_p": pd.NA,
            "adjusted_p": pd.NA,
            "ci95_low": pd.NA,
            "ci95_high": pd.NA,
            "significant": False,
            "skip_reason": f"Need at least {min_pairs} paired runs; found {len(paired)}.",
        }

    baseline_values = paired["baseline_value"].astype(float).to_numpy()
    challenger_values = paired["challenger_value"].astype(float).to_numpy()
    differences = (
        baseline_values - challenger_values
        if lower_is_better
        else challenger_values - baseline_values
    )
    wins = int(np.sum(differences > 0.0))
    losses = int(np.sum(differences < 0.0))
    ties = int(np.sum(np.isclose(differences, 0.0)))
    if np.allclose(differences, 0.0):
        wilcoxon_p = 1.0
        ttest_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(differences, zero_method="pratt", alternative="two-sided").pvalue)
        if differences.size < 2:
            ttest_p = 1.0
        elif np.allclose(differences, differences[0]):
            ttest_p = 0.0
        else:
            ttest_p = float(ttest_rel(challenger_values, baseline_values).pvalue)
    ci_low, ci_high = _bootstrap_ci(differences, reps=int(bootstrap_reps), random_seed=int(random_seed))
    return {
        "metric": metric,
        "baseline": baseline,
        "challenger": challenger,
        "status": "ok",
        "pair_count": int(len(paired)),
        "baseline_mean": float(np.mean(baseline_values)),
        "challenger_mean": float(np.mean(challenger_values)),
        "mean_delta": float(np.mean(differences)),
        "median_delta": float(np.median(differences)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / len(differences)),
        "wilcoxon_p": wilcoxon_p,
        "ttest_p": ttest_p,
        "adjusted_p": pd.NA,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "significant": False,
        "skip_reason": "",
    }


def run_significance_analysis(
    frame: pd.DataFrame,
    *,
    baseline: str,
    metrics: Iterable[str] = DEFAULT_METRICS,
    lower_is_better_metrics: Iterable[str] = DEFAULT_LOWER_IS_BETTER_METRICS,
    min_pairs: int = 5,
    p_adjustment: str = "holm",
    alpha: float = 0.05,
    bootstrap_reps: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute paired significance rows for one baseline versus all other stacks."""

    if "stack_name" not in frame.columns:
        raise ValueError("Input frame must contain a 'stack_name' column.")
    working = frame.copy()
    working = working.loc[working["status"].astype(str) == "ok"].copy()
    if working.empty:
        raise ValueError("No successful comparison rows were found.")
    baseline_name = str(baseline)
    available_stacks = set(working["stack_name"].astype(str))
    if baseline_name not in available_stacks:
        raise ValueError(f"Baseline stack '{baseline_name}' is not present in the input data.")

    lower_is_better = set(str(metric) for metric in lower_is_better_metrics)
    rows: list[dict[str, object]] = []
    challengers = sorted(available_stacks - {baseline_name})
    for metric in metrics:
        metric_name = str(metric)
        if metric_name not in working.columns:
            raise ValueError(f"Metric column '{metric_name}' is not present in the input data.")
        for challenger in challengers:
            rows.append(
                _paired_test_row(
                    working,
                    baseline=baseline_name,
                    challenger=challenger,
                    metric=metric_name,
                    lower_is_better=metric_name in lower_is_better,
                    min_pairs=int(min_pairs),
                    bootstrap_reps=int(bootstrap_reps),
                    random_seed=int(random_seed),
                )
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    for metric_name, metric_frame in result.groupby("metric", sort=False):
        ok_mask = metric_frame["status"].astype(str) == "ok"
        metric_indices = list(metric_frame.index[ok_mask])
        adjusted = _adjust_p_values(
            [float(result.loc[index, "wilcoxon_p"]) for index in metric_indices],
            method=str(p_adjustment),
        )
        for index, adjusted_p in zip(metric_indices, adjusted, strict=True):
            result.loc[index, "adjusted_p"] = float(adjusted_p)
            result.loc[index, "significant"] = bool(float(adjusted_p) < float(alpha))

    for column_name in DEFAULT_OUTPUT_COLUMNS:
        if column_name not in result.columns:
            result[column_name] = pd.NA
    return result.loc[:, DEFAULT_OUTPUT_COLUMNS].copy()


def build_significance_report(
    result_frame: pd.DataFrame,
    *,
    baseline: str,
    p_adjustment: str,
    alpha: float,
) -> str:
    """Render a compact text report for the significance test summary."""

    lines = [
        "FIM Significance Test Summary",
        "=" * 72,
        f"Baseline: {baseline}",
        f"Primary test: Wilcoxon signed-rank",
        f"P-value adjustment: {p_adjustment}",
        f"Alpha: {alpha}",
        "",
    ]
    if result_frame.empty:
        lines.append("No significance rows were produced.")
        return "\n".join(lines)

    for metric_name, metric_frame in result_frame.groupby("metric", sort=False):
        lines.append(metric_name)
        lines.append("-" * 72)
        for row in metric_frame.itertuples(index=False):
            if str(row.status) != "ok":
                lines.append(
                    f"- {row.challenger}: skipped | pairs={row.pair_count} | reason={row.skip_reason}"
                )
                continue
            significance_label = "significant" if bool(row.significant) else "not_significant"
            lines.append(
                f"- {row.challenger}: pairs={row.pair_count} | mean_delta={float(row.mean_delta):.6f} | "
                f"wilcoxon_p={float(row.wilcoxon_p):.6g} | adjusted_p={float(row.adjusted_p):.6g} | "
                f"ci95=[{float(row.ci95_low):.6f}, {float(row.ci95_high):.6f}] | {significance_label}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_significance_report(
    result_frame: pd.DataFrame,
    *,
    output_dir: Path | str,
    baseline: str,
    p_adjustment: str,
    alpha: float,
) -> tuple[Path, Path]:
    """Persist the significance summary as CSV and text."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "fim_significance_summary.csv"
    txt_path = output_path / "fim_significance_summary.txt"
    result_frame.to_csv(csv_path, index=False)
    txt_path.write_text(
        build_significance_report(
            result_frame,
            baseline=baseline,
            p_adjustment=p_adjustment,
            alpha=alpha,
        ),
        encoding="utf-8",
    )
    return csv_path, txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="One aggregated comparison CSV or a directory containing repeated '*_permutation_comparison.csv' files.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline stack name used for paired comparisons.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to test.",
    )
    parser.add_argument(
        "--lower-is-better-metrics",
        nargs="+",
        default=list(DEFAULT_LOWER_IS_BETTER_METRICS),
        help="Metrics where smaller values are better; deltas are sign-flipped so positive still means challenger better.",
    )
    parser.add_argument("--min-pairs", type=int, default=5, help="Minimum paired runs required per comparison.")
    parser.add_argument(
        "--p-adjustment",
        choices=["none", "bonferroni", "holm"],
        default="holm",
        help="Multiple-comparison correction across challengers within each metric.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the significance summary CSV/text files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = _resolve_repo_path(args.input)
    if input_path is None:
        raise ValueError("--input is required.")
    result_frame = run_significance_analysis(
        load_significance_input(input_path),
        baseline=args.baseline,
        metrics=args.metrics,
        lower_is_better_metrics=args.lower_is_better_metrics,
        min_pairs=args.min_pairs,
        p_adjustment=args.p_adjustment,
        alpha=args.alpha,
        bootstrap_reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        output_dir = (input_path.parent if input_path.is_file() else input_path) / "significance"
    csv_path, txt_path = save_significance_report(
        result_frame,
        output_dir=output_dir,
        baseline=args.baseline,
        p_adjustment=args.p_adjustment,
        alpha=args.alpha,
    )
    print(f"Saved significance CSV: {csv_path}")
    print(f"Saved significance report: {txt_path}")
    print("")
    print(build_significance_report(result_frame, baseline=args.baseline, p_adjustment=args.p_adjustment, alpha=args.alpha).rstrip())


if __name__ == "__main__":
    main()
