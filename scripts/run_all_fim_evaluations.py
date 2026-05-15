"""Run FIM evaluations across datasets with per-run wall-clock runtime tracking.

Each subprocess call to run_fim_stack.py is timed individually so that repeated
runs produce a full per-run runtime table alongside aggregated statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Dataset balanced configurations ───────────────────────────────────────────

ALL_CONFIGS: dict[str, dict] = {
    "synth2": {
        "dataset": "synth2",
        "protected": "color",
        "budget": 40,
        "rr_sets": 1024,
        "mc_runs_eval": 1000,
        "population": 30,
        "generations": 20,
        "graphsage_hidden": 16,
        "graphsage_embedding": 32,
        "graphsage_epochs": 50,
        "target_alpha": 0.8,
        "candidate_pool": 250,
        "shortfall_repair_rounds": 8,
        "shortfall_repair_candidate_limit": 300,
        "mf_lift_rounds": 4,
        "mf_lift_candidate_limit": 200,
    },
    "synth3": {
        "dataset": "synth3",
        "protected": "color",
        "budget": 40,
        "rr_sets": 1024,
        "mc_runs_eval": 1000,
        "population": 30,
        "generations": 20,
        "graphsage_hidden": 16,
        "graphsage_embedding": 32,
        "graphsage_epochs": 50,
        "target_alpha": 0.8,
        "candidate_pool": 300,
        "shortfall_repair_rounds": 8,
        "shortfall_repair_candidate_limit": 300,
        "mf_lift_rounds": 4,
        "mf_lift_candidate_limit": 200,
    },
    "rice_subset": {
        "dataset": "rice_subset",
        "protected": "color",
        "budget": 70,
        "rr_sets": 1024,
        "mc_runs_eval": 1000,
        "population": 30,
        "generations": 20,
        "graphsage_hidden": 16,
        "graphsage_embedding": 32,
        "graphsage_epochs": 50,
        "target_alpha": 0.8,
        "candidate_pool": 400,
        "shortfall_repair_rounds": 8,
        "shortfall_repair_candidate_limit": 350,
        "mf_lift_rounds": 4,
        "mf_lift_candidate_limit": 250,
    },
    "twitter": {
        "dataset": "twitter",
        "protected": "color",
        "budget": 120,
        "rr_sets": 1024,
        "mc_runs_eval": 1000,
        "population": 30,
        "generations": 20,
        "graphsage_hidden": 16,
        "graphsage_embedding": 32,
        "graphsage_epochs": 50,
        "target_alpha": 0.8,
        "candidate_pool": 500,
        "shortfall_repair_rounds": 6,
        "shortfall_repair_candidate_limit": 250,
        "mf_lift_rounds": 3,
        "mf_lift_candidate_limit": 150,
    },
    "graph_spa_500_0": {
        "dataset": "graph_spa_500_0",
        "protected": "region",
        "budget": 50,
        "rr_sets": 768,
        "mc_runs_eval": 1000,
        "population": 20,
        "generations": 12,
        "graphsage_hidden": 16,
        "graphsage_embedding": 32,
        "graphsage_epochs": 30,
        "target_alpha": 0.8,
        "candidate_pool": 120,
        "shortfall_repair_rounds": 4,
        "shortfall_repair_candidate_limit": 60,
        "mf_lift_rounds": 2,
        "mf_lift_candidate_limit": 60,
    },
}

TWITTER_MULTI_SEEDS = [7, 21, 42, 77, 99]
DEFAULT_SEED = 42
REPEAT_RUNS_BASE_SEED = 42

CSV_COLUMNS = [
    "run_index",
    "dataset",
    "protected_attribute",
    "budget",
    "seed",
    "mode",
    "status",
    "return_code",
    "algorithm_runtime_seconds",
    "runner_wallclock_seconds",
    "f_score",
    "mf",
    "dcv",
    "target_coverage_ratio",
    "spread",
    "output_dir",
]


# ── Config helpers ─────────────────────────────────────────────────────────────

def _mode_config(base: dict, mode: str) -> dict:
    cfg = dict(base)
    if mode == "fast":
        cfg["rr_sets"] = 512
        cfg["mc_runs_eval"] = 300
        cfg["population"] = 20
        cfg["generations"] = 10
        cfg["graphsage_epochs"] = 30
        cfg["shortfall_repair_rounds"] = max(4, int(cfg["shortfall_repair_rounds"]) // 2)
        cfg["mf_lift_rounds"] = max(2, int(cfg["mf_lift_rounds"]) // 2)
    elif mode == "quality":
        cfg["rr_sets"] = 2048
        cfg["mc_runs_eval"] = 3000
        cfg["population"] = 60
        cfg["generations"] = 50
        cfg["graphsage_epochs"] = 100
        cfg["shortfall_repair_rounds"] = int(cfg["shortfall_repair_rounds"]) + 4
        cfg["shortfall_repair_candidate_limit"] = int(cfg["shortfall_repair_candidate_limit"]) + 100
        cfg["mf_lift_rounds"] = int(cfg["mf_lift_rounds"]) + 2
        cfg["mf_lift_candidate_limit"] = int(cfg["mf_lift_candidate_limit"]) + 100
    return cfg


def _build_run_command(config: dict, seed: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_fim_stack.py"),
        "--dataset", str(config["dataset"]),
        "--protected-attribute", str(config["protected"]),
        "--budget", str(config["budget"]),
        "--ris-num-rr-sets", str(config["rr_sets"]),
        "--mc-runs-eval", str(config["mc_runs_eval"]),
        "--population-size", str(config["population"]),
        "--generations", str(config["generations"]),
        "--graphsage-hidden-dim", str(config["graphsage_hidden"]),
        "--graphsage-embedding-dim", str(config["graphsage_embedding"]),
        "--graphsage-epochs", str(config["graphsage_epochs"]),
        "--target-alpha", str(config["target_alpha"]),
        "--candidate-pool-size", str(config["candidate_pool"]),
        "--shortfall-repair-rounds", str(config["shortfall_repair_rounds"]),
        "--shortfall-repair-candidate-limit", str(config["shortfall_repair_candidate_limit"]),
        "--mf-lift-rounds", str(config["mf_lift_rounds"]),
        "--mf-lift-candidate-limit", str(config["mf_lift_candidate_limit"]),
        "--random-seed", str(seed),
        "--output-dir", str(output_dir),
        "--embedding-method", "graphsage",
        "--graphsage-training-target", "combined_fairness_gain",
        "--community-method", "leiden",
        "--optimizer", "ea_memetic",
        "--use-mf-lift",
        "--include-mf-gain-in-graphsage-label",
        "--save-json",
        "--output-mode", "compact",
    ]


# ── Argument parsing helpers ───────────────────────────────────────────────────

def _split_str_arg(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for v in values:
        result.extend(s.strip() for s in v.split(",") if s.strip())
    return result or None


def _split_int_arg(values: list[str] | None) -> list[int] | None:
    parts = _split_str_arg(values)
    return [int(p) for p in parts] if parts is not None else None


# ── Result-reading helpers ─────────────────────────────────────────────────────

def _read_algorithm_runtime(output_dir: Path) -> float | None:
    """Read runtime_seconds from summary.json, falling back to result.csv."""
    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            row = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
            if row is not None:
                val = row.get("runtime_seconds")
                if val is not None:
                    return float(val)
        except Exception:
            pass
    result_file = output_dir / "result.csv"
    if result_file.exists():
        try:
            with open(result_file, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    raw = row.get("runtime_seconds", "")
                    if raw:
                        return float(raw)
        except Exception:
            pass
    return None


def _read_run_metrics(output_dir: Path) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "f_score": None,
        "mf": None,
        "dcv": None,
        "target_coverage_ratio": None,
        "spread": None,
    }
    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            row = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
            if row is not None:
                for key in ("f_score", "mf", "dcv", "target_coverage_ratio"):
                    val = row.get(key)
                    if val is not None:
                        metrics[key] = float(val)
                spread_val = row.get("total_spread") if row.get("total_spread") is not None else row.get("spread")
                if spread_val is not None:
                    metrics["spread"] = float(spread_val)
        except Exception:
            pass
    return metrics


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(val: float | None, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def _compute_stats(
    values: list[float],
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    if not values:
        return None, None, None, None
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, std, min(values), max(values)


# ── Terminal output ────────────────────────────────────────────────────────────

_TABLE_WIDTH = 100


def _print_runtime_table(records: list[dict]) -> None:
    visible = [r for r in records if r["status"] != "DRY_RUN"]
    if not visible:
        return
    print()
    print("Repeated Run Runtime Summary")
    print("-" * _TABLE_WIDTH)
    print(
        f"{'Run':<4}  {'Dataset':<14}  {'Attr':<10}  {'Seed':<6}  {'Status':<8}  "
        f"{'AlgoRuntime(s)':<16}  {'WallClock(s)':<14}  {'F-score':<8}  {'MF':<8}  {'DCV':<8}"
    )
    print("-" * _TABLE_WIDTH)
    for r in visible:
        print(
            f"{r['run_index']:<4}  "
            f"{r['dataset']:<14}  "
            f"{r['protected_attribute']:<10}  "
            f"{r['seed']:<6}  "
            f"{r['status']:<8}  "
            f"{_fmt(r['algorithm_runtime_seconds']):<16}  "
            f"{_fmt(r['runner_wallclock_seconds']):<14}  "
            f"{_fmt(r['f_score'], 4):<8}  "
            f"{_fmt(r['mf'], 4):<8}  "
            f"{_fmt(r['dcv'], 4):<8}"
        )
    print("-" * _TABLE_WIDTH)


def _print_runtime_stats(records: list[dict]) -> None:
    completed = [r for r in records if r["status"] == "OK"]
    failed = [r for r in records if r["status"] == "FAILED"]
    skipped = [r for r in records if r["status"] == "SKIPPED"]

    algo_times = [r["algorithm_runtime_seconds"] for r in completed if r["algorithm_runtime_seconds"] is not None]
    wall_times = [r["runner_wallclock_seconds"] for r in completed if r["runner_wallclock_seconds"] is not None]

    a_mean, a_std, a_min, a_max = _compute_stats(algo_times)
    w_mean, w_std, w_min, w_max = _compute_stats(wall_times)

    print()
    print("Runtime Statistics")
    print("-" * 62)
    print(f"{'Completed Runs':<32} : {len(completed)}")
    print(f"{'Failed Runs':<32} : {len(failed)}")
    if skipped:
        print(f"{'Skipped Runs':<32} : {len(skipped)}")
    print(f"{'Algorithm Runtime Mean':<32} : {_fmt(a_mean)} s")
    print(f"{'Algorithm Runtime Std':<32} : {_fmt(a_std)} s")
    print(f"{'Algorithm Runtime Min':<32} : {_fmt(a_min)} s")
    print(f"{'Algorithm Runtime Max':<32} : {_fmt(a_max)} s")
    print(f"{'Wall-Clock Runtime Mean':<32} : {_fmt(w_mean)} s")
    print(f"{'Wall-Clock Runtime Std':<32} : {_fmt(w_std)} s")
    print(f"{'Wall-Clock Runtime Min':<32} : {_fmt(w_min)} s")
    print(f"{'Wall-Clock Runtime Max':<32} : {_fmt(w_max)} s")
    print("-" * 62)


# ── Log-file writers ───────────────────────────────────────────────────────────

def _save_runtime_log_csv(records: list[dict], output_root: Path) -> None:
    path = output_root / "repeated_run_runtime_log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Runtime CSV  saved: {path}")


def _save_runtime_log_json(records: list[dict], output_root: Path) -> None:
    path = output_root / "repeated_run_runtime_log.json"
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"Runtime JSON saved: {path}")


def _save_runtime_log_markdown(records: list[dict], output_root: Path) -> None:
    completed = [r for r in records if r["status"] == "OK"]
    failed = [r for r in records if r["status"] == "FAILED"]
    algo_times = [r["algorithm_runtime_seconds"] for r in completed if r["algorithm_runtime_seconds"] is not None]
    wall_times = [r["runner_wallclock_seconds"] for r in completed if r["runner_wallclock_seconds"] is not None]
    a_mean, a_std, a_min, a_max = _compute_stats(algo_times)
    w_mean, w_std, w_min, w_max = _compute_stats(wall_times)

    visible = [r for r in records if r["status"] != "DRY_RUN"]
    lines = [
        "# Repeated Run Runtime Report",
        "",
        f"**Completed runs:** {len(completed)}  ",
        f"**Failed runs:** {len(failed)}",
        "",
        "## Per-Run Results",
        "",
        "| Run | Dataset | Attr | Seed | Status | AlgoRuntime (s) | WallClock (s)"
        " | F-score | MF | DCV | Coverage | Spread |",
        "| --- | ------- | ---- | ---- | ------ | --------------- | -------------"
        " | ------- | -- | --- | -------- | ------ |",
    ]
    for r in visible:
        lines.append(
            f"| {r['run_index']} | {r['dataset']} | {r['protected_attribute']}"
            f" | {r['seed']} | {r['status']}"
            f" | {_fmt(r['algorithm_runtime_seconds'])}"
            f" | {_fmt(r['runner_wallclock_seconds'])}"
            f" | {_fmt(r['f_score'], 4)}"
            f" | {_fmt(r['mf'], 4)}"
            f" | {_fmt(r['dcv'], 4)}"
            f" | {_fmt(r['target_coverage_ratio'], 4)}"
            f" | {_fmt(r['spread'], 3)} |"
        )
    lines += [
        "",
        "## Runtime Statistics",
        "",
        "| Metric | Algorithm Runtime (s) | Wall-Clock Runtime (s) |",
        "| ------ | --------------------- | ---------------------- |",
        f"| Mean   | {_fmt(a_mean)} | {_fmt(w_mean)} |",
        f"| Std    | {_fmt(a_std)} | {_fmt(w_std)} |",
        f"| Min    | {_fmt(a_min)} | {_fmt(w_min)} |",
        f"| Max    | {_fmt(a_max)} | {_fmt(w_max)} |",
    ]
    path = output_root / "repeated_run_runtime_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Runtime MD   saved: {path}")


def _flush_logs(records: list[dict], output_root: Path) -> None:
    _save_runtime_log_csv(records, output_root)
    _save_runtime_log_json(records, output_root)
    _save_runtime_log_markdown(records, output_root)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to evaluate (space or comma separated). Default: all.",
    )
    parser.add_argument(
        "--protected-attributes", nargs="+", default=None, dest="protected_attributes",
        help="Override protected attribute(s) for every dataset (space or comma separated).",
    )
    parser.add_argument(
        "--mode", choices=["fast", "balanced", "quality"], default="balanced",
        help="Preset quality mode. Default: balanced.",
    )
    parser.add_argument(
        "--seeds", nargs="+", default=None,
        help="Explicit seeds to run (space or comma separated). Overrides --repeat-runs.",
    )
    parser.add_argument(
        "--repeat-runs", type=int, default=None, dest="repeat_runs", metavar="N",
        help=(
            "Generate N repeated runs with seeds %(metavar)s=42,43,...,42+N-1. "
            "Ignored when --seeds is also provided."
        ),
    )
    parser.add_argument(
        "--multi-seed-twitter", action="store_true", dest="multi_seed_twitter",
        help="Run Twitter with seeds [7,21,42,77,99] when no seeds are specified.",
    )
    parser.add_argument(
        "--output-root", default="results/fim_evaluations",
        help="Root directory for all run outputs and runtime logs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--skip-existing", action="store_true", dest="skip_existing",
        help="Skip runs whose output directory already contains summary.json or result.csv.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", dest="continue_on_error",
        help="Keep running subsequent runs even if one fails.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Resolve datasets
    datasets = _split_str_arg(args.datasets) or list(ALL_CONFIGS.keys())
    for ds in datasets:
        if ds not in ALL_CONFIGS:
            supported = ", ".join(sorted(ALL_CONFIGS))
            print(f"ERROR: Unknown dataset '{ds}'. Supported: {supported}.", file=sys.stderr)
            return 1

    # Resolve protected-attribute override
    protected_override = _split_str_arg(args.protected_attributes)

    # Resolve seeds
    explicit_seeds = _split_int_arg(args.seeds)
    if explicit_seeds is not None and args.repeat_runs is not None:
        print(
            "WARNING: --repeat-runs is ignored because --seeds was also provided.",
            file=sys.stderr,
        )
        global_seeds: list[int] | None = explicit_seeds
    elif args.repeat_runs is not None:
        global_seeds = list(range(REPEAT_RUNS_BASE_SEED, REPEAT_RUNS_BASE_SEED + args.repeat_runs))
        print(f"Repeat-run mode: {args.repeat_runs} runs with seeds {global_seeds}")
    elif explicit_seeds is not None:
        global_seeds = explicit_seeds
    else:
        global_seeds = None  # resolved per dataset below

    # Resolve output root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Build run specification list
    run_specs: list[dict] = []
    for dataset in datasets:
        base = ALL_CONFIGS[dataset]
        attr_list = protected_override or [base["protected"]]
        for attr in attr_list:
            if dataset == "twitter" and args.multi_seed_twitter and global_seeds is None:
                ds_seeds = TWITTER_MULTI_SEEDS
            else:
                ds_seeds = global_seeds if global_seeds is not None else [DEFAULT_SEED]
            for seed in ds_seeds:
                cfg = _mode_config({**base, "protected": attr}, args.mode)
                output_dir = output_root / f"{dataset}_{attr}_seed_{seed}_{args.mode}"
                run_specs.append(
                    {
                        "dataset": dataset,
                        "protected_attribute": attr,
                        "budget": cfg["budget"],
                        "seed": seed,
                        "mode": args.mode,
                        "config": cfg,
                        "output_dir": output_dir,
                    }
                )

    if not run_specs:
        print("No runs to execute.", file=sys.stderr)
        return 0

    total = len(run_specs)
    print(f"\nTotal runs planned : {total}")
    print(f"Output root        : {output_root}\n")

    records: list[dict] = []

    for i, spec in enumerate(run_specs):
        run_index = i + 1
        output_dir: Path = spec["output_dir"]

        # ── Skip-existing check ──────────────────────────────────────────────
        if args.skip_existing and not args.dry_run:
            if (output_dir / "summary.json").exists() or (output_dir / "result.csv").exists():
                print(f"[{run_index}/{total}] SKIP  {output_dir.name}")
                algo_rt = _read_algorithm_runtime(output_dir)
                metrics = _read_run_metrics(output_dir)
                records.append(
                    {
                        "run_index": run_index,
                        "dataset": spec["dataset"],
                        "protected_attribute": spec["protected_attribute"],
                        "budget": spec["budget"],
                        "seed": spec["seed"],
                        "mode": spec["mode"],
                        "status": "SKIPPED",
                        "return_code": 0,
                        "algorithm_runtime_seconds": algo_rt,
                        "runner_wallclock_seconds": None,
                        **metrics,
                        "output_dir": str(output_dir),
                    }
                )
                continue

        # ── Build & display command ──────────────────────────────────────────
        command = _build_run_command(spec["config"], spec["seed"], output_dir)
        print(
            f"[{run_index}/{total}] {spec['dataset']} / {spec['protected_attribute']}"
            f" / seed={spec['seed']}"
        )
        print("  Command:", " ".join(command))

        # ── Dry-run path ─────────────────────────────────────────────────────
        if args.dry_run:
            records.append(
                {
                    "run_index": run_index,
                    "dataset": spec["dataset"],
                    "protected_attribute": spec["protected_attribute"],
                    "budget": spec["budget"],
                    "seed": spec["seed"],
                    "mode": spec["mode"],
                    "status": "DRY_RUN",
                    "return_code": None,
                    "algorithm_runtime_seconds": None,
                    "runner_wallclock_seconds": None,
                    "f_score": None,
                    "mf": None,
                    "dcv": None,
                    "target_coverage_ratio": None,
                    "spread": None,
                    "output_dir": str(output_dir),
                }
            )
            continue

        # ── Execute with wall-clock timing ───────────────────────────────────
        t_start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        runner_wallclock_seconds = time.perf_counter() - t_start

        success = completed.returncode == 0

        # algorithm_runtime_seconds comes from the child's saved outputs;
        # runner_wallclock_seconds is measured here in the orchestrating process.
        algorithm_runtime_seconds = _read_algorithm_runtime(output_dir)
        metrics = _read_run_metrics(output_dir) if success else {
            "f_score": None, "mf": None, "dcv": None,
            "target_coverage_ratio": None, "spread": None,
        }

        record: dict = {
            "run_index": run_index,
            "dataset": spec["dataset"],
            "protected_attribute": spec["protected_attribute"],
            "budget": spec["budget"],
            "seed": spec["seed"],
            "mode": spec["mode"],
            "status": "OK" if success else "FAILED",
            "return_code": completed.returncode,
            "algorithm_runtime_seconds": algorithm_runtime_seconds,
            "runner_wallclock_seconds": runner_wallclock_seconds,
            **metrics,
            "output_dir": str(output_dir),
        }
        records.append(record)

        print(
            f"  -> {record['status']} | "
            f"algo={_fmt(algorithm_runtime_seconds)} s | "
            f"wall={_fmt(runner_wallclock_seconds)} s"
        )

        if not success:
            if args.continue_on_error:
                print(
                    f"  Run {run_index} failed (exit {completed.returncode}); continuing.",
                    file=sys.stderr,
                )
            else:
                _flush_logs(records, output_root)
                _print_runtime_table(records)
                _print_runtime_stats(records)
                return int(completed.returncode)

    # ── Persist runtime logs ─────────────────────────────────────────────────
    if not args.dry_run and records:
        print()
        _flush_logs(records, output_root)

    # ── Print terminal summary ───────────────────────────────────────────────
    _print_runtime_table(records)
    _print_runtime_stats(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
