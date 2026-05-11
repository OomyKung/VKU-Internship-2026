"""Run the balanced FIM scalability suite and summarize saved results."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


BALANCED_CONFIGS = {
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
}


def _mode_config(dataset: str, mode: str) -> dict[str, object]:
    config = dict(BALANCED_CONFIGS[dataset])
    if mode == "fast":
        config["rr_sets"] = 512
        config["mc_runs_eval"] = 300
        config["population"] = 20
        config["generations"] = 10
        config["graphsage_epochs"] = 30
        config["shortfall_repair_rounds"] = max(4, int(config["shortfall_repair_rounds"]) // 2)
        config["mf_lift_rounds"] = max(2, int(config["mf_lift_rounds"]) // 2)
    elif mode == "quality":
        config["rr_sets"] = 2048
        config["mc_runs_eval"] = 3000
        config["population"] = 60
        config["generations"] = 50
        config["graphsage_epochs"] = 100
        config["shortfall_repair_rounds"] = int(config["shortfall_repair_rounds"]) + 4
        config["shortfall_repair_candidate_limit"] = int(config["shortfall_repair_candidate_limit"]) + 100
        config["mf_lift_rounds"] = int(config["mf_lift_rounds"]) + 2
        config["mf_lift_candidate_limit"] = int(config["mf_lift_candidate_limit"]) + 100
    return config


def _build_run_command(config: dict[str, object], seed: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_fim_stack.py"),
        "--dataset",
        str(config["dataset"]),
        "--protected-attribute",
        str(config["protected"]),
        "--budget",
        str(config["budget"]),
        "--ris-num-rr-sets",
        str(config["rr_sets"]),
        "--mc-runs-eval",
        str(config["mc_runs_eval"]),
        "--population-size",
        str(config["population"]),
        "--generations",
        str(config["generations"]),
        "--graphsage-hidden-dim",
        str(config["graphsage_hidden"]),
        "--graphsage-embedding-dim",
        str(config["graphsage_embedding"]),
        "--graphsage-epochs",
        str(config["graphsage_epochs"]),
        "--target-alpha",
        str(config["target_alpha"]),
        "--candidate-pool-size",
        str(config["candidate_pool"]),
        "--shortfall-repair-rounds",
        str(config["shortfall_repair_rounds"]),
        "--shortfall-repair-candidate-limit",
        str(config["shortfall_repair_candidate_limit"]),
        "--mf-lift-rounds",
        str(config["mf_lift_rounds"]),
        "--mf-lift-candidate-limit",
        str(config["mf_lift_candidate_limit"]),
        "--random-seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--embedding-method",
        "graphsage",
        "--graphsage-training-target",
        "combined_fairness_gain",
        "--community-method",
        "leiden",
        "--optimizer",
        "ea_memetic",
        "--use-mf-lift",
        "--include-mf-gain-in-graphsage-label",
        "--save-json",
        "--output-mode",
        "compact",
    ]


def _build_summary_command(output_root: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "summarize_scalability_results.py"),
        "--results-root",
        str(output_root),
        "--output-dir",
        str(output_root / "summary"),
        "--print-table",
        "--save-csv",
        "--save-json",
        "--save-markdown",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["synth2", "rice_subset", "twitter"])
    parser.add_argument("--mode", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--multi-seed-twitter", action="store_true")
    parser.add_argument("--output-root", default="results/scalability_suite")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later commands if one fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    for dataset in args.datasets:
        if dataset not in BALANCED_CONFIGS:
            supported = ", ".join(sorted(BALANCED_CONFIGS))
            raise ValueError(f"Unsupported scalability dataset '{dataset}'. Supported: {supported}.")
        config = _mode_config(dataset, args.mode)
        if dataset == "twitter" and args.multi_seed_twitter:
            seeds = args.seeds if args.seeds is not None else [7, 21, 42, 77, 99]
        else:
            seeds = args.seeds if args.seeds is not None else [42]
        for seed in seeds:
            output_dir = output_root / f"scalability_{dataset}_seed_{seed}_{args.mode}"
            commands.append(_build_run_command(config, int(seed), output_dir))

    commands.append(_build_summary_command(output_root))

    for command in commands:
        print("Running:", " ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            if args.continue_on_error:
                print(f"Command failed with exit code {completed.returncode}; continuing.", file=sys.stderr)
                continue
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
