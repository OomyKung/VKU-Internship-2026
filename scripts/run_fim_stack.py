"""Run the active GraphSAGE + Leiden + RIS + IC + EA+Memetic FIM stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.permutations import (  # noqa: E402
    FIMPermutationRunConfig,
    format_compact_experiment_summary,
    format_fim_permutation_report,
    get_fim_permutation_spec,
    run_fim_permutation_benchmark_from_config,
)
from fim_hybrid.removed_swarm import ensure_active_optimizer, ensure_active_stack_name  # noqa: E402
from scripts.run_experiment import build_dataset_config  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _sanitized_summary_frame(frame):
    clean = frame.copy()
    for column in clean.columns:
        lowered = str(column).lower()
        if "path" in lowered or lowered == "diagnostics_json":
            clean[column] = ""
    return clean


def _first_row(frame):
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _fmt(value, default: str = "n/a") -> str:
    if value is None:
        return default
    try:
        import pandas as pd  # noqa: PLC0415

        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_requested_outputs(raw_summary, clean_summary, output_dir: Path, report_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_row = _first_row(raw_summary)
    clean_row = _first_row(clean_summary)
    clean_summary.to_csv(output_dir / "result.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(clean_summary.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
    seed_values = []
    try:
        seed_values = json.loads(str(raw_row.get("seed_set", "[]")))
    except Exception:
        seed_values = []
    seed_rows = [{"rank": index + 1, "node": node_id} for index, node_id in enumerate(seed_values)]
    import pandas as pd  # noqa: PLC0415

    pd.DataFrame(seed_rows).to_csv(output_dir / "selected_seed_set.csv", index=False)
    score_table_path = str(raw_row.get("score_table_path", "") or "")
    if score_table_path:
        source = Path(score_table_path)
        if source.exists():
            shutil.copyfile(source, output_dir / "candidate_score_table.csv")


def _format_requested_sections(frame, config: FIMPermutationRunConfig) -> str:
    row = _first_row(frame)
    inactive = row.get("guidance_inactive_components", "[]")
    try:
        inactive_values = json.loads(str(inactive))
    except Exception:
        inactive_values = []
    inactive_text = ", ".join(str(value) for value in inactive_values) if inactive_values else "None"
    try:
        graphsage_effective = float(row.get("graphsage_effective_weight", 0.0) or 0.0)
    except (TypeError, ValueError):
        graphsage_effective = 0.0
    graphsage_usage = (
        "disabled"
        if graphsage_effective <= 1e-12
        else "candidate ranking / mutation / initialization"
    )
    lines = [
        "GraphSAGE Training",
        "-" * 60,
        f"{'Training Target':<30}: {_fmt(row.get('graphsage_training_target'))}",
        f"{'Hidden Dim':<30}: {_fmt(row.get('graphsage_hidden_dim'))}",
        f"{'Embedding Dim':<30}: {_fmt(row.get('graphsage_embedding_dim'))}",
        f"{'Epochs Run':<30}: {_fmt(row.get('graphsage_epochs_run'))}",
        f"{'Best Epoch':<30}: {_fmt(row.get('graphsage_best_epoch'))}",
        f"{'Train Loss':<30}: {_fmt(row.get('graphsage_train_loss'))}",
        f"{'Validation Loss':<30}: {_fmt(row.get('graphsage_validation_loss'))}",
        f"{'Spearman':<30}: {_fmt(row.get('graphsage_spearman'))}",
        f"{'Precision@Budget':<30}: {_fmt(row.get('graphsage_precision_at_budget'))}",
        f"{'Prediction Std':<30}: {_fmt(row.get('graphsage_prediction_std'))}",
        f"{'GraphSAGE Effective Weight':<30}: {_fmt(row.get('graphsage_effective_weight'))}",
        f"{'MF Gain In Label':<30}: {_fmt(row.get('include_mf_gain_in_graphsage_label'))}",
        f"{'MF Gain Label Weight':<30}: {_fmt(row.get('mf_gain_label_weight'))}",
    ]
    warning = str(row.get("graphsage_warning", "") or "").strip()
    if warning and warning.lower() != "nan":
        lines.append(f"{'Warning':<30}: {warning}")
    lines.extend(
        [
            "",
            "Candidate Scoring",
            "-" * 60,
            f"{'GraphSAGE Score Weight':<30}: {_fmt(row.get('graphsage_score_weight'))}",
            f"{'RIS Score Weight':<30}: {_fmt(row.get('ris_score_weight'))}",
            f"{'Fair RIS Score Weight':<30}: {_fmt(row.get('fair_ris_score_weight'))}",
            f"{'Shortfall Gain Weight':<30}: {_fmt(row.get('shortfall_gain_weight'))}",
            f"{'Inactive Components':<30}: {inactive_text}",
            "",
            "EA+Memetic Guidance",
            "-" * 60,
            f"{'GraphSAGE Used For':<30}: {graphsage_usage}",
            f"{'Fair RIS Used For':<30}: fitness / repair / local search",
            f"{'Monte Carlo Used For':<30}: final evaluation",
            "",
            "Final Result",
            "-" * 60,
            f"{'F-score':<30}: {_fmt(row.get('F-score', row.get('f_score')))}",
            f"{'MF':<30}: {_fmt(row.get('MF', row.get('mf')))}",
            f"{'DCV_shortfall':<30}: {_fmt(row.get('dcv_shortfall'))}",
            f"{'DCV_disparity':<30}: {_fmt(row.get('dcv_disparity'))}",
            f"{'Target Coverage Ratio':<30}: {_fmt(row.get('target_coverage_ratio'))}",
            f"{'Target Alpha':<30}: {_fmt(row.get('target_alpha'))}",
            f"{'Spread':<30}: {_fmt(row.get('total_spread'))}",
            f"{'Runtime':<30}: {_fmt(row.get('runtime_seconds'))}",
        ]
    )
    try:
        solved_shortfall = (
            float(row.get("dcv_shortfall", 1.0) or 0.0) <= 1e-9
            and float(row.get("target_coverage_ratio", 0.0) or 0.0) >= 1.0 - 1e-9
        )
    except (TypeError, ValueError):
        solved_shortfall = False
    if solved_shortfall:
        lines.append("Shortfall fairness target satisfied. Optimizing MF is now the next objective.")
        try:
            if float(row.get("MF", row.get("mf", 1.0)) or 0.0) < 0.10:
                lines.append("MF remains low because the weakest group has low normalized influence even though its shortfall target is met.")
        except (TypeError, ValueError):
            pass
    lines.extend(
        [
            "",
            "MF-Lift Diagnostics",
            "-" * 60,
            f"{'Enabled':<30}: {_fmt(row.get('mf_lift_enabled', row.get('use_mf_lift')))}",
            f"{'Started':<30}: {_fmt(row.get('mf_lift_started'))}",
            f"{'Reason Not Started':<30}: {_fmt(row.get('mf_lift_reason_not_started'))}",
            f"{'Initial MF':<30}: {_fmt(row.get('mf_lift_initial_mf'))}",
            f"{'Final Approx MF':<30}: {_fmt(row.get('mf_lift_final_approx_mf'))}",
            f"{'Final MC MF':<30}: {_fmt(row.get('mf_lift_final_mc_mf', row.get('MF', row.get('mf'))))}",
            f"{'MC Selection':<30}: {_fmt(row.get('mf_lift_mc_selection'))}",
            f"{'Pre-lift MC MF':<30}: {_fmt(row.get('mf_lift_pre_mc_mf'))}",
            f"{'Weakest Group Before':<30}: {_fmt(row.get('mf_lift_weakest_group_before'))}",
            f"{'Weakest Group After':<30}: {_fmt(row.get('mf_lift_weakest_group_after_mc', row.get('mf_lift_weakest_group_after')))}",
            f"{'MF-Lift Rounds':<30}: {_fmt(row.get('mf_lift_rounds'))}",
            f"{'MF-Lift Attempts':<30}: {_fmt(row.get('mf_lift_attempts'))}",
            f"{'MF-Lift Successful Swaps':<30}: {_fmt(row.get('mf_lift_successful_swaps'))}",
            f"{'Rejected Target Loss':<30}: {_fmt(row.get('mf_lift_rejected_target_loss'))}",
            f"{'Rejected DCV Shortfall Worsen':<30}: {_fmt(row.get('mf_lift_rejected_dcv_shortfall_worsen'))}",
            f"{'Rejected MF Drop':<30}: {_fmt(row.get('mf_lift_rejected_mf_drop'))}",
            f"{'DCV Shortfall Preserved':<30}: {_fmt(row.get('mf_lift_dcv_shortfall_preserved'))}",
            f"{'Target Coverage Preserved':<30}: {_fmt(row.get('mf_lift_target_coverage_preserved'))}",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="graph_spa_500_0")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--attributes-path", "--attribute-path", dest="attributes_path", default=None)
    parser.add_argument("--dataset-format", choices=["auto", "pickle", "pkl", "txt", "csv"], default="auto")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--node-id-col", default=None)
    parser.add_argument("--protected-attribute", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--embedding-method", default="graphsage")
    parser.add_argument("--community-method", default="leiden")
    parser.add_argument("--optimizer", "--optimizer-mode", dest="optimizer", default="ea_memetic")
    parser.add_argument("--fim-stack", default="graphsage_leiden_ris_ic_ea_memetic")
    parser.add_argument(
        "--graphsage-training-target",
        choices=["fairness_ris_score", "shortfall_gain", "combined_fairness_gain"],
        default="combined_fairness_gain",
    )
    parser.add_argument("--graphsage-hidden-dim", type=int, default=32)
    parser.add_argument("--graphsage-embedding-dim", type=int, default=64)
    parser.add_argument("--graphsage-epochs", type=int, default=100)
    parser.add_argument("--graphsage-learning-rate", type=float, default=0.005)
    parser.add_argument("--graphsage-dropout", type=float, default=0.2)
    parser.add_argument("--graphsage-weight-decay", type=float, default=1e-4)
    parser.add_argument("--graphsage-early-stopping-patience", type=int, default=15)
    parser.add_argument("--graphsage-train-ratio", type=float, default=0.70)
    parser.add_argument("--graphsage-val-ratio", type=float, default=0.15)
    parser.add_argument("--graphsage-loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--graphsage-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate-graphsage-cache", action="store_true")
    parser.add_argument("--allow-structural-fallback", action="store_true")
    parser.add_argument("--allow-protected-features-in-ml", action="store_true")
    parser.add_argument("--ris-num-rr-sets", type=int, default=512)
    parser.add_argument("--ris-mode", choices=["standard", "weak_group_weighted", "group_balanced"], default="weak_group_weighted")
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=300)
    parser.add_argument("--candidate-pool-fraction", type=float, default=0.60)
    parser.add_argument("--group-candidate-floor", type=int, default=30)
    parser.add_argument("--shortfall-repair-rounds", type=int, default=12)
    parser.add_argument("--shortfall-repair-candidate-limit", type=int, default=400)
    parser.add_argument("--shortfall-repair-aggressive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-mf-lift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mf-lift-rounds", type=int, default=5)
    parser.add_argument("--mf-lift-candidate-limit", type=int, default=300)
    parser.add_argument("--mf-lift-weight", type=float, default=3.0)
    parser.add_argument("--mf-lift-disparity-tolerance", type=float, default=0.02)
    parser.add_argument("--mf-lift-spread-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--mf-lift-require-shortfall-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mf-lift-require-target-coverage", type=float, default=1.0)
    parser.add_argument("--mf-drop-tolerance", type=float, default=0.0005)
    parser.add_argument("--shortfall-dcv-worsen-tolerance", type=float, default=0.0001)
    parser.add_argument("--include-mf-gain-in-graphsage-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mf-gain-label-weight", type=float, default=1.5)
    parser.add_argument("--target-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--propagation-prob", type=float, default=0.01)
    parser.add_argument("--output-dir", default="results/fim_stack")
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--primary-dcv-mode", choices=["disparity", "shortfall"], default="shortfall")
    parser.add_argument("--shortfall-dcv-weight", type=float, default=1.0)
    parser.add_argument("--disparity-dcv-weight", type=float, default=0.25)
    parser.add_argument("--graphsage-score-weight", type=float, default=0.8)
    parser.add_argument("--ris-score-weight", type=float, default=1.0)
    parser.add_argument("--fair-ris-score-weight", type=float, default=2.0)
    parser.add_argument("--shortfall-gain-weight", type=float, default=3.0)
    parser.add_argument("--community-diversity-weight", type=float, default=0.5)
    parser.add_argument("--spread-proxy-weight", type=float, default=0.3)
    parser.add_argument("--auto-downweight-bad-ml", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-graphsage-guidance", action="store_true")
    parser.add_argument("--disable-shortfall-gain", action="store_true")
    parser.add_argument("--disable-fair-ris-guidance", action="store_true")
    parser.add_argument("--disable-community-diversity", action="store_true")
    parser.add_argument("--output-mode", choices=["verbose", "compact"], default="verbose")
    args = parser.parse_args()
    try:
        args.optimizer = ensure_active_optimizer(args.optimizer)
        args.fim_stack = ensure_active_stack_name(args.fim_stack)
    except ValueError as exc:
        parser.error(str(exc))
    if str(args.embedding_method).strip().lower() != "graphsage":
        parser.error("The active framework uses --embedding-method graphsage.")
    if str(args.community_method).strip().lower() != "leiden":
        parser.error("The active framework uses --community-method leiden.")
    if args.fim_stack != "graphsage_leiden_ris_ic_ea_memetic":
        parser.error("The active framework primary stack is graphsage_leiden_ris_ic_ea_memetic.")
    return args


def main() -> int:
    args = parse_args()
    output_dir = _resolve_repo_path(args.output_dir)
    dataset_config = build_dataset_config(args)
    spec = get_fim_permutation_spec("graphsage_leiden_ris_ic_ea_memetic")
    config = FIMPermutationRunConfig(
        protected_attribute=str(args.protected_attribute),
        budget=int(args.budget),
        propagation_probability=float(args.propagation_prob),
        mc_runs_search=int(args.mc_runs_search),
        mc_runs_eval=int(args.mc_runs_eval),
        population_size=int(args.population_size),
        generations=int(args.generations),
        max_candidate_pool_size=int(args.candidate_pool_size),
        candidate_pool_fraction=float(args.candidate_pool_fraction),
        min_group_candidate_floor=int(args.group_candidate_floor),
        shortfall_repair_rounds=int(args.shortfall_repair_rounds),
        shortfall_repair_candidate_limit=int(args.shortfall_repair_candidate_limit),
        shortfall_repair_aggressive=bool(args.shortfall_repair_aggressive),
        use_mf_lift=bool(args.use_mf_lift),
        mf_lift_rounds=int(args.mf_lift_rounds),
        mf_lift_candidate_limit=int(args.mf_lift_candidate_limit),
        mf_lift_weight=float(args.mf_lift_weight),
        mf_lift_disparity_tolerance=float(args.mf_lift_disparity_tolerance),
        mf_lift_spread_drop_tolerance=float(args.mf_lift_spread_drop_tolerance),
        mf_lift_require_shortfall_zero=bool(args.mf_lift_require_shortfall_zero),
        mf_lift_require_target_coverage=float(args.mf_lift_require_target_coverage),
        mf_drop_tolerance=float(args.mf_drop_tolerance),
        shortfall_dcv_worsen_tolerance=float(args.shortfall_dcv_worsen_tolerance),
        include_mf_gain_in_graphsage_label=bool(args.include_mf_gain_in_graphsage_label),
        mf_gain_label_weight=float(args.mf_gain_label_weight),
        target_alpha=float(args.target_alpha),
        random_seed=int(args.random_seed),
        output_dir=output_dir,
        continue_on_error=False,
        raise_errors=True,
        graphsage_training_target=str(args.graphsage_training_target),
        graphsage_hidden_dim=int(args.graphsage_hidden_dim),
        graphsage_embedding_dim=int(args.graphsage_embedding_dim),
        graphsage_epochs=int(args.graphsage_epochs),
        graphsage_learning_rate=float(args.graphsage_learning_rate),
        graphsage_dropout=float(args.graphsage_dropout),
        graphsage_weight_decay=float(args.graphsage_weight_decay),
        graphsage_early_stopping_patience=int(args.graphsage_early_stopping_patience),
        graphsage_train_ratio=float(args.graphsage_train_ratio),
        graphsage_val_ratio=float(args.graphsage_val_ratio),
        graphsage_loss=str(args.graphsage_loss),
        graphsage_cache=bool(args.graphsage_cache),
        regenerate_graphsage_cache=bool(args.regenerate_graphsage_cache),
        allow_structural_fallback=bool(args.allow_structural_fallback),
        allow_protected_features_in_ml=bool(args.allow_protected_features_in_ml),
        ris_num_rr_sets=int(args.ris_num_rr_sets),
        ris_mode=str(args.ris_mode),
        ris_score_weight=float(args.ris_score_weight),
        fair_ris_score_weight=float(args.fair_ris_score_weight),
        graphsage_score_weight=float(args.graphsage_score_weight),
        shortfall_gain_weight=float(args.shortfall_gain_weight),
        community_diversity_weight=float(args.community_diversity_weight),
        spread_proxy_weight=float(args.spread_proxy_weight),
        auto_downweight_bad_ml=bool(args.auto_downweight_bad_ml),
        disable_graphsage_guidance=bool(args.disable_graphsage_guidance),
        disable_shortfall_gain=bool(args.disable_shortfall_gain),
        disable_fair_ris_guidance=bool(args.disable_fair_ris_guidance),
        disable_community_diversity=bool(args.disable_community_diversity),
        use_ris=True,
        use_fair_ris=True,
        force_ris_for_all_stacks=True,
        require_ris=True,
        ranking_policy="professor_priority",
        primary_dcv_mode=str(args.primary_dcv_mode),
        shortfall_dcv_weight=float(args.shortfall_dcv_weight),
        disparity_dcv_weight=float(args.disparity_dcv_weight),
        fscore_mode="shortfall_primary" if str(args.primary_dcv_mode) == "shortfall" else "disparity_primary",
        output_mode=str(args.output_mode),
    )
    result = run_fim_permutation_benchmark_from_config(dataset_config, config, permutations=[spec])
    raw_summary = result.summary_frame.copy()
    clean_summary = _sanitized_summary_frame(result.summary_frame)
    requested_sections = _format_requested_sections(clean_summary, config)
    if result.comparison_csv_path is not None:
        clean_summary.to_csv(result.comparison_csv_path, index=False)
    if result.report_path is not None:
        result.report_path.write_text(format_fim_permutation_report(clean_summary, config), encoding="utf-8")
    if output_dir is not None:
        _write_requested_outputs(
            raw_summary,
            clean_summary,
            output_dir,
            requested_sections + "\n\n" + format_fim_permutation_report(clean_summary, config),
        )
    if args.save_json:
        if output_dir is None:
            raise ValueError("--output-dir is required when --save-json is used.")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "fim_stack_summary.json"
        json_path.write_text(
            json.dumps(clean_summary.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8",
        )
        print("JSON Summary                  : saved")
    print(requested_sections)
    print(format_compact_experiment_summary(clean_summary, config))
    if result.comparison_csv_path is not None:
        print("CSV Results                   : saved")
    if result.report_path is not None:
        print("Report                        : saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
