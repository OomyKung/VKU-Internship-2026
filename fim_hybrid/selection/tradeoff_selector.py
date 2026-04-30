"""Fairness-runtime trade-off selection for ML-FIM benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from fim_hybrid.priority_policy import (
    PROFESSOR_PRIORITY,
    ProfessorPriorityConfig,
    fairness_gate_notes,
    fairness_valid_mask,
    normalize_ranking_policy,
    professor_priority_warning,
    rank_frame_professor_priority,
)


@dataclass(frozen=True, slots=True)
class TradeoffSelectionConfig:
    """Thresholds and behavior for fairness-runtime stack selection."""

    close_fscore_threshold: float = 0.003
    min_f_score: float = 0.0
    min_mf: float = 0.0001
    max_dcv: float = 0.25
    min_fraction_groups_covered: float = 0.80
    scalability_required: bool = True
    runtime_tiebreak_only: bool = True
    warn_only_fairness_gates: bool = False
    runtime_priority_when_close: bool = True
    max_dcv_delta_vs_best: float = 0.01
    min_mf_ratio_vs_best: float = 0.95
    selection_policy: str = "fairness_runtime_tradeoff"
    quality_runtime_lambda: float = 0.0
    fallback_stack: str | None = None


def _professor_config_from_selection(config: TradeoffSelectionConfig) -> ProfessorPriorityConfig:
    return ProfessorPriorityConfig(
        fairness_close_threshold=float(config.close_fscore_threshold),
        scalability_required=bool(config.scalability_required),
        runtime_tiebreak_only=bool(config.runtime_tiebreak_only),
        min_f_score=float(config.min_f_score),
        min_mf=float(config.min_mf),
        max_dcv=float(config.max_dcv),
        min_fraction_groups_covered=float(config.min_fraction_groups_covered),
        warn_only_fairness_gates=bool(config.warn_only_fairness_gates),
    )


@dataclass(frozen=True, slots=True)
class StackPipelineConfig:
    """Executable pipeline metadata for one selected stack."""

    stack_name: str
    community_method: str
    embedding_method: str
    ranking_model: str
    spread_estimator_search: str
    spread_estimator_final: str
    optimizer_mode: str
    use_repair: bool = False
    use_swap_local_search: bool = False
    use_ml_scores_in_initialization: bool = True
    use_ml_scores_in_mutation: bool = True
    use_ml_scores_in_repair: bool = True
    use_ml_scores_in_local_search: bool = True
    ml_score_weight: float = 1.0
    ris_score_weight: float = 1.0
    fair_ris_score_weight: float = 0.5
    fairness_bonus_weight: float = 0.0
    diversity_bonus_weight: float = 0.0
    weak_group_bonus_weight: float = 0.0
    protected_group_coverage_weight: float = 0.0
    dcv_penalty_weight: float = 2.5


@dataclass(frozen=True, slots=True)
class SelectedStackConfig:
    """Serializable selected-stack configuration."""

    selected_stack: str
    protected_attribute: str
    budget: int
    reason: str
    pipeline: StackPipelineConfig
    selection_policy: str = "fairness_runtime_tradeoff"
    benchmark_row: dict[str, Any] = field(default_factory=dict)
    decision_status: str = "selected_from_benchmark"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(asdict(self.pipeline))
        payload["pipeline"] = asdict(self.pipeline)
        return payload

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_ready(self.to_dict()), indent=2), encoding="utf-8")
        return output_path


@dataclass(frozen=True, slots=True)
class TradeoffSelectionResult:
    """Selection result plus rejected-row diagnostics."""

    selected_config: SelectedStackConfig
    fallback_config: SelectedStackConfig
    normalized_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    rejected_frame: pd.DataFrame


STACK_PIPELINE_CONFIGS: dict[str, StackPipelineConfig] = {
    "community_aware_fair_greedy": StackPipelineConfig(
        stack_name="community_aware_fair_greedy",
        community_method="leiden",
        embedding_method="none",
        ranking_model="fairness_weighted_greedy",
        spread_estimator_search="monte_carlo",
        spread_estimator_final="monte_carlo",
        optimizer_mode="local_search",
        use_swap_local_search=True,
    ),
    "community_fair_greedy_baseline": StackPipelineConfig(
        stack_name="community_fair_greedy_baseline",
        community_method="leiden",
        embedding_method="none",
        ranking_model="fairness_weighted_greedy",
        spread_estimator_search="monte_carlo",
        spread_estimator_final="monte_carlo",
        optimizer_mode="local_search",
        use_repair=True,
        use_swap_local_search=True,
    ),
    "fairness_first_scalable_ml_siea": StackPipelineConfig(
        stack_name="fairness_first_scalable_ml_siea",
        community_method="leiden",
        embedding_method="graphsage",
        ranking_model="graphsage_plus_fair_ris",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.8,
        ris_score_weight=0.8,
        fair_ris_score_weight=1.2,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.0,
        protected_group_coverage_weight=1.0,
    ),
    "graphsage_community_siea": StackPipelineConfig(
        stack_name="graphsage_community_siea",
        community_method="leiden",
        embedding_method="graphsage",
        ranking_model="graphsage",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.7,
        ris_score_weight=0.7,
        fair_ris_score_weight=1.3,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.2,
        protected_group_coverage_weight=1.0,
        dcv_penalty_weight=2.5,
    ),
    "gcn_community_siea": StackPipelineConfig(
        stack_name="gcn_community_siea",
        community_method="leiden",
        embedding_method="gcn",
        ranking_model="gcn",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.6,
        ris_score_weight=0.7,
        fair_ris_score_weight=1.4,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.3,
        protected_group_coverage_weight=1.0,
        dcv_penalty_weight=2.5,
    ),
    "node2vec_xgboost_community_siea": StackPipelineConfig(
        stack_name="node2vec_xgboost_community_siea",
        community_method="leiden",
        embedding_method="node2vec",
        ranking_model="xgboost",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.8,
        ris_score_weight=0.7,
        fair_ris_score_weight=1.2,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.2,
        protected_group_coverage_weight=1.2,
        dcv_penalty_weight=2.5,
    ),
    "node2vec_xgboost_fair_siea": StackPipelineConfig(
        stack_name="node2vec_xgboost_fair_siea",
        community_method="leiden",
        embedding_method="node2vec",
        ranking_model="xgboost",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.8,
        ris_score_weight=0.8,
        fair_ris_score_weight=1.2,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.0,
        protected_group_coverage_weight=1.0,
    ),
    "gcn_fair_siea": StackPipelineConfig(
        stack_name="gcn_fair_siea",
        community_method="leiden",
        embedding_method="gcn",
        ranking_model="gcn_plus_fair_ris",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
        ml_score_weight=0.8,
        ris_score_weight=0.8,
        fair_ris_score_weight=1.2,
        fairness_bonus_weight=1.0,
        diversity_bonus_weight=0.4,
        weak_group_bonus_weight=1.0,
        protected_group_coverage_weight=1.0,
    ),
    "graphsage_fair_ris_hybrid": StackPipelineConfig(
        stack_name="graphsage_fair_ris_hybrid",
        community_method="leiden",
        embedding_method="graphsage",
        ranking_model="graphsage_plus_fair_ris",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
    ),
    "gcn_fair_ris_hybrid": StackPipelineConfig(
        stack_name="gcn_fair_ris_hybrid",
        community_method="leiden",
        embedding_method="gcn",
        ranking_model="gcn_plus_fair_ris",
        spread_estimator_search="fairness_aware_ris",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
    ),
    "node2vec_xgboost": StackPipelineConfig(
        stack_name="node2vec_xgboost",
        community_method="leiden",
        embedding_method="node2vec",
        ranking_model="xgboost",
        spread_estimator_search="monte_carlo",
        spread_estimator_final="monte_carlo",
        optimizer_mode="hybrid_si_ea",
        use_repair=True,
        use_swap_local_search=True,
    ),
    "deepwalk_mlp": StackPipelineConfig(
        stack_name="deepwalk_mlp",
        community_method="leiden",
        embedding_method="deepwalk",
        ranking_model="mlp",
        spread_estimator_search="monte_carlo",
        spread_estimator_final="monte_carlo",
        optimizer_mode="greedy",
    ),
    "line_fast_ml": StackPipelineConfig(
        stack_name="line_fast_ml",
        community_method="leiden",
        embedding_method="line",
        ranking_model="logistic_regression",
        spread_estimator_search="monte_carlo",
        spread_estimator_final="monte_carlo",
        optimizer_mode="greedy",
    ),
}

_ALIASES = {
    "permutation_name": "stack_name",
    "method_name": "stack_name",
    "method": "method",
    "protectedattribute": "protected_attribute",
    "F-score": "f_score",
    "f_score": "f_score",
    "mean_f_score": "f_score",
    "fscore": "f_score",
    "MF": "mf",
    "mf": "mf",
    "mean_mf": "mf",
    "DCV": "dcv",
    "dcv": "dcv",
    "mean_dcv": "dcv",
    "spread": "total_spread",
    "total_spread": "total_spread",
    "mean_spread": "total_spread",
    "runtime": "runtime_seconds",
    "runtime_seconds": "runtime_seconds",
    "mean_runtime": "runtime_seconds",
    "zero_covered_groups": "zero_covered_groups_count",
    "zero_covered_groups_count": "zero_covered_groups_count",
    "fraction_groups_covered": "fraction_groups_covered",
    "scalability_pass": "scalability_pass",
}
_METRIC_REQUIRED_COLUMNS = ("stack_name", "f_score", "mf", "dcv", "runtime_seconds")
_SUCCESS_STATUSES = {"ok", "success", "successful", "succeeded", "complete", "completed"}


def _canonical_column(column_name: object) -> str:
    text = str(column_name).strip()
    compact = "".join(character.lower() if character.isalnum() else "_" for character in text)
    compact = "_".join(part for part in compact.split("_") if part)
    return _ALIASES.get(text, _ALIASES.get(compact, compact))


def _first_available(frame: pd.DataFrame, column_name: str) -> pd.Series:
    duplicate_frame = frame.loc[:, frame.columns == column_name]
    if duplicate_frame.empty:
        raise KeyError(column_name)
    return duplicate_frame.bfill(axis=1).iloc[:, 0]


def normalize_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported raw/aggregate benchmark schemas."""

    if frame.empty:
        raise ValueError("Benchmark CSV contains no rows.")
    renamed = frame.copy()
    renamed.columns = [_canonical_column(column) for column in renamed.columns]
    if "stack_name" not in renamed.columns and "method" in renamed.columns:
        renamed["stack_name"] = renamed["method"]

    normalized = pd.DataFrame(index=renamed.index)
    for column_name in dict.fromkeys(renamed.columns):
        normalized[column_name] = _first_available(renamed, column_name)
    missing = [column for column in _METRIC_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(
            "Benchmark CSV is missing required metric columns after alias normalization: "
            f"{missing}. Required aliases include stack_name, f_score/F-score, mf/MF, "
            "dcv/DCV, and runtime_seconds/runtime."
        )
    if "total_spread" not in normalized.columns:
        normalized["total_spread"] = pd.NA
    if "status" not in normalized.columns:
        normalized["status"] = "ok"

    for metric in ("budget", "f_score", "mf", "dcv", "total_spread", "runtime_seconds", "zero_covered_groups_count", "fraction_groups_covered"):
        if metric in normalized.columns:
            normalized[metric] = pd.to_numeric(normalized[metric], errors="coerce")
    if "scalability_pass" in normalized.columns:
        normalized["scalability_pass"] = normalized["scalability_pass"].map(
            lambda value: str(value).strip().lower() not in {"false", "0", "no"}
        )
    normalized["status"] = normalized["status"].fillna("").astype(str)
    normalized["stack_name"] = normalized["stack_name"].fillna("").astype(str)
    if "protected_attribute" in normalized.columns:
        normalized["protected_attribute"] = normalized["protected_attribute"].fillna("").astype(str)
    return normalized


def _load_benchmark_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV does not exist: {csv_path}")
    return normalize_benchmark_frame(pd.read_csv(csv_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - best-effort JSON serialization.
            return value
    return value


def _benchmark_row_dict(row: pd.Series) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in row.to_dict().items()}


def pipeline_config_for_stack(stack_name: str) -> StackPipelineConfig:
    """Return executable pipeline metadata for a supported stack name."""

    try:
        return STACK_PIPELINE_CONFIGS[str(stack_name)]
    except KeyError as exc:
        supported = ", ".join(sorted(STACK_PIPELINE_CONFIGS))
        raise ValueError(f"No stack-to-pipeline mapping exists for '{stack_name}'. Supported stacks: {supported}.") from exc


def _pipeline_config_for_stack(stack_name: str) -> StackPipelineConfig:
    return pipeline_config_for_stack(stack_name)


def _normalize_metric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if pd.isna(minimum) or pd.isna(maximum) or maximum <= minimum:
        return pd.Series([0.0] * len(numeric), index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def _with_quality_runtime_score(frame: pd.DataFrame, lambda_runtime: float) -> pd.DataFrame:
    scored = frame.copy()
    scored["normalized_f_score"] = _normalize_metric(scored["f_score"])
    scored["normalized_runtime"] = _normalize_metric(scored["runtime_seconds"])
    scored["quality_runtime_score"] = (
        scored["normalized_f_score"] - float(lambda_runtime) * scored["normalized_runtime"]
    )
    return scored


def _selected_config_from_row(
    row: pd.Series,
    *,
    protected_attribute: str,
    budget: int,
    reason: str,
    selection_policy: str,
    decision_status: str,
) -> SelectedStackConfig:
    selected_stack = str(row["stack_name"])
    return SelectedStackConfig(
        selected_stack=selected_stack,
        protected_attribute=str(protected_attribute),
        budget=int(budget),
        reason=reason,
        pipeline=_pipeline_config_for_stack(selected_stack),
        selection_policy=selection_policy,
        benchmark_row=_benchmark_row_dict(row),
        decision_status=decision_status,
    )


def _valid_fairness_rows(frame: pd.DataFrame, config: TradeoffSelectionConfig) -> pd.DataFrame:
    normalized = normalize_benchmark_frame(frame)
    successful = normalized[
        normalized["status"].astype(str).str.strip().str.lower().isin(_SUCCESS_STATUSES)
    ].copy()
    if successful.empty:
        return successful
    collapse_mask = (
        pd.to_numeric(successful["f_score"], errors="coerce").lt(float(config.min_f_score))
        | pd.to_numeric(successful["mf"], errors="coerce").le(float(config.min_mf))
        | pd.to_numeric(successful["dcv"], errors="coerce").ge(float(config.max_dcv))
    )
    if "fraction_groups_covered" in successful.columns:
        fraction_groups = pd.to_numeric(successful["fraction_groups_covered"], errors="coerce")
        collapse_mask |= fraction_groups.notna() & fraction_groups.lt(float(config.min_fraction_groups_covered))
    return successful[~collapse_mask].dropna(subset=["f_score", "mf", "dcv", "runtime_seconds"]).copy()


def select_fairness_fallback(
    rows: pd.DataFrame,
    candidate_stack: str,
    config: TradeoffSelectionConfig = TradeoffSelectionConfig(),
    *,
    protected_attribute: str,
    budget: int,
) -> SelectedStackConfig:
    """Select a fairness-first fallback, preferring an independent stack."""

    normalized = normalize_benchmark_frame(rows)
    if "protected_attribute" in normalized.columns and "budget" in normalized.columns:
        normalized = normalized[
            normalized["protected_attribute"].astype(str).eq(str(protected_attribute))
            & pd.to_numeric(normalized["budget"], errors="coerce").eq(int(budget))
        ].copy()
    if normalized.empty:
        raise ValueError(
            f"No benchmark rows match protected_attribute='{protected_attribute}' and budget={int(budget)} "
            "for fallback selection."
        )
    valid = _valid_fairness_rows(normalized, config)
    if valid.empty:
        raise ValueError("No successful non-collapse benchmark rows are available for fallback selection.")
    ordered = valid.sort_values(
        ["f_score", "mf", "dcv", "total_spread", "runtime_seconds", "stack_name"],
        ascending=[False, False, True, False, True, True],
        kind="mergesort",
    )
    forced_stack = None if config.fallback_stack is None else str(config.fallback_stack).strip()
    if forced_stack:
        forced_rows = ordered[ordered["stack_name"].astype(str).eq(forced_stack)]
        if forced_rows.empty:
            available = ", ".join(str(value) for value in ordered["stack_name"].drop_duplicates().tolist())
            raise ValueError(
                f"Forced fallback_stack='{forced_stack}' is not a valid successful non-collapse row "
                f"for protected_attribute='{protected_attribute}', budget={int(budget)}. "
                f"Available valid stacks: {available or 'none'}."
            )
        independent = ordered[~ordered["stack_name"].astype(str).eq(str(candidate_stack))]
        if forced_stack == str(candidate_stack) and not independent.empty:
            raise ValueError(
                f"Forced fallback_stack='{forced_stack}' equals candidate_stack, but independent valid "
                "fallback stacks exist. Pick a different --fallback-stack or omit it."
            )
        fallback_row = forced_rows.iloc[0]
        reason = f"forced fairness-first fallback stack from --fallback-stack={forced_stack}"
    else:
        independent = ordered[~ordered["stack_name"].astype(str).eq(str(candidate_stack))]
        if not independent.empty:
            fallback_row = independent.iloc[0]
            reason = "highest valid benchmark F-score independent fairness-first fallback"
        else:
            fallback_row = ordered.iloc[0]
            reason = "No independent fallback stack available; validation is self-comparison only"

    decision_status = (
        "fallback_self_comparison"
        if str(fallback_row["stack_name"]) == str(candidate_stack)
        else "fallback_candidate"
    )
    return _selected_config_from_row(
        fallback_row,
        protected_attribute=protected_attribute,
        budget=budget,
        reason=reason,
        selection_policy=str(config.selection_policy),
        decision_status=decision_status,
    )


def select_stack_from_benchmark_frame(
    frame: pd.DataFrame,
    *,
    protected_attribute: str,
    budget: int,
    config: TradeoffSelectionConfig = TradeoffSelectionConfig(),
) -> TradeoffSelectionResult:
    """Select one stack from a normalized or raw benchmark frame."""

    normalized = normalize_benchmark_frame(frame)
    missing_selection = [column for column in ("protected_attribute", "budget") if column not in normalized.columns]
    if missing_selection:
        raise ValueError(
            "Benchmark CSV contains valid metric columns but cannot be filtered for selection because "
            f"it is missing required selection columns: {missing_selection}."
        )
    filtered = normalized[
        normalized["protected_attribute"].astype(str).eq(str(protected_attribute))
        & pd.to_numeric(normalized["budget"], errors="coerce").eq(int(budget))
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No benchmark rows match protected_attribute='{protected_attribute}' and budget={int(budget)}."
        )
    successful = filtered[
        filtered["status"].astype(str).str.strip().str.lower().isin(_SUCCESS_STATUSES)
    ].copy()
    if successful.empty:
        raise ValueError("No successful benchmark rows are available for selection.")

    policy_config = _professor_config_from_selection(config)
    collapse_mask = ~fairness_valid_mask(successful, policy_config)
    rejected = successful[collapse_mask].copy()
    valid = successful[~collapse_mask].dropna(subset=["f_score", "runtime_seconds"]).copy()
    all_failed_gates = False
    if valid.empty:
        all_failed_gates = True
        valid = successful.dropna(subset=["f_score", "runtime_seconds"]).copy()
        if valid.empty:
            raise ValueError(
                "No successful benchmark rows remain after fairness-collapse rejection "
                f"(min_f_score={config.min_f_score}, min_mf={config.min_mf}, max_dcv={config.max_dcv})."
            )
        warning = professor_priority_warning(successful, policy_config)
        if warning:
            valid["selection_warning"] = warning

    valid = _with_quality_runtime_score(valid, float(config.quality_runtime_lambda))
    valid["fairness_gate_failures"] = fairness_gate_notes(valid, policy_config)
    best_candidates = rank_frame_professor_priority(valid, policy_config)
    best_row = best_candidates.iloc[0]
    policy = normalize_ranking_policy(config.selection_policy)
    if policy not in {"fairness_runtime_tradeoff", "quality_runtime", PROFESSOR_PRIORITY}:
        raise ValueError("selection_policy must be 'fairness_runtime_tradeoff', 'quality_runtime', or 'professor_priority'.")
    if policy == "quality_runtime":
        candidates = valid.sort_values(
            ["quality_runtime_score", "f_score", "mf", "dcv", "runtime_seconds", "stack_name"],
            ascending=[False, False, False, True, True, True],
            kind="mergesort",
        )
        reason = (
            "selected by explicit quality_runtime policy "
            f"(lambda_runtime={float(config.quality_runtime_lambda):.6f})"
        )
    elif policy == PROFESSOR_PRIORITY:
        candidates = best_candidates
        reason = (
            "selected by professor_priority: F-score, MF, DCV, scalability, "
            "spread, extra spread, then runtime"
        )
    else:
        max_f_score = float(best_row["f_score"])
        best_mf = float(best_row["mf"])
        best_dcv = float(best_row["dcv"])
        close = valid[
            ((max_f_score - pd.to_numeric(valid["f_score"], errors="coerce")) <= float(config.close_fscore_threshold))
            & (pd.to_numeric(valid["dcv"], errors="coerce") <= best_dcv + float(config.max_dcv_delta_vs_best))
            & (pd.to_numeric(valid["mf"], errors="coerce") >= best_mf * float(config.min_mf_ratio_vs_best))
        ].copy()
        if close.empty or not bool(config.runtime_priority_when_close):
            candidates = best_candidates
            reason = (
                f"selected highest valid F-score {float(candidates.iloc[0]['f_score']):.6f}; "
                "no runtime-priority row satisfied F-score/MF/DCV closeness"
            )
        else:
            candidates = close.sort_values(
                ["runtime_seconds", "f_score", "mf", "dcv", "total_spread", "stack_name"],
                ascending=[True, False, False, True, False, True],
                kind="mergesort",
            )
            reason = (
                f"within {float(config.close_fscore_threshold):.6f} F-score of best valid stack "
                f"({max_f_score:.6f}), within DCV delta {float(config.max_dcv_delta_vs_best):.6f}, "
                f"MF ratio >= {float(config.min_mf_ratio_vs_best):.6f}, and lowest runtime among close rows"
            )

    selected_row = candidates.iloc[0]
    selected = _selected_config_from_row(
        selected_row,
        protected_attribute=protected_attribute,
        budget=budget,
        reason=reason + ("; warning: all methods failed fairness gates, using least-bad row" if all_failed_gates else ""),
        selection_policy=policy,
        decision_status="selected_with_fairness_gate_warning" if all_failed_gates else "selected_from_benchmark",
    )
    try:
        fallback = select_fairness_fallback(
            valid,
            selected.selected_stack,
            config,
            protected_attribute=protected_attribute,
            budget=budget,
        )
    except ValueError:
        if not all_failed_gates:
            raise
        fallback_row = best_candidates.iloc[1] if len(best_candidates) > 1 else selected_row
        fallback = _selected_config_from_row(
            fallback_row,
            protected_attribute=protected_attribute,
            budget=budget,
            reason="fallback selected from least-bad rows because all methods failed fairness gates",
            selection_policy=policy,
            decision_status="fallback_with_fairness_gate_warning",
        )
    return TradeoffSelectionResult(
        selected_config=selected,
        fallback_config=fallback,
        normalized_frame=normalized,
        valid_frame=valid,
        rejected_frame=rejected,
    )


def select_stack_from_benchmark_csv(
    path: str | Path,
    *,
    protected_attribute: str,
    budget: int,
    config: TradeoffSelectionConfig = TradeoffSelectionConfig(),
) -> TradeoffSelectionResult:
    """Read a benchmark CSV and select one stack by fairness-runtime trade-off."""

    return select_stack_from_benchmark_frame(
        _load_benchmark_csv(path),
        protected_attribute=protected_attribute,
        budget=budget,
        config=config,
    )


def load_selected_stack_config(path: str | Path) -> SelectedStackConfig:
    """Load a JSON config produced by ``SelectedStackConfig.to_json``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pipeline_payload: Mapping[str, Any] = payload.get("pipeline", payload)
    pipeline = StackPipelineConfig(
        stack_name=str(pipeline_payload["stack_name"]),
        community_method=str(pipeline_payload["community_method"]),
        embedding_method=str(pipeline_payload["embedding_method"]),
        ranking_model=str(pipeline_payload["ranking_model"]),
        spread_estimator_search=str(pipeline_payload["spread_estimator_search"]),
        spread_estimator_final=str(pipeline_payload["spread_estimator_final"]),
        optimizer_mode=str(pipeline_payload["optimizer_mode"]),
        use_repair=bool(pipeline_payload.get("use_repair", False)),
        use_swap_local_search=bool(pipeline_payload.get("use_swap_local_search", False)),
        use_ml_scores_in_initialization=bool(pipeline_payload.get("use_ml_scores_in_initialization", True)),
        use_ml_scores_in_mutation=bool(pipeline_payload.get("use_ml_scores_in_mutation", True)),
        use_ml_scores_in_repair=bool(pipeline_payload.get("use_ml_scores_in_repair", True)),
        use_ml_scores_in_local_search=bool(pipeline_payload.get("use_ml_scores_in_local_search", True)),
        ml_score_weight=float(pipeline_payload.get("ml_score_weight", 1.0)),
        ris_score_weight=float(pipeline_payload.get("ris_score_weight", 1.0)),
        fair_ris_score_weight=float(pipeline_payload.get("fair_ris_score_weight", 0.5)),
        fairness_bonus_weight=float(pipeline_payload.get("fairness_bonus_weight", 0.0)),
        diversity_bonus_weight=float(pipeline_payload.get("diversity_bonus_weight", 0.0)),
        weak_group_bonus_weight=float(pipeline_payload.get("weak_group_bonus_weight", 0.0)),
        protected_group_coverage_weight=float(pipeline_payload.get("protected_group_coverage_weight", 0.0)),
        dcv_penalty_weight=float(pipeline_payload.get("dcv_penalty_weight", 2.5)),
    )
    return SelectedStackConfig(
        selected_stack=str(payload.get("selected_stack", pipeline.stack_name)),
        protected_attribute=str(payload["protected_attribute"]),
        budget=int(payload["budget"]),
        reason=str(payload.get("reason", "loaded selected stack config")),
        pipeline=pipeline,
        selection_policy=str(payload.get("selection_policy", "fairness_runtime_tradeoff")),
        benchmark_row=dict(payload.get("benchmark_row", {})),
        decision_status=str(payload.get("decision_status", "selected_from_benchmark")),
    )
