"""Run a unified ML-focused FIM benchmark with shared final Monte Carlo evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.embeddings.registry import available_embedding_methods  # noqa: E402
from fim_hybrid.ml_training import available_ranking_models  # noqa: E402
from fim_hybrid.permutations import (  # noqa: E402
    FIMPermutationRunConfig,
    FIMPermutationSpec,
    get_fim_permutation_spec,
    run_fim_permutation_benchmark_from_config,
)
from scripts.evaluate_fim_results import (  # noqa: E402
    InsightThresholds,
    build_evaluation_report,
    build_json_summary,
    evaluate_result_frame,
)
from scripts.run_experiment import build_dataset_config  # noqa: E402


DEFAULT_BENCHMARK_NAME = "ml_fim_benchmark"
DEFAULT_STRONG_STACKS = (
    "community_fair_greedy_baseline",
    "fairness_first_scalable_ml_siea",
    "node2vec_xgboost_fair_siea",
    "gcn_fair_siea",
)
WEAK_ML_BASELINES = (
    "deepwalk_mlp",
    "line_fast_ml",
)
DEFAULT_ALL_STACKS = DEFAULT_STRONG_STACKS + (
    "graphsage_fair_ris_hybrid",
    "gcn_fair_ris_hybrid",
    "graphcl_maximin",
    "dgi_fair_ris",
    "vgae_fair_ris",
    "node2vec_xgboost",
    "node2vec_kmeans_logreg_hybrid",
) + WEAK_ML_BASELINES
_VALID_SPREAD_SEARCH = {"monte_carlo", "ris_guidance", "fairness_aware_ris"}
_VALID_FINAL_ESTIMATORS = {"monte_carlo"}
_VALID_OPTIMIZER_MODES = {"greedy", "local_search", "hybrid_si_ea"}


@dataclass(slots=True)
class MLFIMBenchmarkResult:
    raw_comparison_frame: pd.DataFrame
    evaluation_result: object
    report_text: str
    comparison_csv_path: Path | None = None
    normalized_csv_path: Path | None = None
    ranked_csv_path: Path | None = None
    report_path: Path | None = None
    json_path: Path | None = None


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _clone_named_spec(name: str, *, public_name: str | None = None, **updates: object) -> FIMPermutationSpec:
    base = get_fim_permutation_spec(name)
    changes = dict(updates)
    if public_name is not None:
        changes["name"] = public_name
    return replace(base, **changes)


def _named_ml_stack_registry() -> dict[str, FIMPermutationSpec]:
    registry = {
        "community_aware_fair_greedy": get_fim_permutation_spec("community_aware_fair_greedy"),
        "community_fair_greedy_baseline": get_fim_permutation_spec("community_fair_greedy_baseline"),
        "fairness_first_scalable_ml_siea": get_fim_permutation_spec("fairness_first_scalable_ml_siea"),
        "node2vec_xgboost_fair_siea": get_fim_permutation_spec("node2vec_xgboost_fair_siea"),
        "gcn_fair_siea": get_fim_permutation_spec("gcn_fair_siea"),
        "graphsage_community_siea": FIMPermutationSpec(
            name="graphsage_community_siea",
            description="Leiden communities with GraphSAGE guidance, Fair RIS, and Hybrid SI+EA.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="graphsage",
            ranking_model="graphsage",
            optimizer_mode="hybrid_si_ea",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; ml_guided_only=true",
        ),
        "node2vec_xgboost_community_siea": FIMPermutationSpec(
            name="node2vec_xgboost_community_siea",
            description="Leiden communities with Node2Vec+XGBoost guidance and Hybrid SI+EA.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="hybrid_si_ea",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; lightweight_quality_runtime=true; ml_guided_only=true",
        ),
        "gcn_community_siea": FIMPermutationSpec(
            name="gcn_community_siea",
            description="Leiden communities with GCN guidance, Fair RIS, and Hybrid SI+EA.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="gcn",
            ranking_model="gcn",
            optimizer_mode="hybrid_si_ea",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="optional_gnn_comparison=true; ml_guided_only=true",
        ),
        "graphsage_fair_ris_hybrid": _clone_named_spec(
            "leiden_graphsage_fair_ris_hybrid",
            public_name="graphsage_fair_ris_hybrid",
        ),
        "gcn_fair_ris_hybrid": _clone_named_spec(
            "leiden_gcn_fair_ris_hybrid",
            public_name="gcn_fair_ris_hybrid",
        ),
        "graphcl_maximin": _clone_named_spec(
            "infomap_graphcl_logreg_maximin",
            public_name="graphcl_maximin",
        ),
        "node2vec_xgboost": _clone_named_spec(
            "leiden_node2vec_xgboost_hybrid",
            public_name="node2vec_xgboost",
        ),
        "line_fast_ml": _clone_named_spec(
            "leiden_line_logreg_greedy",
            public_name="line_fast_ml",
            notes="weak_baseline=true; not_default=true; fairness_collapse_risk=true; Fast shallow-embedding ML baseline.",
        ),
        "node2vec_kmeans_logreg_hybrid": _clone_named_spec(
            "leiden_node2vec_kmeans_logreg_hybrid",
            public_name="node2vec_kmeans_logreg_hybrid",
        ),
        "dgi_fair_ris": FIMPermutationSpec(
            name="dgi_fair_ris",
            description=(
                "Leiden communities with DGI embeddings, logistic-regression ranking, fair RIS guidance, "
                "and a hybrid SI+EA optimizer."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="dgi",
            ranking_model="logistic_regression",
            optimizer_mode="hybrid_si_ea",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="Embedding-guided DGI stack with logistic-regression ranking fallback.",
        ),
        "vgae_fair_ris": FIMPermutationSpec(
            name="vgae_fair_ris",
            description=(
                "Leiden communities with VGAE embeddings, logistic-regression ranking, fair RIS guidance, "
                "and a hybrid SI+EA optimizer."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="vgae",
            ranking_model="logistic_regression",
            optimizer_mode="hybrid_si_ea",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="Embedding-guided VGAE stack with logistic-regression ranking fallback.",
        ),
        "deepwalk_mlp": FIMPermutationSpec(
            name="deepwalk_mlp",
            description=(
                "Leiden communities with DeepWalk embeddings, MLP ranking, and candidate-restricted greedy search."
            ),
            runner_kind="ranked_greedy",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="monte_carlo",
            spread_estimator_final="monte_carlo",
            embedding_method="deepwalk",
            ranking_model="mlp",
            optimizer_mode="greedy",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            notes="weak_baseline=true; not_default=true; fairness_collapse_risk=true; Shallow embedding plus MLP ranking benchmark stack.",
        ),
    }
    return registry


def available_ml_benchmark_stacks() -> tuple[str, ...]:
    return tuple(_named_ml_stack_registry())


def _ranking_models_for_cli() -> tuple[str, ...]:
    return tuple(model for model in available_ranking_models() if model != "ris_guidance")


def _default_ranking_model(embedding_method: str) -> str:
    mapping = {
        "graphsage": "graphsage",
        "gcn": "gcn",
        "graphcl": "logistic_regression",
        "dgi": "logistic_regression",
        "vgae": "logistic_regression",
        "node2vec": "xgboost",
        "deepwalk": "mlp",
        "line": "logistic_regression",
    }
    return mapping.get(str(embedding_method).strip().lower(), "logistic_regression")


def _apply_stack_filters(
    specs: Sequence[FIMPermutationSpec],
    *,
    embedding_methods: Sequence[str] | None = None,
    ranking_models: Sequence[str] | None = None,
    community_method: str | None = None,
    clustering_method: str | None = None,
    spread_estimator_search: str | None = None,
    spread_estimator_final: str | None = None,
    optimizer_mode: str | None = None,
) -> list[FIMPermutationSpec]:
    filtered = list(specs)
    if embedding_methods:
        allowed = {str(value).strip().lower() for value in embedding_methods}
        filtered = [spec for spec in filtered if spec.embedding_method in allowed]
    if ranking_models:
        allowed = {str(value).strip().lower() for value in ranking_models}
        filtered = [spec for spec in filtered if spec.ranking_model in allowed]
    if community_method is not None:
        filtered = [spec for spec in filtered if spec.community_method == str(community_method).strip().lower()]
    if clustering_method is not None:
        filtered = [spec for spec in filtered if spec.clustering_method == str(clustering_method).strip().lower()]
    if spread_estimator_search is not None:
        filtered = [spec for spec in filtered if spec.spread_estimator_search == str(spread_estimator_search).strip().lower()]
    if spread_estimator_final is not None:
        filtered = [spec for spec in filtered if spec.spread_estimator_final == str(spread_estimator_final).strip().lower()]
    if optimizer_mode is not None:
        filtered = [spec for spec in filtered if spec.optimizer_mode == str(optimizer_mode).strip().lower()]
    return filtered


def _generic_runner_kind(
    *,
    embedding_method: str,
    ranking_model: str,
    optimizer_mode: str,
    spread_estimator_search: str,
) -> tuple[str, str]:
    normalized_embedding = str(embedding_method).strip().lower()
    normalized_ranker = str(ranking_model).strip().lower()
    normalized_optimizer = str(optimizer_mode).strip().lower()
    normalized_search = str(spread_estimator_search).strip().lower()

    if normalized_ranker in {"graphsage", "gcn"}:
        if normalized_embedding != normalized_ranker:
            raise ValueError(
                f"ranking_model='{ranking_model}' requires embedding_method='{ranking_model}' for the GNN-guided path."
            )
        if normalized_optimizer != "hybrid_si_ea" or normalized_search != "fairness_aware_ris":
            raise ValueError(
                f"{ranking_model} benchmark stacks require optimizer_mode='hybrid_si_ea' and "
                "spread_estimator_search='fairness_aware_ris'."
            )
        return "experiment_runner_gnn_ris", "f_score"

    if normalized_optimizer == "hybrid_si_ea":
        return "ranked_hybrid", "f_score"
    if normalized_optimizer == "greedy":
        return "ranked_greedy", "f_score"
    if normalized_optimizer == "local_search":
        return "ranked_maximin", "maximin"

    raise ValueError(f"Unsupported optimizer_mode '{optimizer_mode}'.")


def _generated_stack_name(
    *,
    community_method: str,
    embedding_method: str,
    ranking_model: str,
    optimizer_mode: str,
    clustering_method: str,
    spread_estimator_search: str,
) -> str:
    parts = [community_method, embedding_method, ranking_model]
    if clustering_method not in {"", "none"}:
        parts.append(clustering_method)
    if spread_estimator_search == "fairness_aware_ris":
        parts.append("fair_ris")
    elif spread_estimator_search == "ris_guidance":
        parts.append("ris")
    parts.append(optimizer_mode)
    return "_".join(parts)


def _make_generated_stack_spec(
    *,
    embedding_method: str,
    ranking_model: str,
    community_method: str,
    clustering_method: str,
    spread_estimator_search: str,
    spread_estimator_final: str,
    optimizer_mode: str,
) -> FIMPermutationSpec:
    if spread_estimator_final not in _VALID_FINAL_ESTIMATORS:
        raise ValueError("All benchmark stacks must use spread_estimator_final='monte_carlo'.")
    if spread_estimator_search not in _VALID_SPREAD_SEARCH:
        raise ValueError(f"Unsupported spread_estimator_search '{spread_estimator_search}'.")
    if optimizer_mode not in _VALID_OPTIMIZER_MODES:
        raise ValueError(f"Unsupported optimizer_mode '{optimizer_mode}'.")

    runner_kind, fairness_objective = _generic_runner_kind(
        embedding_method=embedding_method,
        ranking_model=ranking_model,
        optimizer_mode=optimizer_mode,
        spread_estimator_search=spread_estimator_search,
    )
    use_ris_guidance = spread_estimator_search in {"ris_guidance", "fairness_aware_ris"}
    use_fair_ris = spread_estimator_search == "fairness_aware_ris"
    normalized_clustering = str(clustering_method).strip().lower()
    return FIMPermutationSpec(
        name=_generated_stack_name(
            community_method=str(community_method).strip().lower(),
            embedding_method=str(embedding_method).strip().lower(),
            ranking_model=str(ranking_model).strip().lower(),
            optimizer_mode=str(optimizer_mode).strip().lower(),
            clustering_method=normalized_clustering,
            spread_estimator_search=str(spread_estimator_search).strip().lower(),
        ),
        description="CLI-generated ML benchmark stack.",
        runner_kind=runner_kind,
        diffusion_model="ic",
        community_method=str(community_method).strip().lower(),
        spread_estimator_search=str(spread_estimator_search).strip().lower(),
        spread_estimator_final=str(spread_estimator_final).strip().lower(),
        embedding_method=str(embedding_method).strip().lower(),
        clustering_method=normalized_clustering if normalized_clustering else "none",
        clustering_input_mode="embedding" if normalized_clustering not in {"", "none"} else "none",
        ranking_model=str(ranking_model).strip().lower(),
        optimizer_mode=str(optimizer_mode).strip().lower(),
        debias_mode="none",
        fairness_objective=fairness_objective,
        variant_family="clustering" if normalized_clustering not in {"", "none"} else ("fairness" if fairness_objective == "maximin" else "ml"),
        candidate_top_fraction=0.5,
        use_ris_guidance=use_ris_guidance,
        use_fair_ris=use_fair_ris,
        notes="generated_from_cli=true",
    )


def resolve_ml_benchmark_specs(
    *,
    ml_stacks: Sequence[str] | None,
    include_baseline: bool,
    embedding_methods: Sequence[str] | None,
    ranking_models: Sequence[str] | None,
    community_method: str | None,
    clustering_method: str | None,
    spread_estimator_search: str | None,
    spread_estimator_final: str,
    optimizer_mode: str | None,
    include_weak_ml_baselines: bool = False,
) -> list[FIMPermutationSpec]:
    registry = _named_ml_stack_registry()
    requested_tokens = [str(value).strip().lower() for value in (ml_stacks or ["strong_ml"]) if str(value).strip()]

    selected_names: list[str]
    if "all_available" in requested_tokens:
        selected_names = list(DEFAULT_ALL_STACKS)
    elif "strong_ml" in requested_tokens:
        selected_names = list(DEFAULT_STRONG_STACKS)
        if include_weak_ml_baselines:
            selected_names.extend(WEAK_ML_BASELINES)
    else:
        selected_names = requested_tokens

    selected_specs: list[FIMPermutationSpec] = []
    for name in selected_names:
        if name not in registry:
            supported = sorted(set(registry) | {"all_available", "strong_ml"})
            raise ValueError(f"Unknown ML benchmark stack '{name}'. Supported values: {supported}.")
        selected_specs.append(registry[name])

    selected_names_set = {spec.name for spec in selected_specs}
    if include_baseline and not ({"community_aware_fair_greedy", "community_fair_greedy_baseline"} & selected_names_set):
        selected_specs.insert(0, registry["community_aware_fair_greedy"])

    search_filter = None if spread_estimator_search is None else str(spread_estimator_search).strip().lower()
    final_filter = str(spread_estimator_final).strip().lower()
    filtered_specs = _apply_stack_filters(
        selected_specs,
        embedding_methods=embedding_methods,
        ranking_models=ranking_models,
        community_method=community_method,
        clustering_method=clustering_method,
        spread_estimator_search=search_filter,
        spread_estimator_final=final_filter,
        optimizer_mode=optimizer_mode,
    )

    if filtered_specs:
        return filtered_specs

    if not embedding_methods:
        raise ValueError(
            "No benchmark stacks remain after filtering. Provide explicit --embedding-methods to generate benchmark stacks "
            "or relax the filters."
        )

    generated_specs: list[FIMPermutationSpec] = []
    selected_embeddings = [str(value).strip().lower() for value in embedding_methods if str(value).strip()]
    requested_rankers = [str(value).strip().lower() for value in (ranking_models or []) if str(value).strip()]
    for embedding_method in selected_embeddings:
        rankers = requested_rankers or [_default_ranking_model(embedding_method)]
        for ranking_model in rankers:
            generated_specs.append(
                _make_generated_stack_spec(
                    embedding_method=embedding_method,
                    ranking_model=ranking_model,
                    community_method=(community_method or "leiden"),
                    clustering_method=(clustering_method or "none"),
                    spread_estimator_search=(search_filter or ("fairness_aware_ris" if ranking_model in {"graphsage", "gcn"} else "monte_carlo")),
                    spread_estimator_final=final_filter,
                    optimizer_mode=(optimizer_mode or ("hybrid_si_ea" if ranking_model in {"graphsage", "gcn", "xgboost", "logistic_regression"} else "greedy")),
                )
            )
    if include_baseline:
        generated_specs.insert(0, registry["community_aware_fair_greedy"])
    return generated_specs


def _is_baseline_row(row: pd.Series) -> bool:
    variant = str(row.get("variant_type", "")).strip().lower()
    if variant == "interpretable_baseline":
        return True
    return str(row.get("embedding_method", "none")).strip().lower() in {"", "none"}


def _rank_ml_subset(frame: pd.DataFrame, columns: Sequence[str], ascending: Sequence[bool]) -> pd.DataFrame:
    available_columns = [column for column in columns if column in frame.columns and frame[column].notna().any()]
    available_ascending = [ascending[index] for index, column in enumerate(columns) if column in available_columns]
    if not available_columns:
        return frame.sort_values(["stack_name"], ascending=[True], kind="mergesort")
    return frame.sort_values(available_columns + ["stack_name"], ascending=available_ascending + [True], kind="mergesort")


def build_ml_benchmark_insights(
    evaluation_result: object,
    raw_frame: pd.DataFrame,
    *,
    thresholds: InsightThresholds,
) -> tuple[list[str], dict[str, str | None]]:
    ranked = evaluation_result.ranked_frame.copy()
    if ranked.empty:
        return ["No successful benchmark rows were available for ML insight generation."], {
            "best_current_overall_ml_stack": None,
            "best_fairness_first_ml_stack": None,
            "best_spread_first_ml_stack": None,
            "best_fast_ml_stack": None,
            "best_interpretable_baseline": None,
            "strongest_practical_default": None,
            "best_fairness_quality_method": None,
            "best_scalable_method": None,
            "best_spread_method": None,
            "best_runtime_method": None,
            "final_professor_priority_recommendation": None,
            "stacks_needing_more_work": None,
        }

    metric_columns = [
        column_name
        for column_name in ("f_score", "mf", "dcv", "total_spread", "extra_spread", "runtime_seconds")
        if column_name in ranked.columns
    ]
    descriptor_columns = [
        column_name
        for column_name in ("stack_name", "variant_type", "embedding_method", "ranking_model")
        if column_name in ranked.columns
    ]
    if "stack_name" in ranked.columns and ranked["stack_name"].duplicated().any():
        aggregated = (
            ranked.groupby("stack_name", dropna=False, sort=False)[metric_columns]
            .mean()
            .reset_index()
        )
        descriptors = (
            ranked.groupby("stack_name", dropna=False, sort=False)[descriptor_columns[1:]]
            .agg("first")
            .reset_index()
            if len(descriptor_columns) > 1
            else ranked.loc[:, ["stack_name"]].drop_duplicates().reset_index(drop=True)
        )
        ranked = descriptors.merge(aggregated, on="stack_name", how="left", validate="one_to_one")

    ml_ranked = ranked[~ranked.apply(_is_baseline_row, axis=1)].copy()
    baseline_ranked = ranked[ranked.apply(_is_baseline_row, axis=1)].copy()

    collapse_mask = pd.Series([False] * len(ml_ranked), index=ml_ranked.index)
    if "f_score" in ml_ranked.columns:
        collapse_mask |= pd.to_numeric(ml_ranked["f_score"], errors="coerce").fillna(-1.0) < float(getattr(thresholds, "min_f_score", 0.0))
    if "mf" in ml_ranked.columns:
        collapse_mask |= pd.to_numeric(ml_ranked["mf"], errors="coerce").fillna(0.0) <= float(thresholds.mf_collapse_threshold)
    if "dcv" in ml_ranked.columns:
        collapse_mask |= pd.to_numeric(ml_ranked["dcv"], errors="coerce").fillna(1.0) >= float(thresholds.dcv_collapse_threshold)
    if "fraction_groups_covered" in ml_ranked.columns:
        fraction_groups = pd.to_numeric(ml_ranked["fraction_groups_covered"], errors="coerce")
        collapse_mask |= fraction_groups.notna() & (fraction_groups < float(getattr(thresholds, "min_fraction_groups_covered", 0.80)))
    fairness_candidates = ml_ranked.loc[~collapse_mask].copy()
    all_ml_invalid = bool(not ml_ranked.empty and fairness_candidates.empty)
    if fairness_candidates.empty:
        fairness_candidates = ml_ranked.copy()

    ml_ranked = _rank_ml_subset(fairness_candidates, ["f_score", "mf", "dcv", "zero_covered_groups_count", "fraction_groups_covered", "scalability_pass", "total_spread", "runtime_seconds"], [False, False, True, True, False, False, False, True])
    fairness_ranked = ml_ranked.copy()
    spread_ranked = _rank_ml_subset(ml_ranked, ["total_spread", "extra_spread", "mf", "dcv", "runtime_seconds"], [False, False, False, True, True])
    runtime_ranked = _rank_ml_subset(ml_ranked, ["runtime_seconds", "f_score", "mf", "dcv", "total_spread"], [True, False, False, True, False])

    best_overall_ml = None if ml_ranked.empty else str(ml_ranked.iloc[0]["stack_name"])

    baseline_ranked = _rank_ml_subset(baseline_ranked, ["f_score", "mf", "dcv", "total_spread", "runtime_seconds"], [False, False, True, False, True])

    best_fairness_ml = None if fairness_ranked.empty else str(fairness_ranked.iloc[0]["stack_name"])
    best_spread_ml = None if spread_ranked.empty else str(spread_ranked.iloc[0]["stack_name"])
    best_fast_ml = None if runtime_ranked.empty else str(runtime_ranked.iloc[0]["stack_name"])
    best_baseline = None if baseline_ranked.empty else str(baseline_ranked.iloc[0]["stack_name"])
    scalable_ranked = fairness_ranked
    if "scalability_pass" in fairness_ranked.columns:
        scalable_ranked = fairness_ranked[fairness_ranked["scalability_pass"].fillna(True).astype(bool)]
        if scalable_ranked.empty:
            scalable_ranked = fairness_ranked
    best_scalable_ml = None if scalable_ranked.empty else str(scalable_ranked.iloc[0]["stack_name"])

    practical_default = best_overall_ml
    if not ml_ranked.empty and len(ml_ranked) > 1:
        fastest = runtime_ranked.iloc[0]
        best = ml_ranked.iloc[0]
        if (
            pd.notna(fastest.get("runtime_seconds"))
            and pd.notna(best.get("runtime_seconds"))
            and pd.notna(fastest.get("f_score"))
            and pd.notna(best.get("f_score"))
            and float(fastest["runtime_seconds"]) <= 0.8 * float(best["runtime_seconds"])
            and abs(float(best["f_score"]) - float(fastest["f_score"])) <= float(thresholds.close_threshold)
        ):
            practical_default = str(fastest["stack_name"])

    close_text = None
    if len(ml_ranked) >= 2 and pd.notna(ml_ranked.iloc[0].get("f_score")) and pd.notna(ml_ranked.iloc[1].get("f_score")):
        gap = float(ml_ranked.iloc[0]["f_score"]) - float(ml_ranked.iloc[1]["f_score"])
        if gap <= float(thresholds.close_threshold):
            close_text = f"Close result; top two ML F-scores differ by {gap:.4f}."

    underperformers = sorted(
        {
            str(row["stack_name"])
            for _, row in ml_ranked.iterrows()
            if (
                (pd.notna(row.get("mf")) and float(row["mf"]) <= float(thresholds.mf_collapse_threshold))
                or (pd.notna(row.get("dcv")) and float(row["dcv"]) >= float(thresholds.dcv_collapse_threshold))
                or (pd.notna(row.get("f_score")) and float(row["f_score"]) < 0.0)
            )
        }
    )
    skipped_rows = raw_frame[raw_frame["status"].astype(str).str.lower() != "ok"]
    skipped_summaries = [
        f"{row['stack_name']} ({row.get('skip_reason', 'no reason provided')})"
        for _, row in skipped_rows.iterrows()
    ]

    lines = [
        f"Best current overall ML stack: {best_overall_ml or 'n/a'}",
        f"Best fairness-first ML stack: {best_fairness_ml or 'n/a'}",
        f"Best spread-first ML stack: {best_spread_ml or 'n/a'}",
        f"Best fast ML stack: {best_fast_ml or 'n/a'}",
        f"Best interpretable baseline: {best_baseline or 'n/a'}",
        f"Strongest practical default: {practical_default or 'n/a'}",
    ]
    if close_text is not None:
        lines.append(close_text)
    if best_spread_ml is not None and best_overall_ml is not None and best_spread_ml != best_overall_ml:
        lines.append(
            f"{best_spread_ml} is the best spread-oriented ML stack, but not the best fairness-adjusted ML stack."
        )
    if underperformers:
        lines.append("Fairness collapse warning: " + ", ".join(underperformers))
    if all_ml_invalid:
        lines.append("No ML stack passed fairness-first collapse thresholds; professor-priority recommendation is n/a.")
    if skipped_summaries:
        lines.append("Skipped stacks: " + "; ".join(skipped_summaries))

    recommendations = {
        "best_current_overall_ml_stack": best_overall_ml,
        "best_fairness_first_ml_stack": best_fairness_ml,
        "best_spread_first_ml_stack": best_spread_ml,
        "best_fast_ml_stack": best_fast_ml,
        "best_interpretable_baseline": best_baseline,
        "strongest_practical_default": practical_default,
        "best_fairness_quality_method": None if all_ml_invalid else best_fairness_ml,
        "best_scalable_method": None if all_ml_invalid else best_scalable_ml,
        "best_spread_method": best_spread_ml,
        "best_runtime_method": best_fast_ml,
        "final_professor_priority_recommendation": None if all_ml_invalid else best_fairness_ml,
        "stacks_needing_more_work": ", ".join(underperformers) if underperformers else None,
    }
    return lines, recommendations


def _benchmark_run_output_dir(base_output_dir: Path, *, seed: int, budget: int, protected_attribute: str) -> Path:
    safe_attribute = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in protected_attribute)
    return base_output_dir / "runs" / f"seed_{seed}" / f"budget_{budget}" / safe_attribute


def _resolve_repeat_values(single_value: object, repeated_values: Sequence[object] | None) -> list[object]:
    if repeated_values:
        return list(repeated_values)
    return [single_value]


def run_ml_fim_benchmark(
    *,
    dataset_config: object,
    specs: Sequence[FIMPermutationSpec],
    protected_attributes: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    base_run_config: FIMPermutationRunConfig,
    output_dir: Path,
    report_name: str,
    insight_thresholds: InsightThresholds,
    ranking_policy: str = "fim_default",
    save_json: bool = False,
) -> MLFIMBenchmarkResult:
    raw_frames: list[pd.DataFrame] = []
    for protected_attribute in protected_attributes:
        for budget in budgets:
            for seed in seeds:
                run_output_dir = _benchmark_run_output_dir(
                    output_dir,
                    seed=int(seed),
                    budget=int(budget),
                    protected_attribute=str(protected_attribute),
                )
                run_config = replace(
                    base_run_config,
                    protected_attribute=str(protected_attribute),
                    budget=int(budget),
                    random_seed=int(seed),
                    output_dir=run_output_dir,
                )
                benchmark_result = run_fim_permutation_benchmark_from_config(
                    dataset_config=dataset_config,
                    config=run_config,
                    permutations=specs,
                )
                frame = benchmark_result.summary_frame.copy()
                frame["budget"] = int(budget)
                frame["random_seed"] = int(seed)
                raw_frames.append(frame)

    raw_comparison_frame = pd.concat(raw_frames, ignore_index=True)
    evaluation_result = evaluate_result_frame(
        raw_comparison_frame,
        rank_by=ranking_policy,
        group_by=["dataset", "protected_attribute", "budget"],
        thresholds=insight_thresholds,
    )
    base_report = build_evaluation_report(
        evaluation_result,
        input_paths=["ml_fim_benchmark"],
        rank_by=ranking_policy,
        report_name=report_name,
    ).rstrip()
    insight_lines, ml_recommendations = build_ml_benchmark_insights(
        evaluation_result,
        raw_comparison_frame,
        thresholds=insight_thresholds,
    )
    report_text = (
        base_report
        + "\n\nML Benchmark Insights\n"
        + "\n".join(f"- {line}" for line in insight_lines)
        + "\n\nML Benchmark Recommendation\n"
        + "\n".join(f"- {key}={value or 'n/a'}" for key, value in ml_recommendations.items())
        + "\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = output_dir / f"{report_name}_comparison.csv"
    normalized_csv_path = output_dir / f"{report_name}_normalized.csv"
    ranked_csv_path = output_dir / f"{report_name}_ranked.csv"
    report_path = output_dir / f"{report_name}_report.txt"
    raw_comparison_frame.to_csv(comparison_csv_path, index=False)
    evaluation_result.normalized_frame.to_csv(normalized_csv_path, index=False)
    evaluation_result.ranked_frame.to_csv(ranked_csv_path, index=False)
    report_path.write_text(report_text, encoding="utf-8")

    json_path = None
    if save_json:
        json_path = output_dir / f"{report_name}_summary.json"
        payload = build_json_summary(evaluation_result)
        payload["ml_benchmark_insights"] = insight_lines
        payload["ml_benchmark_recommendation"] = ml_recommendations
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return MLFIMBenchmarkResult(
        raw_comparison_frame=raw_comparison_frame,
        evaluation_result=evaluation_result,
        report_text=report_text,
        comparison_csv_path=comparison_csv_path,
        normalized_csv_path=normalized_csv_path,
        ranked_csv_path=ranked_csv_path,
        report_path=report_path,
        json_path=json_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="graph_spa_500_0",
        help="Built-in dataset name, supported custom dataset path, or dataset stem under networks/.",
    )
    parser.add_argument("--graph-path", default=None, help="Path to an external graph file.")
    parser.add_argument("--attributes-path", "--attribute-path", dest="attributes_path", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "pickle", "pkl", "txt", "csv"],
        default="auto",
        help="External graph format.",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional JSON dataset config file.")
    parser.add_argument(
        "--directed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat external edge-list datasets as directed.",
    )
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--node-id-col", default=None)
    parser.add_argument("--protected-attribute", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--ml-stacks",
        nargs="+",
        default=["strong_ml"],
        help="Named stack set to benchmark. Use 'strong_ml', 'all_available', or explicit stack names.",
    )
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        default=None,
        choices=list(available_embedding_methods()),
        help="Optional embedding-method filter or generator input.",
    )
    parser.add_argument(
        "--ranking-models",
        nargs="+",
        default=None,
        choices=list(_ranking_models_for_cli()),
        help="Optional ranking-model filter or generator input.",
    )
    parser.add_argument("--community-method", default=None, help="Optional stack filter or generator override.")
    parser.add_argument("--clustering-method", default=None, help="Optional stack filter or generator override.")
    parser.add_argument(
        "--spread-estimator-search",
        default=None,
        choices=sorted(_VALID_SPREAD_SEARCH),
        help="Optional stack filter or generator override.",
    )
    parser.add_argument(
        "--spread-estimator-final",
        default="monte_carlo",
        choices=sorted(_VALID_FINAL_ESTIMATORS),
        help="Final estimator for all successful benchmark stacks; must remain monte_carlo.",
    )
    parser.add_argument(
        "--optimizer-mode",
        default=None,
        choices=sorted(_VALID_OPTIMIZER_MODES),
        help="Optional stack filter or generator override.",
    )
    parser.add_argument("--propagation-prob", type=float, default=0.01)
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=100)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.2)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ris-num-rr-sets", type=int, default=128)
    parser.add_argument("--swap-candidate-pool-size", type=int, default=24)
    parser.add_argument("--local-search-steps", type=int, default=1)
    parser.add_argument("--ranking-policy", choices=["fim_default", "fairness_first", "fairness_first_priority", "spread_first", "runtime_first"], default="fairness_first_priority")
    parser.add_argument("--min-f-score", type=float, default=0.0)
    parser.add_argument("--min-mf", type=float, default=0.0001)
    parser.add_argument("--max-dcv", type=float, default=0.25)
    parser.add_argument("--min-fraction-groups-covered", type=float, default=0.80)
    parser.add_argument("--fairness-close-threshold", type=float, default=0.003)
    parser.add_argument("--runtime-tiebreak-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ml-score-weight", type=float, default=None)
    parser.add_argument("--ris-score-weight", type=float, default=None)
    parser.add_argument("--fair-ris-score-weight", type=float, default=None)
    parser.add_argument("--fairness-bonus-weight", type=float, default=None)
    parser.add_argument("--weak-group-bonus-weight", type=float, default=None)
    parser.add_argument("--community-diversity-weight", type=float, default=None)
    parser.add_argument("--protected-group-coverage-weight", type=float, default=None)
    parser.add_argument("--fscore-weight", type=float, default=3.0)
    parser.add_argument("--mf-weight", type=float, default=2.0)
    parser.add_argument("--dcv-weight", type=float, default=2.0)
    parser.add_argument("--spread-weight", type=float, default=0.5)
    parser.add_argument("--runtime-penalty-weight", type=float, default=0.05)
    parser.add_argument("--fairness-tolerance-dcv", type=float, default=0.005)
    parser.add_argument("--fairness-tolerance-fscore-drop", type=float, default=0.001)
    parser.add_argument("--use-fairness-first-swap-acceptance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scalability-mode", choices=["auto", "off", "large_graph"], default="auto")
    parser.add_argument("--max-candidate-pool-size", type=int, default=500)
    parser.add_argument("--candidate-pool-fraction", type=float, default=0.30)
    parser.add_argument("--adaptive-ris-rr-sets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--community-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ris-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-protected-features-in-ml",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow raw protected-attribute features in ML ranker inputs. Disabled by default.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--repeat-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--repeat-protected-attributes", nargs="+", default=None)
    parser.add_argument("--output-dir", default="results/ml_fim_benchmark")
    parser.add_argument("--report-name", default=DEFAULT_BENCHMARK_NAME)
    parser.add_argument("--close-threshold", type=float, default=0.002)
    parser.add_argument("--dcv-collapse-threshold", type=float, default=0.25)
    parser.add_argument("--mf-collapse-threshold", type=float, default=0.001)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert stack failures into skipped rows instead of aborting the full benchmark.",
    )
    parser.add_argument(
        "--debug-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full traceback details when a benchmark stack fails.",
    )
    parser.add_argument(
        "--raise-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-raise the first benchmark stack exception after optional debug traceback output.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the non-ML community-aware baseline in the benchmark.",
    )
    parser.add_argument(
        "--include-weak-ml-baselines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include deprioritized weak ML baselines such as LINE and DeepWalk in the default strong_ml set.",
    )
    parser.add_argument(
        "--use-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse saved embedding artifacts when available.",
    )
    parser.add_argument(
        "--use-score-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse saved node-score caches when available.",
    )
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    specs = resolve_ml_benchmark_specs(
        ml_stacks=args.ml_stacks,
        include_baseline=bool(args.include_baseline),
        embedding_methods=args.embedding_methods,
        ranking_models=args.ranking_models,
        community_method=args.community_method,
        clustering_method=args.clustering_method,
        spread_estimator_search=args.spread_estimator_search,
        spread_estimator_final=args.spread_estimator_final,
        optimizer_mode=args.optimizer_mode,
        include_weak_ml_baselines=bool(args.include_weak_ml_baselines),
    )
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir could not be resolved.")

    protected_attributes = [str(value) for value in _resolve_repeat_values(args.protected_attribute, args.repeat_protected_attributes)]
    budgets = [int(value) for value in _resolve_repeat_values(args.budget, args.repeat_budgets)]
    seeds = [int(value) for value in _resolve_repeat_values(args.random_seed, args.repeat_seeds)]
    fairness_first_weights = str(args.ranking_policy) == "fairness_first_priority"
    ml_score_weight = (0.8 if fairness_first_weights else 1.0) if args.ml_score_weight is None else float(args.ml_score_weight)
    ris_score_weight = (0.8 if fairness_first_weights else 1.0) if args.ris_score_weight is None else float(args.ris_score_weight)
    fair_ris_score_weight = (1.2 if fairness_first_weights else 0.5) if args.fair_ris_score_weight is None else float(args.fair_ris_score_weight)
    fairness_bonus_weight = (1.0 if fairness_first_weights else 0.2) if args.fairness_bonus_weight is None else float(args.fairness_bonus_weight)
    weak_group_bonus_weight = (1.0 if fairness_first_weights else 0.2) if args.weak_group_bonus_weight is None else float(args.weak_group_bonus_weight)
    community_diversity_weight = (0.4 if fairness_first_weights else 0.2) if args.community_diversity_weight is None else float(args.community_diversity_weight)
    protected_group_coverage_weight = (1.0 if fairness_first_weights else 0.0) if args.protected_group_coverage_weight is None else float(args.protected_group_coverage_weight)
    base_run_config = FIMPermutationRunConfig(
        protected_attribute=args.protected_attribute,
        budget=int(args.budget),
        propagation_probability=float(args.propagation_prob),
        mc_runs_search=int(args.mc_runs_search),
        mc_runs_eval=int(args.mc_runs_eval),
        lambda_weight=float(args.lambda_weight),
        random_seed=int(args.random_seed),
        output_dir=output_dir,
        continue_on_error=bool(args.continue_on_error),
        debug_errors=bool(args.debug_errors),
        raise_errors=bool(args.raise_errors),
        swap_candidate_pool_size=int(args.swap_candidate_pool_size),
        local_search_steps=int(args.local_search_steps),
        population_size=int(args.population_size),
        generations=int(args.generations),
        gnn_epochs=int(args.gnn_epochs),
        gnn_hidden_dim=int(args.gnn_hidden_dim),
        gnn_num_layers=int(args.gnn_num_layers),
        gnn_dropout=float(args.gnn_dropout),
        gnn_learning_rate=float(args.gnn_learning_rate),
        gnn_weight_decay=float(args.gnn_weight_decay),
        ris_num_rr_sets=int(args.ris_num_rr_sets),
        embedding_dim=int(args.embedding_dim),
        use_embedding_cache=bool(args.use_embedding_cache and args.embedding_cache),
        use_score_cache=bool(args.use_score_cache),
        allow_protected_features_in_ml=bool(args.allow_protected_features_in_ml),
        ml_score_weight=ml_score_weight,
        ris_score_weight=ris_score_weight,
        fair_ris_score_weight=fair_ris_score_weight,
        fairness_bonus_weight=fairness_bonus_weight,
        weak_group_bonus_weight=weak_group_bonus_weight,
        diversity_bonus_weight=community_diversity_weight,
        protected_group_coverage_weight=protected_group_coverage_weight,
        ranking_policy=str(args.ranking_policy),
        min_f_score=float(args.min_f_score),
        min_mf=float(args.min_mf),
        max_dcv=float(args.max_dcv),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        fairness_close_threshold=float(args.fairness_close_threshold),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        fscore_weight=float(args.fscore_weight),
        mf_weight=float(args.mf_weight),
        dcv_weight=float(args.dcv_weight),
        spread_weight=float(args.spread_weight),
        runtime_penalty_weight=float(args.runtime_penalty_weight),
        fairness_tolerance_dcv=float(args.fairness_tolerance_dcv),
        fairness_tolerance_fscore_drop=float(args.fairness_tolerance_fscore_drop),
        use_fairness_first_swap_acceptance=bool(args.use_fairness_first_swap_acceptance),
        scalability_mode=str(args.scalability_mode),
        max_candidate_pool_size=int(args.max_candidate_pool_size),
        candidate_pool_fraction=float(args.candidate_pool_fraction),
        adaptive_ris_rr_sets=bool(args.adaptive_ris_rr_sets),
        community_cache=bool(args.community_cache),
        ris_cache=bool(args.ris_cache),
    )
    insight_thresholds = InsightThresholds(
        close_threshold=float(args.fairness_close_threshold if args.fairness_close_threshold is not None else args.close_threshold),
        dcv_collapse_threshold=float(args.max_dcv),
        mf_collapse_threshold=float(args.min_mf),
        min_f_score=float(args.min_f_score),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
    )
    result = run_ml_fim_benchmark(
        dataset_config=dataset_config,
        specs=specs,
        protected_attributes=protected_attributes,
        budgets=budgets,
        seeds=seeds,
        base_run_config=base_run_config,
        output_dir=output_dir,
        report_name=str(args.report_name),
        insight_thresholds=insight_thresholds,
        ranking_policy=str(args.ranking_policy),
        save_json=bool(args.save_json),
    )
    print(result.report_text.rstrip())
    if result.comparison_csv_path is not None:
        print("")
        print(f"Saved comparison CSV: {result.comparison_csv_path}")
        print(f"Saved normalized CSV: {result.normalized_csv_path}")
        print(f"Saved ranked CSV: {result.ranked_csv_path}")
        print(f"Saved report: {result.report_path}")
        if result.json_path is not None:
            print(f"Saved JSON summary: {result.json_path}")


if __name__ == "__main__":
    main()
