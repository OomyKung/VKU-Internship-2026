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
    format_compact_experiment_summary,
    get_fim_permutation_spec,
    run_fim_permutation_benchmark_from_config,
)
from fim_hybrid.priority_policy import (  # noqa: E402
    PROFESSOR_PRIORITY,
    ProfessorPriorityConfig,
    normalize_ranking_policy,
    professor_priority_warning,
    rank_frame_professor_priority,
)
from fim_hybrid.removed_swarm import ensure_active_optimizer, ensure_active_stack_name  # noqa: E402
from scripts.evaluate_fim_results import (  # noqa: E402
    InsightThresholds,
    build_evaluation_report,
    build_json_summary,
    delta_vs_baseline_frame,
    evaluate_result_frame,
)
from scripts.run_experiment import build_dataset_config  # noqa: E402


DEFAULT_BENCHMARK_NAME = "ml_fim_benchmark"
DEFAULT_STRONG_STACKS = (
    "graphsage_leiden_ris_ic_ea_memetic",
    "graphsage_community_ea_memetic",
    "node2vec_xgboost_community_ea_memetic",
    "gcn_community_ea_memetic",
)
WEAK_ML_BASELINES = (
    "deepwalk_mlp",
    "line_fast_ml",
)
DEFAULT_ALL_STACKS = DEFAULT_STRONG_STACKS + (
    "community_fair_greedy_baseline",
    "graphcl_maximin",
    "dgi_fair_ris",
    "vgae_fair_ris",
    "node2vec_xgboost",
) + WEAK_ML_BASELINES
_VALID_SPREAD_SEARCH = {"monte_carlo", "ris", "ris_guidance", "fairness_aware_ris"}
_CLI_SPREAD_SEARCH = {"auto"} | _VALID_SPREAD_SEARCH
_VALID_FINAL_ESTIMATORS = {"monte_carlo"}
_VALID_OPTIMIZER_MODES = {"ea_memetic"}
_ML_GUIDED_EA_MEMETIC_STACKS = {
    "graphsage_community_ea_memetic",
    "gcn_community_ea_memetic",
    "node2vec_xgboost_community_ea_memetic",
    "graphsage_leiden_ris_ic_ea_memetic",
    "node2vec_xgboost",
}


@dataclass(slots=True)
class MLFIMBenchmarkResult:
    raw_comparison_frame: pd.DataFrame
    evaluation_result: object
    report_text: str
    comparison_csv_path: Path | None = None
    normalized_csv_path: Path | None = None
    ranked_csv_path: Path | None = None
    delta_vs_baseline_csv_path: Path | None = None
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


def _normalize_spread_estimator_search(value: object) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized == "ris_guidance":
        return "ris"
    if normalized not in {"monte_carlo", "ris", "fairness_aware_ris"}:
        raise ValueError(f"Unsupported spread_estimator_search '{value}'.")
    return normalized


def _is_graphsage_leiden_ris_ic_ea_memetic(spec: FIMPermutationSpec) -> bool:
    if spec.name in _ML_GUIDED_EA_MEMETIC_STACKS:
        return True
    return (
        spec.runner_kind == "ranked_hybrid"
        and spec.optimizer_mode == "ea_memetic"
        and spec.embedding_method not in {"", "none"}
    )


def _with_appended_note(spec: FIMPermutationSpec, note: str) -> str:
    existing = str(spec.notes or "").strip()
    return "; ".join(part for part in [existing, note] if part)


def _configure_spec_ris(
    spec: FIMPermutationSpec,
    *,
    spread_estimator_search: object,
    use_ris: bool = False,
    use_fair_ris: bool = False,
) -> FIMPermutationSpec:
    search = _normalize_spread_estimator_search(spread_estimator_search)
    fair_requested = bool(use_fair_ris) or search == "fairness_aware_ris"
    ris_requested = bool(use_ris) or search == "ris"

    if fair_requested:
        return replace(
            spec,
            spread_estimator_search="fairness_aware_ris",
            use_ris_guidance=True,
            use_fair_ris=True,
            notes=_with_appended_note(spec, "fair_ris_explicit=true"),
        )
    if ris_requested:
        return replace(
            spec,
            spread_estimator_search="ris",
            use_ris_guidance=True,
            use_fair_ris=False,
            fair_ris_weight=0.0,
            notes=_with_appended_note(spec, "ris_explicit=true"),
        )
    if search == "monte_carlo":
        return replace(
            spec,
            spread_estimator_search="monte_carlo",
            use_ris_guidance=False,
            use_fair_ris=False,
            fair_ris_weight=0.0,
            notes=_with_appended_note(spec, "ris_disabled_explicit=true"),
        )
    if _is_graphsage_leiden_ris_ic_ea_memetic(spec):
        return replace(
            spec,
            spread_estimator_search="fairness_aware_ris",
            use_ris_guidance=True,
            use_fair_ris=True,
            notes=_with_appended_note(spec, "fair_ris_auto=true"),
        )
    if spec.spread_estimator_search == "ris_guidance":
        return replace(spec, spread_estimator_search="ris")
    return spec


def _configure_specs_ris(
    specs: Sequence[FIMPermutationSpec],
    *,
    spread_estimator_search: object,
    use_ris: bool = False,
    use_fair_ris: bool = False,
) -> list[FIMPermutationSpec]:
    return [
        _configure_spec_ris(
            spec,
            spread_estimator_search=spread_estimator_search,
            use_ris=use_ris,
            use_fair_ris=use_fair_ris,
        )
        for spec in specs
    ]


def _ea_memetic_override_spec(spec: FIMPermutationSpec, registry: dict[str, FIMPermutationSpec]) -> FIMPermutationSpec:
    if spec.variant_family == "baseline" or spec.embedding_method in {"", "none"}:
        return spec
    ea_memetic_name_by_stack = {
        "graphsage_community_ea_memetic": "graphsage_community_ea_memetic",
        "node2vec_xgboost_community_ea_memetic": "node2vec_xgboost_community_ea_memetic",
        "gcn_community_ea_memetic": "gcn_community_ea_memetic",
        "graphsage_community_memetic": "graphsage_community_ea_memetic",
        "node2vec_xgboost_community_memetic": "node2vec_xgboost_community_ea_memetic",
        "gcn_community_memetic": "gcn_community_ea_memetic",
    }
    mapped_name = ea_memetic_name_by_stack.get(spec.name)
    if mapped_name is not None and mapped_name in registry:
        return registry[mapped_name]
    return replace(
        spec,
        runner_kind="ranked_hybrid",
        optimizer_mode="ea_memetic",
        notes=_with_appended_note(spec, "optimizer_override=ea_memetic"),
    )


def _memetic_override_spec(spec: FIMPermutationSpec, registry: dict[str, FIMPermutationSpec]) -> FIMPermutationSpec:
    if spec.variant_family == "baseline" or spec.embedding_method in {"", "none"}:
        return spec
    memetic_name_by_stack = {
        "graphsage_community_ea_memetic": "graphsage_community_memetic",
        "node2vec_xgboost_community_ea_memetic": "node2vec_xgboost_community_memetic",
        "gcn_community_ea_memetic": "gcn_community_memetic",
    }
    mapped_name = memetic_name_by_stack.get(spec.name)
    if mapped_name is not None and mapped_name in registry:
        return registry[mapped_name]
    return replace(
        spec,
        runner_kind="ranked_hybrid",
        optimizer_mode="memetic",
        notes=_with_appended_note(spec, "optimizer_override=memetic"),
    )


def _effective_requested_search_estimator(
    *,
    spread_estimator_search: object,
    use_ris: bool,
    use_fair_ris: bool,
    force_ris_for_all_stacks: bool,
) -> str:
    search = _normalize_spread_estimator_search(spread_estimator_search)
    if bool(force_ris_for_all_stacks):
        if search == "monte_carlo":
            raise ValueError(
                "--force-ris-for-all-stacks conflicts with --spread-estimator-search monte_carlo. "
                "Use ris or fairness_aware_ris."
            )
        if bool(use_fair_ris) or search == "fairness_aware_ris":
            return "fairness_aware_ris"
        if bool(use_ris) or search == "ris":
            return "ris"
        return "fairness_aware_ris"
    if bool(use_fair_ris):
        return "fairness_aware_ris"
    if bool(use_ris):
        return "ris"
    return search


def _validate_ris_forced_specs(
    specs: Sequence[FIMPermutationSpec],
    *,
    force_ris_for_all_stacks: bool,
) -> None:
    if not bool(force_ris_for_all_stacks):
        return
    invalid = [
        spec.name
        for spec in specs
        if _normalize_spread_estimator_search(spec.spread_estimator_search) not in {"ris", "fairness_aware_ris"}
        or not bool(spec.use_ris_guidance)
    ]
    if invalid:
        raise ValueError(
            "--force-ris-for-all-stacks was requested, but these stacks are not configured for RIS search: "
            + ", ".join(invalid)
        )


def _named_ml_stack_registry() -> dict[str, FIMPermutationSpec]:
    registry = {
        "community_aware_fair_greedy": get_fim_permutation_spec("community_aware_fair_greedy"),
        "community_fair_greedy_baseline": get_fim_permutation_spec("community_fair_greedy_baseline"),
        "graphsage_leiden_ris_ic_ea_memetic": get_fim_permutation_spec("graphsage_leiden_ris_ic_ea_memetic"),
        "node2vec_xgboost_community_ea_memetic": get_fim_permutation_spec("node2vec_xgboost_community_ea_memetic"),
        "gcn_community_ea_memetic": get_fim_permutation_spec("gcn_community_ea_memetic"),
        "graphsage_community_ea_memetic": FIMPermutationSpec(
            name="graphsage_community_ea_memetic",
            description="Leiden communities with GraphSAGE guidance, Fair RIS, and EA+Memetic.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="graphsage",
            ranking_model="graphsage",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; ml_guided_only=true",
        ),
        "graphsage_community_memetic": FIMPermutationSpec(
            name="graphsage_community_memetic",
            description="Leiden communities with GraphSAGE guidance, Fair RIS, and the fairness-first Memetic Algorithm.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="graphsage",
            ranking_model="graphsage",
            optimizer_mode="memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; ml_guided_only=true; memetic=true",
        ),
        "node2vec_xgboost_community_ea_memetic": FIMPermutationSpec(
            name="node2vec_xgboost_community_ea_memetic",
            description="Leiden communities with Node2Vec+XGBoost guidance and EA+Memetic.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; lightweight_quality_runtime=true; ml_guided_only=true",
        ),
        "node2vec_xgboost_community_memetic": FIMPermutationSpec(
            name="node2vec_xgboost_community_memetic",
            description="Leiden communities with Node2Vec+XGBoost guidance and the fairness-first Memetic Algorithm.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; lightweight_quality_runtime=true; ml_guided_only=true; memetic=true",
        ),
        "gcn_community_ea_memetic": FIMPermutationSpec(
            name="gcn_community_ea_memetic",
            description="Leiden communities with GCN guidance, Fair RIS, and EA+Memetic.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="gcn",
            ranking_model="gcn",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="optional_gnn_comparison=true; ml_guided_only=true",
        ),
        "gcn_community_memetic": FIMPermutationSpec(
            name="gcn_community_memetic",
            description="Leiden communities with GCN guidance, Fair RIS, and the fairness-first Memetic Algorithm.",
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="gcn",
            ranking_model="gcn",
            optimizer_mode="memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="optional_gnn_comparison=true; ml_guided_only=true; memetic=true",
        ),
        "graphsage_community_ea_memetic": FIMPermutationSpec(
            name="graphsage_community_ea_memetic",
            description=(
                "Leiden communities with GraphSAGE guidance, Fair RIS, and the "
                "Evolutionary Memetic Algorithm (ea_* tuning knobs)."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="graphsage",
            ranking_model="graphsage",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; ml_guided_only=true; ea_memetic=true",
        ),
        "node2vec_xgboost_community_ea_memetic": FIMPermutationSpec(
            name="node2vec_xgboost_community_ea_memetic",
            description=(
                "Leiden communities with Node2Vec+XGBoost guidance, Fair RIS, and the "
                "Evolutionary Memetic Algorithm (ea_* tuning knobs)."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="main_method=true; lightweight_quality_runtime=true; ml_guided_only=true; ea_memetic=true",
        ),
        "gcn_community_ea_memetic": FIMPermutationSpec(
            name="gcn_community_ea_memetic",
            description=(
                "Leiden communities with GCN guidance, Fair RIS, and the "
                "Evolutionary Memetic Algorithm (ea_* tuning knobs)."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="gcn",
            ranking_model="gcn",
            optimizer_mode="ea_memetic",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
            notes="optional_gnn_comparison=true; ml_guided_only=true; ea_memetic=true",
        ),
        "graphsage_fair_ris_hybrid": _clone_named_spec(
            "leiden_graphsage_fair_ris_hybrid",
            public_name="graphsage_fair_ris_hybrid",
            runner_kind="ranked_hybrid",
        ),
        "gcn_fair_ris_hybrid": _clone_named_spec(
            "leiden_gcn_fair_ris_hybrid",
            public_name="gcn_fair_ris_hybrid",
            runner_kind="ranked_hybrid",
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
                "and a EA+Memetic optimizer."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="dgi",
            ranking_model="logistic_regression",
            optimizer_mode="ea_memetic",
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
                "and a EA+Memetic optimizer."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="vgae",
            ranking_model="logistic_regression",
            optimizer_mode="ea_memetic",
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
    active_names = (
        "community_aware_fair_greedy",
        "community_fair_greedy_baseline",
        "graphsage_leiden_ris_ic_ea_memetic",
        "graphsage_community_ea_memetic",
        "gcn_community_ea_memetic",
        "node2vec_xgboost_community_ea_memetic",
        "node2vec_xgboost",
        "graphcl_maximin",
        "dgi_fair_ris",
        "vgae_fair_ris",
        "line_fast_ml",
        "deepwalk_mlp",
    )
    return {name: registry[name] for name in active_names if name in registry}


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
        if normalized_optimizer != "ea_memetic" or normalized_search != "fairness_aware_ris":
            raise ValueError(
                f"{ranking_model} benchmark stacks require optimizer_mode='ea_memetic' and "
                "spread_estimator_search='fairness_aware_ris'."
            )
        return "ranked_hybrid", "f_score"

    if normalized_optimizer == "ea_memetic":
        return "ranked_hybrid", "f_score"

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
    normalized_search = _normalize_spread_estimator_search(spread_estimator_search)
    if normalized_search == "fairness_aware_ris":
        parts.append("fair_ris")
    elif normalized_search == "ris":
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
    normalized_search = _normalize_spread_estimator_search(spread_estimator_search)
    if normalized_search == "auto":
        normalized_search = "monte_carlo"
    if normalized_search not in _VALID_SPREAD_SEARCH:
        raise ValueError(f"Unsupported spread_estimator_search '{spread_estimator_search}'.")
    if optimizer_mode not in _VALID_OPTIMIZER_MODES:
        raise ValueError(f"Unsupported optimizer_mode '{optimizer_mode}'.")

    runner_kind, fairness_objective = _generic_runner_kind(
        embedding_method=embedding_method,
        ranking_model=ranking_model,
        optimizer_mode=optimizer_mode,
        spread_estimator_search=normalized_search,
    )
    use_ris_guidance = normalized_search in {"ris", "ris_guidance", "fairness_aware_ris"}
    use_fair_ris = normalized_search == "fairness_aware_ris"
    normalized_clustering = str(clustering_method).strip().lower()
    return FIMPermutationSpec(
        name=_generated_stack_name(
            community_method=str(community_method).strip().lower(),
            embedding_method=str(embedding_method).strip().lower(),
            ranking_model=str(ranking_model).strip().lower(),
            optimizer_mode=str(optimizer_mode).strip().lower(),
            clustering_method=normalized_clustering,
            spread_estimator_search=normalized_search,
        ),
        description="CLI-generated ML benchmark stack.",
        runner_kind=runner_kind,
        diffusion_model="ic",
        community_method=str(community_method).strip().lower(),
        spread_estimator_search=normalized_search,
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
    use_ris: bool = False,
    use_fair_ris: bool = False,
    force_ris_for_all_stacks: bool = False,
) -> list[FIMPermutationSpec]:
    registry = _named_ml_stack_registry()
    if optimizer_mode is not None:
        optimizer_mode = ensure_active_optimizer(optimizer_mode)
    requested_tokens = [str(value).strip().lower() for value in (ml_stacks or ["strong_ml"]) if str(value).strip()]
    effective_search_request = _effective_requested_search_estimator(
        spread_estimator_search=spread_estimator_search,
        use_ris=use_ris,
        use_fair_ris=use_fair_ris,
        force_ris_for_all_stacks=force_ris_for_all_stacks,
    )

    selected_names: list[str]
    explicit_stack_names: bool
    if "all_available" in requested_tokens:
        selected_names = list(DEFAULT_ALL_STACKS)
        explicit_stack_names = False
    elif "strong_ml" in requested_tokens:
        selected_names = list(DEFAULT_STRONG_STACKS)
        if include_weak_ml_baselines:
            selected_names.extend(WEAK_ML_BASELINES)
        explicit_stack_names = False
    else:
        selected_names = requested_tokens
        explicit_stack_names = True

    selected_specs: list[FIMPermutationSpec] = []
    for name in selected_names:
        name = ensure_active_stack_name(name).strip().lower()
        if name not in registry:
            supported = sorted(set(registry) | {"all_available", "strong_ml"})
            raise ValueError(f"Unknown ML benchmark stack '{name}'. Supported values: {supported}.")
        selected_specs.append(registry[name])

    selected_names_set = {spec.name for spec in selected_specs}
    if include_baseline and "community_aware_fair_greedy" not in selected_names_set:
        selected_specs.insert(0, registry["community_aware_fair_greedy"])

    optimizer_filter = optimizer_mode
    if optimizer_mode is not None and str(optimizer_mode).strip().lower() == "ea_memetic":
        selected_specs = [_ea_memetic_override_spec(spec, registry) for spec in selected_specs]
        optimizer_filter = None

    search_filter = _normalize_spread_estimator_search(effective_search_request)
    final_filter = str(spread_estimator_final).strip().lower()
    # When explicit stack names are provided, --clustering-method is a runtime config
    # override for FIMPermutationRunConfig, not a spec-level filter. Preset stacks all
    # have clustering_method='none' in their spec definition; the actual clustering is
    # controlled at run time via the RunConfig. Only filter by clustering_method when
    # auto-selecting from a group (strong_ml / all_available) to allow registry-defined
    # clustering variants to be selected.
    clustering_filter = None if explicit_stack_names else clustering_method
    filtered_specs = _apply_stack_filters(
        selected_specs,
        embedding_methods=embedding_methods,
        ranking_models=ranking_models,
        community_method=community_method,
        clustering_method=clustering_filter,
        spread_estimator_search=None,
        spread_estimator_final=final_filter,
        optimizer_mode=optimizer_filter,
    )

    if filtered_specs:
        configured_specs = _configure_specs_ris(
            filtered_specs,
            spread_estimator_search=search_filter,
            use_ris=use_ris,
            use_fair_ris=use_fair_ris,
        )
        _validate_ris_forced_specs(configured_specs, force_ris_for_all_stacks=force_ris_for_all_stacks)
        return configured_specs

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
                    spread_estimator_search=(
                        ("fairness_aware_ris" if ranking_model in {"graphsage", "gcn"} else "monte_carlo")
                        if search_filter == "auto"
                        else search_filter
                    ),
                    spread_estimator_final=final_filter,
                    optimizer_mode=(optimizer_mode or "ea_memetic"),
                )
            )
    if include_baseline:
        generated_specs.insert(0, registry["community_aware_fair_greedy"])
    configured_specs = _configure_specs_ris(
        generated_specs,
        spread_estimator_search=search_filter,
        use_ris=use_ris,
        use_fair_ris=use_fair_ris,
    )
    _validate_ris_forced_specs(configured_specs, force_ris_for_all_stacks=force_ris_for_all_stacks)
    return configured_specs


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
    policy_config = ProfessorPriorityConfig(
        fairness_close_threshold=float(thresholds.close_threshold),
        min_f_score=float(thresholds.min_f_score),
        min_mf=float(thresholds.mf_collapse_threshold),
        max_dcv=float(thresholds.dcv_collapse_threshold),
        min_fraction_groups_covered=float(thresholds.min_fraction_groups_covered),
        scalability_required=bool(getattr(thresholds, "scalability_required", True)),
        runtime_tiebreak_only=bool(getattr(thresholds, "runtime_tiebreak_only", True)),
        warn_only_fairness_gates=bool(getattr(thresholds, "warn_only_fairness_gates", False)),
    )
    if ranked.empty:
        return ["No successful benchmark rows were available for ML insight generation."], {
            "best_overall_method": None,
            "best_ml_only_method": None,
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
        for column_name in (
            "f_score",
            "mf",
            "dcv",
            "total_spread",
            "extra_spread",
            "runtime_seconds",
            "zero_covered_groups_count",
            "fraction_groups_covered",
            "scalability_pass",
        )
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
    if "zero_covered_groups_count" in ml_ranked.columns:
        collapse_mask |= pd.to_numeric(ml_ranked["zero_covered_groups_count"], errors="coerce").fillna(0.0) > 0.0
    fairness_candidates = ml_ranked.loc[~collapse_mask].copy()
    all_ml_invalid = bool(not ml_ranked.empty and fairness_candidates.empty)
    if fairness_candidates.empty:
        fairness_candidates = ml_ranked.copy()

    ml_ranked = rank_frame_professor_priority(fairness_candidates, policy_config)
    fairness_ranked = ml_ranked.copy()
    spread_ranked = _rank_ml_subset(ml_ranked, ["total_spread", "extra_spread", "mf", "dcv", "runtime_seconds"], [False, False, False, True, True])
    runtime_ranked = _rank_ml_subset(ml_ranked, ["runtime_seconds", "f_score", "mf", "dcv", "total_spread"], [True, False, False, True, False])

    overall_ranked = rank_frame_professor_priority(ranked.copy(), policy_config)
    best_overall_method = None if overall_ranked.empty else str(overall_ranked.iloc[0]["stack_name"])
    best_overall_ml = None if ml_ranked.empty else str(ml_ranked.iloc[0]["stack_name"])

    baseline_ranked = rank_frame_professor_priority(baseline_ranked, policy_config)

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
    if all_ml_invalid:
        practical_default = None
    if not ml_ranked.empty and len(ml_ranked) > 1:
        fastest = runtime_ranked.iloc[0]
        best = ml_ranked.iloc[0]
        if (
            not all_ml_invalid
            and
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
        gap = abs(float(ml_ranked.iloc[0]["f_score"]) - float(ml_ranked.iloc[1]["f_score"]))
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
        f"Best overall method: {best_overall_method or 'n/a'}",
        f"Best ML-only method: {best_overall_ml or 'n/a'}",
        f"Best current overall ML stack: {best_overall_ml or 'n/a'}",
        f"Best fairness-first ML stack: {best_fairness_ml or 'n/a'}",
        f"Best spread-first ML stack: {best_spread_ml or 'n/a'}",
        f"Best fast ML stack: {best_fast_ml or 'n/a'}",
        f"Best interpretable baseline: {best_baseline or 'n/a'}",
        (
            "Strongest practical default: n/a (no ML stack passed professor-priority fairness gates)"
            if all_ml_invalid
            else f"Strongest practical default: {practical_default or 'n/a'}"
        ),
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
        warning = professor_priority_warning(ml_ranked, policy_config)
        lines.append(warning or "No ML stack passed professor-priority fairness gates; showing the least-bad method with warning.")
        if best_fairness_ml is not None:
            lines.append(f"Least-bad ML method under professor priority: {best_fairness_ml} (warning: fairness gates failed).")
    if skipped_summaries:
        lines.append("Skipped stacks: " + "; ".join(skipped_summaries))
    if best_overall_method is not None and best_overall_ml is not None and best_overall_method != best_overall_ml:
        lines.append(
            "Overall recommendation differs from ML-only recommendation because baseline rows are included."
        )

    recommendations = {
        "best_overall_method": best_overall_method,
        "best_ml_only_method": best_overall_ml,
        "best_current_overall_ml_stack": best_overall_ml,
        "best_fairness_first_ml_stack": best_fairness_ml,
        "best_spread_first_ml_stack": best_spread_ml,
        "best_fast_ml_stack": best_fast_ml,
        "best_interpretable_baseline": best_baseline,
        "strongest_practical_default": practical_default,
        "best_fairness_quality_method": best_fairness_ml,
        "best_scalable_method": best_scalable_ml,
        "best_spread_method": best_spread_ml,
        "best_runtime_method": best_fast_ml,
        "final_professor_priority_recommendation": best_overall_method,
        "stacks_needing_more_work": ", ".join(underperformers) if underperformers else None,
    }
    return lines, recommendations


def _format_optimizer_diagnostics_summary(raw_frame: pd.DataFrame) -> list[str]:
    if raw_frame.empty or "optimizer_mode" not in raw_frame.columns:
        return []
    rows = raw_frame[raw_frame["optimizer_mode"].astype(str).str.lower().eq("memetic")]
    if rows.empty:
        return []
    lines = ["Optimizer Diagnostics Summary"]
    for _, row in rows.iterrows():
        if str(row.get("status", "")).strip().lower() != "ok":
            lines.append(f"- {row.get('stack_name')}: skipped ({row.get('skip_reason', row.get('skipped_reason', ''))})")
            continue
        lines.append(
            "- "
            f"{row.get('stack_name')}: mode=memetic | "
            f"population={row.get('population_size')} | "
            f"generations={row.get('generations')} | "
            f"crossover={row.get('memetic_crossover_rate')} | "
            f"mutation={row.get('memetic_mutation_rate')} | "
            f"local_search_elites={row.get('memetic_local_search_top_elites')} | "
            f"local_search_attempts={row.get('memetic_local_search_attempts')} | "
            f"local_search_improvements={row.get('memetic_local_search_improvements')} | "
            f"rejected_fairness_drops={row.get('rejected_fairness_drops')} | "
            f"best_generation={row.get('best_generation')} | "
            f"final_fitness={row.get('final_fitness')}"
        )
    return lines


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
    baseline_inclusion_status: str = "unspecified",
) -> MLFIMBenchmarkResult:
    ranking_policy = normalize_ranking_policy(ranking_policy)
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
    reporting_start = pd.Timestamp.now()
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
        thresholds=insight_thresholds,
        print_delta_vs_baseline=bool(base_run_config.print_delta_vs_baseline),
        print_decision_trace=bool(base_run_config.print_decision_trace),
        print_collapse_explanations=bool(base_run_config.print_collapse_explanations),
    ).rstrip()
    insight_lines, ml_recommendations = build_ml_benchmark_insights(
        evaluation_result,
        raw_comparison_frame,
        thresholds=insight_thresholds,
    )
    optimizer_diagnostics_lines = _format_optimizer_diagnostics_summary(raw_comparison_frame)
    report_text = (
        base_report
        + f"\n\nBaseline inclusion: {baseline_inclusion_status}\n"
        + "\n\nML Benchmark Insights\n"
        + "\n".join(f"- {line}" for line in insight_lines)
        + (
            "\n\n"
            + optimizer_diagnostics_lines[0]
            + "\n"
            + "\n".join(
                f"- {line[2:]}" if line.startswith("- ") else f"- {line}"
                for line in optimizer_diagnostics_lines[1:]
            )
            if optimizer_diagnostics_lines
            else ""
        )
        + "\n\nML Benchmark Recommendation\n"
        + "\n".join(f"- {key}={value or 'n/a'}" for key, value in ml_recommendations.items())
        + "\n"
    )
    reporting_seconds = (pd.Timestamp.now() - reporting_start).total_seconds()
    raw_comparison_frame["time_reporting"] = float(reporting_seconds)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = output_dir / f"{report_name}_comparison.csv"
    normalized_csv_path = output_dir / f"{report_name}_normalized.csv"
    ranked_csv_path = output_dir / f"{report_name}_ranked.csv"
    delta_vs_baseline_csv_path = output_dir / f"{report_name}_delta_vs_baseline.csv"
    report_path = output_dir / f"{report_name}_report.txt"
    raw_comparison_frame.to_csv(comparison_csv_path, index=False)
    evaluation_result.normalized_frame.to_csv(normalized_csv_path, index=False)
    evaluation_result.ranked_frame.to_csv(ranked_csv_path, index=False)
    delta_frames: list[pd.DataFrame] = []
    for group in evaluation_result.group_results:
        delta_frame = delta_vs_baseline_frame(group.ranked_frame)
        for key, value in group.context.items():
            delta_frame[key] = value
        if not delta_frame.empty:
            delta_frames.append(delta_frame)
    if delta_frames:
        pd.concat(delta_frames, ignore_index=True).to_csv(delta_vs_baseline_csv_path, index=False)
    else:
        pd.DataFrame().to_csv(delta_vs_baseline_csv_path, index=False)
    report_path.write_text(report_text, encoding="utf-8")

    json_path = None
    if save_json:
        json_path = output_dir / f"{report_name}_summary.json"
        payload = build_json_summary(evaluation_result, thresholds=insight_thresholds)
        payload["ml_benchmark_insights"] = insight_lines
        payload["ml_benchmark_recommendation"] = ml_recommendations
        payload["baseline_inclusion"] = baseline_inclusion_status
        payload["reporting_runtime_seconds"] = float(reporting_seconds)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return MLFIMBenchmarkResult(
        raw_comparison_frame=raw_comparison_frame,
        evaluation_result=evaluation_result,
        report_text=report_text,
        comparison_csv_path=comparison_csv_path,
        normalized_csv_path=normalized_csv_path,
        ranked_csv_path=ranked_csv_path,
        delta_vs_baseline_csv_path=delta_vs_baseline_csv_path,
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
    parser.add_argument("--num-clusters", default="auto", help="Number of embedding clusters: integer or 'auto'.")
    parser.add_argument("--use-clustering-features", action=argparse.BooleanOptionalAction, default=False, help="Add cluster_id and cluster_size columns to ML feature table.")
    parser.add_argument("--use-cluster-diversity-bonus", action=argparse.BooleanOptionalAction, default=False, help="Apply cluster diversity bonus to combined candidate scores.")
    parser.add_argument("--cluster-diversity-weight", type=float, default=0.3, help="Weight for cluster diversity bonus in combined score (default 0.3).")
    parser.add_argument("--cluster-balance-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable cluster-aware initialization and mutation in EA+Memetic.")
    parser.add_argument("--cluster-repair-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable cluster rebalancing repair pass in EA+Memetic.")
    parser.add_argument(
        "--spread-estimator-search",
        default="auto",
        choices=sorted(_CLI_SPREAD_SEARCH),
        help="Search-time spread guidance: auto, monte_carlo, ris, or fairness_aware_ris. 'ris_guidance' is accepted as a legacy alias for ris.",
    )
    parser.add_argument(
        "--spread-estimator-final",
        default="monte_carlo",
        choices=sorted(_VALID_FINAL_ESTIMATORS),
        help="Final estimator for all successful benchmark stacks; must remain monte_carlo.",
    )
    parser.add_argument(
        "--optimizer-mode",
        "--optimizer",
        dest="optimizer_mode",
        default=None,
        help="Optimizer for generated ML stacks. Valid value: ea_memetic.",
    )
    parser.add_argument("--propagation-prob", type=float, default=0.01)
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=100)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--memetic-population-size", type=int, default=None)
    parser.add_argument("--memetic-random-immigrant-rate", type=float, default=0.10)
    parser.add_argument("--memetic-initialization-mode", choices=["fairness_guided", "score_guided", "random"], default="fairness_guided")
    parser.add_argument("--memetic-fscore-weight", type=float, default=4.0)
    parser.add_argument("--memetic-mf-weight", type=float, default=2.0)
    parser.add_argument("--memetic-dcv-weight", type=float, default=2.5)
    parser.add_argument("--memetic-group-coverage-weight", type=float, default=1.0)
    parser.add_argument("--memetic-community-coverage-weight", type=float, default=0.5)
    parser.add_argument("--memetic-spread-weight", type=float, default=0.4)
    parser.add_argument("--memetic-selection", choices=["tournament", "rank", "roulette"], default="tournament")
    parser.add_argument("--memetic-tournament-size", type=int, default=3)
    parser.add_argument("--memetic-crossover", choices=["uniform", "fairness_preserving"], default="fairness_preserving")
    parser.add_argument("--memetic-crossover-rate", type=float, default=0.9)
    parser.add_argument("--memetic-mutation-rate", type=float, default=0.25)
    parser.add_argument("--memetic-mutation-strength", type=int, default=None)
    parser.add_argument("--memetic-weak-group-mutation-bias", type=float, default=0.70)
    parser.add_argument("--memetic-repair-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memetic-repair-rounds", type=int, default=2)
    parser.add_argument("--memetic-local-search-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memetic-local-search-frequency", choices=["every_generation", "final_generation", "never"], default="every_generation")
    parser.add_argument("--memetic-local-search-intensity", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--memetic-local-search-top-elites", type=float, default=0.25)
    parser.add_argument("--memetic-local-search-candidate-limit", type=int, default=None)
    parser.add_argument("--memetic-fairness-tolerance-fscore-drop", type=float, default=0.001)
    parser.add_argument("--memetic-fairness-tolerance-dcv", type=float, default=0.005)
    parser.add_argument("--memetic-elitism-rate", type=float, default=0.10)
    parser.add_argument("--memetic-diversity-preservation", action=argparse.BooleanOptionalAction, default=True)
    # -- Evolutionary Memetic (ea_*) args - independent tuning knobs for ea_memetic mode --
    parser.add_argument("--ea-selection", choices=["tournament", "rank", "roulette"], default="tournament", help="Parent selection method for ea_memetic optimizer.")
    parser.add_argument("--ea-tournament-size", type=int, default=3, help="Tournament size for ea-selection=tournament.")
    parser.add_argument("--ea-crossover-rate", type=float, default=0.90, help="Crossover probability for ea_memetic.")
    parser.add_argument("--ea-crossover-mode", choices=["uniform", "fairness_preserving"], default="fairness_preserving", help="Crossover operator for ea_memetic.")
    parser.add_argument("--ea-mutation-rate", type=float, default=0.25, help="Mutation probability for ea_memetic.")
    parser.add_argument("--ea-mutation-strength", type=int, default=None, help="Seeds to swap per mutation (default: max(1, budget*0.05)).")
    parser.add_argument("--weak-group-mutation-bias", type=float, default=0.70, help="Bias towards weak-group candidates during EA mutation.")
    parser.add_argument("--memetic-elite-fraction", type=float, default=0.25, help="Fraction of offspring receiving local search in ea_memetic.")
    parser.add_argument("--random-immigrant-rate", type=float, default=0.10, help="Fraction of random immigrants injected each generation in ea_memetic.")
    parser.add_argument("--diversity-preservation", action=argparse.BooleanOptionalAction, default=True, help="Enable Jaccard-diversity filtering during survival selection in ea_memetic.")
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.2)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)
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
    parser.add_argument("--regenerate-graphsage-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-structural-fallback", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--use-ris", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-fair-ris", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-ris-for-all-stacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-ris", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ris-mode",
        choices=["standard", "weak_group_weighted", "group_balanced"],
        default="weak_group_weighted",
    )
    parser.add_argument("--ris-num-rr-sets", type=int, default=128)
    parser.add_argument("--ris-reuse-rr-sets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swap-candidate-pool-size", type=int, default=24)
    parser.add_argument("--local-search-steps", type=int, default=1)
    parser.add_argument("--ranking-policy", choices=["fim_default", "fairness_first", "fairness_first_priority", "professor_priority", "spread_first", "runtime_first"], default="professor_priority")
    parser.add_argument("--min-f-score", type=float, default=0.0)
    parser.add_argument("--min-mf", type=float, default=0.0001)
    parser.add_argument("--max-dcv", type=float, default=0.25)
    parser.add_argument("--min-fraction-groups-covered", type=float, default=0.80)
    parser.add_argument("--fairness-close-threshold", type=float, default=0.003)
    parser.add_argument("--scalability-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-tiebreak-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warn-only-fairness-gates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ml-score-weight", type=float, default=None)
    parser.add_argument("--ris-score-weight", type=float, default=None)
    parser.add_argument("--fair-ris-score-weight", type=float, default=None)
    parser.add_argument("--fairness-bonus-weight", type=float, default=None)
    parser.add_argument("--weak-group-bonus-weight", type=float, default=None)
    parser.add_argument("--community-diversity-weight", type=float, default=None)
    parser.add_argument("--protected-group-coverage-weight", type=float, default=None)
    parser.add_argument("--spread-proxy-weight", type=float, default=None)
    parser.add_argument("--graphsage-score-weight", type=float, default=0.8)
    parser.add_argument("--shortfall-gain-weight", type=float, default=3.0)
    parser.add_argument("--auto-downweight-bad-ml", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-graphsage-guidance", action="store_true")
    parser.add_argument("--disable-shortfall-gain", action="store_true")
    parser.add_argument("--disable-fair-ris-guidance", action="store_true")
    parser.add_argument("--disable-community-diversity", action="store_true")
    parser.add_argument("--fscore-weight", type=float, default=4.0)
    parser.add_argument("--mf-weight", type=float, default=2.0)
    parser.add_argument("--dcv-weight", type=float, default=2.5)
    parser.add_argument("--group-coverage-weight", type=float, default=1.0)
    parser.add_argument("--scalability-weight", type=float, default=0.5)
    parser.add_argument("--spread-weight", type=float, default=0.5)
    parser.add_argument("--runtime-penalty-weight", type=float, default=0.05)
    parser.add_argument("--runtime-weight", type=float, default=0.05)
    parser.add_argument("--fairness-tolerance-dcv", type=float, default=0.005)
    parser.add_argument("--fairness-tolerance-fscore-drop", type=float, default=0.001)
    parser.add_argument("--use-fairness-first-swap-acceptance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-professor-priority-fitness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fair-greedy-objective", choices=["default", "f_score", "maximin", "professor_priority"], default="professor_priority")
    parser.add_argument("--dcv-penalty-weight", type=float, default=2.5)
    parser.add_argument("--community-coverage-weight", type=float, default=0.5)
    parser.add_argument("--use-ris-greedy-approximation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-fairness-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imbalance-threshold-medium", type=float, default=5.0)
    parser.add_argument("--imbalance-threshold-high", type=float, default=10.0)
    parser.add_argument("--adaptive-fairness-multiplier-medium", type=float, default=1.5)
    parser.add_argument("--adaptive-fairness-multiplier-high", type=float, default=2.0)
    parser.add_argument("--large-imbalance-fairness-mode", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--large-imbalance-threshold", type=float, default=5.0)
    parser.add_argument("--large-graph-threshold", type=int, default=1000)
    parser.add_argument("--use-group-stratified-candidate-pool", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-group-candidate-floor", type=int, default=20)
    parser.add_argument("--group-candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--use-protected-group-quota-initialization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--small-group-seed-fraction", type=float, default=0.10)
    parser.add_argument("--initialization-quota-mode", choices=["proportional", "sqrt", "uniform_min"], default="sqrt")
    parser.add_argument("--score-normalization", choices=["global", "per_group", "hybrid"], default=None)
    parser.add_argument("--large-imbalance-ml-score-weight", type=float, default=0.4)
    parser.add_argument("--large-imbalance-ris-score-weight", type=float, default=0.5)
    parser.add_argument("--large-imbalance-fair-ris-score-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-weak-group-bonus-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-protected-group-coverage-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-community-diversity-weight", type=float, default=0.8)
    parser.add_argument("--large-imbalance-spread-proxy-weight", type=float, default=0.2)
    parser.add_argument("--use-group-quota-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-seeds-per-protected-group", type=int, default=1)
    parser.add_argument("--quota-mode", choices=["none", "at_least_one", "proportional", "support_aware"], default="support_aware")
    parser.add_argument("--quota-min-group-support", type=int, default=5)
    parser.add_argument("--use-fairness-first-repair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-group-repair-rounds", type=int, default=0)
    parser.add_argument("--majority-overconcentration-threshold", type=float, default=0.60)
    parser.add_argument("--use-dcv-targeting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dcv-target-weight", type=float, default=3.0)
    parser.add_argument("--parity-error-weight", type=float, default=2.0)
    parser.add_argument("--over-served-penalty-weight", type=float, default=2.0)
    parser.add_argument("--under-served-bonus-weight", type=float, default=1.5)
    parser.add_argument("--parity-tolerance", type=float, default=0.005)
    parser.add_argument("--use-over-served-group-penalty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-dcv-first-swap-acceptance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dcv-improvement-epsilon", type=float, default=0.0005)
    parser.add_argument("--mf-drop-tolerance", type=float, default=0.001)
    parser.add_argument("--fscore-drop-tolerance", type=float, default=0.001)
    parser.add_argument("--spread-safe-dcv-tolerance", type=float, default=0.002)
    parser.add_argument("--use-dcv-parity-repair", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dcv-parity-repair-rounds", type=int, default=5)
    parser.add_argument("--dcv-parity-repair-candidate-limit", type=int, default=100)
    parser.add_argument("--use-dcv-minimization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dcv-target-mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--parity-error-improvement-epsilon", type=float, default=0.0005)
    parser.add_argument("--auto-disable-constant-score-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constant-score-epsilon", type=float, default=1e-12)
    parser.add_argument("--use-ris-parity-weighted-weak-bonus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-ideal-influence-group-bonus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the ML-score-parity group bonus with an ideal-influence-proportional bonus. "
            "Nodes in groups with a larger share of total ideal influence receive a higher "
            "under_served_group_bonus, making the component non-constant and targeting "
            "under-served large groups (e.g. lancaster, palmdale). Requires ideal_influences "
            "to be computed or an --ideal-influence-mode to be selected. "
            "Default: off (backward compatible)."
        ),
    )
    parser.add_argument(
        "--use-cluster-coverage-bonus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a per-node cluster_coverage_bonus equal to the fraction of distinct protected "
            "groups reachable via the node's direct neighbors. Cross-group bridge nodes score "
            "highest. Provides a non-constant, topology-derived signal at zero MC-run cost. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--cluster-coverage-bonus-weight",
        type=float,
        default=0.3,
        help="Weight for cluster_coverage_bonus in the combined candidate score (default 0.3).",
    )
    parser.add_argument(
        "--weak-group-overshoot-cap",
        type=float,
        default=0.0,
        help=(
            "When > 0, prevent the weak-group repair loop from placing seeds into a group "
            "whose current seed count already exceeds this multiple of its proportional quota "
            "(e.g. 1.10 = cap at 110%% of proportional share). Seeds that would go to a "
            "capped group are redirected to the next-most-underserved group. "
            "Reported as 'Seeds redirected by overshoot cap' in diagnostics. "
            "Default: 0.0 (disabled, backward compatible)."
        ),
    )
    parser.add_argument(
        "--large-group-min-quota-ratio",
        type=float,
        default=0.0,
        help=(
            "When > 0, groups whose node-fraction >= large-group-size-threshold receive a "
            "minimum repair quota of ceil(budget x size_ratio x ratio) seeds. "
            "Applied on top of any other quota mode. Default: 0.0 (disabled)."
        ),
    )
    parser.add_argument(
        "--large-group-size-threshold",
        type=float,
        default=0.20,
        help="Groups with node-fraction >= this value are considered 'large' for quota and in-group RIS purposes (default 0.20).",
    )
    parser.add_argument(
        "--use-ingroup-ris-for-large-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute IC reverse-reachability within each large group's induced subgraph. "
            "Nodes that can reach many in-group peers (internal bridge nodes) receive a bonus "
            "in the candidate score frame. Rewards lancaster/palmdale internal connectivity. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--ingroup-ris-score-weight",
        type=float,
        default=1.0,
        help="Weight for the ingroup_ris_score component in the combined candidate score (default 1.0).",
    )
    parser.add_argument(
        "--ingroup-ris-num-rr-sets",
        type=int,
        default=50,
        help="Number of RR sets per large group for in-group RIS computation (default 50).",
    )
    parser.add_argument("--scalability-mode", choices=["auto", "off", "large_graph"], default="auto")
    parser.add_argument("--max-candidate-pool-size", type=int, default=500)
    parser.add_argument("--candidate-pool-fraction", type=float, default=0.30)
    parser.add_argument("--adaptive-ris-rr-sets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swap-reject-spread-gain-if-fairness-collapses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-budget-node-ratio-warning", type=float, default=0.02)
    parser.add_argument("--imbalance-ratio-warning-threshold", type=float, default=5.0)
    parser.add_argument("--warn-if-communities-exceed-budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-seeds-per-group-warning", type=int, default=5)
    parser.add_argument("--embedding-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--community-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ris-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ris-cache-dir", default=None)
    parser.add_argument("--regenerate-ris-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evaluation-mode", choices=["fast_search", "final_confirmation", "debug_mc"], default=None)
    parser.add_argument("--ris-mc-sanity-check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ris-mc-sanity-check-seeds", type=int, default=20)
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
    parser.add_argument("--close-threshold", type=float, default=0.003)
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
        "--print-raw-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the old raw one-line benchmark diagnostics for debugging.",
    )
    parser.add_argument(
        "--debug-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print detailed debug diagnostics, including raw internal diagnostics.",
    )
    parser.add_argument(
        "--print-stack-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print resolved algorithm/module composition before each FIM stack starts.",
    )
    parser.add_argument("--print-experiment-header", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-budget-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-runtime-breakdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-seed-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-group-influence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-score-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-optimizer-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-delta-vs-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-decision-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-collapse-explanations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-mode",
        choices=["compact", "verbose"],
        default="compact",
        help=(
            "Terminal output verbosity. 'compact' (default) shows a clean results table only. "
            "'verbose' shows all per-stack diagnostics, seed distributions, and fairness tables."
        ),
    )
    parser.add_argument(
        "--verbose-evaluation-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable all detailed terminal/report diagnostics. Equivalent to --output-mode verbose.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the non-ML community-aware baseline in the benchmark. Use --no-include-baseline to disable auto-inclusion.",
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
    # Shortfall DCV arguments.
    parser.add_argument(
        "--ideal-influence-mode",
        choices=["proportional_budget_internal", "proportional_budget_feasible"],
        default="proportional_budget_internal",
        help=(
            "How to compute per-group ideal influence targets. "
            "'proportional_budget_internal': IC spread from k_g = ceil(budget * |g| / |V|) "
            "seeds in the induced subgraph (default). "
            "'proportional_budget_feasible': two-pass variant that also estimates the "
            "achievable ceiling using the full budget in the induced subgraph and caps "
            "ideal_g = min(proportional, ceiling * feasible-ceiling-factor). "
            "Prevents setting targets that exceed what a sparse/disconnected group subgraph "
            "can realistically absorb."
        ),
    )
    parser.add_argument(
        "--feasible-ceiling-factor",
        type=float,
        default=0.95,
        help=(
            "Multiplier applied to the achievable ceiling in proportional_budget_feasible mode "
            "(default 0.95). Lower values set more conservative targets."
        ),
    )
    parser.add_argument(
        "--shortfall-dcv-weight",
        type=float,
        default=1.0,
        help="Weight for DCV_shortfall in F-score.",
    )
    parser.add_argument(
        "--shortfall-dcv-worsen-tolerance",
        type=float,
        default=0.001,
        help="Allow shortfall DCV to worsen by this amount during swap acceptance.",
    )
    parser.add_argument(
        "--max-shortfall-dcv",
        type=float,
        default=0.01,
        help="Fairness gate: maximum acceptable DCV_shortfall.",
    )
    parser.add_argument(
        "--min-target-coverage-ratio",
        type=float,
        default=1.0,
        help="Fairness gate: minimum fraction of groups that must meet ideal target.",
    )
    parser.add_argument(
        "--use-shortfall-repair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable post-optimization shortfall repair to help below-target groups.",
    )
    parser.add_argument("--shortfall-repair-rounds", type=int, default=3)
    parser.add_argument("--shortfall-repair-candidate-limit", type=int, default=100)
    args = parser.parse_args()
    if args.optimizer_mode is not None:
        try:
            args.optimizer_mode = ensure_active_optimizer(args.optimizer_mode)
        except ValueError as exc:
            parser.error(str(exc))
    normalized_stacks: list[str] = []
    for stack_name in args.ml_stacks:
        try:
            normalized_stacks.append(ensure_active_stack_name(stack_name))
        except ValueError as exc:
            parser.error(str(exc))
    args.ml_stacks = normalized_stacks
    return args


def main() -> None:
    args = parse_args()
    _output_mode = str(getattr(args, "output_mode", "compact")).strip().lower()
    if bool(args.debug_diagnostics):
        args.print_raw_diagnostics = True
    if bool(args.verbose_evaluation_report) or _output_mode == "verbose":
        _output_mode = "verbose"
        for flag_name in (
            "print_experiment_header",
            "print_budget_check",
            "print_stack_summary",
            "print_runtime_breakdown",
            "print_seed_diagnostics",
            "print_group_influence",
            "print_score_diagnostics",
            "print_optimizer_diagnostics",
            "print_delta_vs_baseline",
            "print_decision_trace",
            "print_collapse_explanations",
        ):
            setattr(args, flag_name, True)
    elif _output_mode == "compact":
        for flag_name in (
            "print_experiment_header",
            "print_budget_check",
            "print_stack_summary",
            "print_runtime_breakdown",
            "print_seed_diagnostics",
            "print_group_influence",
            "print_score_diagnostics",
            "print_optimizer_diagnostics",
            "print_delta_vs_baseline",
            "print_decision_trace",
            "print_collapse_explanations",
        ):
            setattr(args, flag_name, False)
    dataset_config = build_dataset_config(args)
    if args.include_baseline is None:
        include_baseline = True
        baseline_inclusion_status = "auto-included"
        if _output_mode != "compact":
            print("Baseline auto-included by default.")
    elif bool(args.include_baseline):
        include_baseline = True
        baseline_inclusion_status = "explicit"
    else:
        include_baseline = False
        baseline_inclusion_status = "disabled"
    if _output_mode != "compact":
        print(f"Baseline inclusion: {baseline_inclusion_status}.")
    specs = resolve_ml_benchmark_specs(
        ml_stacks=args.ml_stacks,
        include_baseline=include_baseline,
        embedding_methods=args.embedding_methods,
        ranking_models=args.ranking_models,
        community_method=args.community_method,
        clustering_method=args.clustering_method,
        spread_estimator_search=args.spread_estimator_search,
        spread_estimator_final=args.spread_estimator_final,
        optimizer_mode=args.optimizer_mode,
        include_weak_ml_baselines=bool(args.include_weak_ml_baselines),
        use_ris=bool(args.use_ris),
        use_fair_ris=bool(args.use_fair_ris),
        force_ris_for_all_stacks=bool(args.force_ris_for_all_stacks),
    )
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir could not be resolved.")

    protected_attributes = [str(value) for value in _resolve_repeat_values(args.protected_attribute, args.repeat_protected_attributes)]
    budgets = [int(value) for value in _resolve_repeat_values(args.budget, args.repeat_budgets)]
    seeds = [int(value) for value in _resolve_repeat_values(args.random_seed, args.repeat_seeds)]
    ranking_policy = normalize_ranking_policy(args.ranking_policy)
    fairness_first_weights = ranking_policy == PROFESSOR_PRIORITY
    ml_score_weight = None if args.ml_score_weight is None else float(args.ml_score_weight)
    ris_score_weight = None if args.ris_score_weight is None else float(args.ris_score_weight)
    fair_ris_score_weight = None if args.fair_ris_score_weight is None else float(args.fair_ris_score_weight)
    fairness_bonus_weight = (1.0 if fairness_first_weights else 0.2) if args.fairness_bonus_weight is None else float(args.fairness_bonus_weight)
    weak_group_bonus_weight = None if args.weak_group_bonus_weight is None else float(args.weak_group_bonus_weight)
    community_diversity_weight = 0.2 if args.community_diversity_weight is None else float(args.community_diversity_weight)
    protected_group_coverage_weight = 0.0 if args.protected_group_coverage_weight is None else float(args.protected_group_coverage_weight)
    spread_proxy_weight = 0.0 if args.spread_proxy_weight is None else float(args.spread_proxy_weight)
    requested_search_estimator = _effective_requested_search_estimator(
        spread_estimator_search=args.spread_estimator_search,
        use_ris=bool(args.use_ris),
        use_fair_ris=bool(args.use_fair_ris),
        force_ris_for_all_stacks=bool(args.force_ris_for_all_stacks),
    )
    evaluation_mode = str(args.evaluation_mode or (
        "fast_search" if requested_search_estimator in {"ris", "fairness_aware_ris"} else "debug_mc"
    ))
    if evaluation_mode == "fast_search" and requested_search_estimator in {"ris", "fairness_aware_ris"} and int(args.mc_runs_eval) > 300:
        print("Evaluation mode note: fast_search uses RIS/Fair RIS during search; consider --mc-runs-eval 100-300 for quick iteration.")
    if evaluation_mode == "final_confirmation" and int(args.mc_runs_eval) < 500:
        print("Evaluation mode note: final_confirmation is usually paired with --mc-runs-eval 500-1000.")
    if evaluation_mode == "debug_mc" and requested_search_estimator in {"ris", "fairness_aware_ris"}:
        print("Evaluation mode note: debug_mc requested while RIS/Fair RIS search is active; final metrics still use Monte Carlo.")
    score_normalization = (
        "per_group"
        if args.score_normalization is None and bool(args.use_dcv_targeting)
        else (str(args.score_normalization) if args.score_normalization is not None else "global")
    )
    use_dcv_parity_repair = bool(args.use_dcv_targeting) if args.use_dcv_parity_repair is None else bool(args.use_dcv_parity_repair)
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
        memetic_population_size=args.memetic_population_size,
        memetic_random_immigrant_rate=float(args.memetic_random_immigrant_rate),
        memetic_initialization_mode=str(args.memetic_initialization_mode),
        memetic_fscore_weight=float(args.memetic_fscore_weight),
        memetic_mf_weight=float(args.memetic_mf_weight),
        memetic_dcv_weight=float(args.memetic_dcv_weight),
        memetic_group_coverage_weight=float(args.memetic_group_coverage_weight),
        memetic_community_coverage_weight=float(args.memetic_community_coverage_weight),
        memetic_spread_weight=float(args.memetic_spread_weight),
        memetic_selection=str(args.memetic_selection),
        memetic_tournament_size=int(args.memetic_tournament_size),
        memetic_crossover=str(args.memetic_crossover),
        memetic_crossover_rate=float(args.memetic_crossover_rate),
        memetic_mutation_rate=float(args.memetic_mutation_rate),
        memetic_mutation_strength=args.memetic_mutation_strength,
        memetic_weak_group_mutation_bias=float(args.memetic_weak_group_mutation_bias),
        memetic_repair_enabled=bool(args.memetic_repair_enabled),
        memetic_repair_rounds=int(args.memetic_repair_rounds),
        memetic_local_search_enabled=bool(args.memetic_local_search_enabled),
        memetic_local_search_frequency=str(args.memetic_local_search_frequency),
        memetic_local_search_intensity=str(args.memetic_local_search_intensity),
        memetic_local_search_top_elites=float(args.memetic_local_search_top_elites),
        memetic_local_search_candidate_limit=args.memetic_local_search_candidate_limit,
        memetic_fairness_tolerance_fscore_drop=float(args.memetic_fairness_tolerance_fscore_drop),
        memetic_fairness_tolerance_dcv=float(args.memetic_fairness_tolerance_dcv),
        memetic_elitism_rate=float(args.memetic_elitism_rate),
        memetic_diversity_preservation=bool(args.memetic_diversity_preservation),
        gnn_epochs=int(args.gnn_epochs),
        gnn_hidden_dim=int(args.gnn_hidden_dim),
        gnn_num_layers=int(args.gnn_num_layers),
        gnn_dropout=float(args.gnn_dropout),
        gnn_learning_rate=float(args.gnn_learning_rate),
        gnn_weight_decay=float(args.gnn_weight_decay),
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
        ris_num_rr_sets=int(args.ris_num_rr_sets),
        use_ris=bool(args.use_ris),
        use_fair_ris=bool(args.use_fair_ris),
        force_ris_for_all_stacks=bool(args.force_ris_for_all_stacks),
        require_ris=bool(args.require_ris),
        ris_mode=str(args.ris_mode),
        ris_reuse_rr_sets=bool(args.ris_reuse_rr_sets),
        embedding_dim=int(args.embedding_dim),
        use_embedding_cache=bool(args.use_embedding_cache and args.embedding_cache),
        use_score_cache=bool(args.use_score_cache),
        allow_protected_features_in_ml=bool(args.allow_protected_features_in_ml),
        ml_score_weight=ml_score_weight,
        ris_score_weight=ris_score_weight,
        fair_ris_score_weight=fair_ris_score_weight,
        graphsage_score_weight=float(args.graphsage_score_weight),
        shortfall_gain_weight=float(args.shortfall_gain_weight),
        community_diversity_weight=community_diversity_weight,
        auto_downweight_bad_ml=bool(args.auto_downweight_bad_ml),
        disable_graphsage_guidance=bool(args.disable_graphsage_guidance),
        disable_shortfall_gain=bool(args.disable_shortfall_gain),
        disable_fair_ris_guidance=bool(args.disable_fair_ris_guidance),
        disable_community_diversity=bool(args.disable_community_diversity),
        fairness_bonus_weight=fairness_bonus_weight,
        weak_group_bonus_weight=weak_group_bonus_weight,
        diversity_bonus_weight=community_diversity_weight,
        protected_group_coverage_weight=protected_group_coverage_weight,
        ranking_policy=ranking_policy,
        min_f_score=float(args.min_f_score),
        min_mf=float(args.min_mf),
        max_dcv=float(args.max_dcv),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        fairness_close_threshold=float(args.fairness_close_threshold),
        scalability_required=bool(args.scalability_required),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        warn_only_fairness_gates=bool(args.warn_only_fairness_gates),
        fscore_weight=float(args.fscore_weight),
        mf_weight=float(args.mf_weight),
        dcv_weight=float(args.dcv_weight),
        group_coverage_weight=float(args.group_coverage_weight),
        scalability_weight=float(args.scalability_weight),
        spread_weight=float(args.spread_weight),
        runtime_penalty_weight=float(args.runtime_penalty_weight),
        runtime_weight=float(args.runtime_weight),
        fairness_tolerance_dcv=float(args.fairness_tolerance_dcv),
        fairness_tolerance_fscore_drop=float(args.fairness_tolerance_fscore_drop),
        use_fairness_first_swap_acceptance=bool(args.use_fairness_first_swap_acceptance),
        use_professor_priority_fitness=bool(args.use_professor_priority_fitness),
        fair_greedy_objective=str(args.fair_greedy_objective),
        dcv_penalty_weight=float(args.dcv_penalty_weight),
        community_coverage_weight=float(args.community_coverage_weight),
        use_ris_greedy_approximation=bool(args.use_ris_greedy_approximation),
        adaptive_fairness_weights=bool(args.adaptive_fairness_weights),
        imbalance_threshold_medium=float(args.imbalance_threshold_medium),
        imbalance_threshold_high=float(args.imbalance_threshold_high),
        adaptive_fairness_multiplier_medium=float(args.adaptive_fairness_multiplier_medium),
        adaptive_fairness_multiplier_high=float(args.adaptive_fairness_multiplier_high),
        large_imbalance_fairness_mode=str(args.large_imbalance_fairness_mode),
        large_imbalance_threshold=float(args.large_imbalance_threshold),
        large_graph_threshold=int(args.large_graph_threshold),
        use_group_stratified_candidate_pool=args.use_group_stratified_candidate_pool,
        min_group_candidate_floor=int(args.min_group_candidate_floor),
        group_candidate_multiplier=float(args.group_candidate_multiplier),
        use_protected_group_quota_initialization=args.use_protected_group_quota_initialization,
        small_group_seed_fraction=float(args.small_group_seed_fraction),
        initialization_quota_mode=str(args.initialization_quota_mode),
        score_normalization=score_normalization,
        large_imbalance_ml_score_weight=float(args.large_imbalance_ml_score_weight),
        large_imbalance_ris_score_weight=float(args.large_imbalance_ris_score_weight),
        large_imbalance_fair_ris_score_weight=float(args.large_imbalance_fair_ris_score_weight),
        large_imbalance_weak_group_bonus_weight=float(args.large_imbalance_weak_group_bonus_weight),
        large_imbalance_protected_group_coverage_weight=float(args.large_imbalance_protected_group_coverage_weight),
        large_imbalance_community_diversity_weight=float(args.large_imbalance_community_diversity_weight),
        large_imbalance_spread_proxy_weight=float(args.large_imbalance_spread_proxy_weight),
        use_group_quota_repair=bool(args.use_group_quota_repair),
        min_seeds_per_protected_group=int(args.min_seeds_per_protected_group),
        quota_mode=str(args.quota_mode),
        quota_min_group_support=int(args.quota_min_group_support),
        use_fairness_first_repair=bool(args.use_fairness_first_repair),
        weak_group_repair_rounds=int(args.weak_group_repair_rounds),
        majority_overconcentration_threshold=float(args.majority_overconcentration_threshold),
        use_dcv_targeting=bool(args.use_dcv_targeting),
        dcv_target_weight=float(args.dcv_target_weight),
        parity_error_weight=float(args.parity_error_weight),
        over_served_penalty_weight=float(args.over_served_penalty_weight),
        under_served_bonus_weight=float(args.under_served_bonus_weight),
        parity_tolerance=float(args.parity_tolerance),
        use_over_served_group_penalty=bool(args.use_over_served_group_penalty or args.use_dcv_targeting),
        use_dcv_first_swap_acceptance=bool(args.use_dcv_first_swap_acceptance or args.use_dcv_targeting),
        dcv_improvement_epsilon=float(args.dcv_improvement_epsilon),
        mf_drop_tolerance=float(args.mf_drop_tolerance),
        fscore_drop_tolerance=float(args.fscore_drop_tolerance),
        spread_safe_dcv_tolerance=float(args.spread_safe_dcv_tolerance),
        use_dcv_parity_repair=use_dcv_parity_repair,
        dcv_parity_repair_rounds=int(args.dcv_parity_repair_rounds),
        dcv_parity_repair_candidate_limit=int(args.dcv_parity_repair_candidate_limit),
        use_dcv_minimization=bool(args.use_dcv_minimization),
        dcv_target_mode=str(args.dcv_target_mode),
        parity_error_improvement_epsilon=float(args.parity_error_improvement_epsilon),
        auto_disable_constant_score_components=bool(args.auto_disable_constant_score_components),
        constant_score_epsilon=float(args.constant_score_epsilon),
        use_ris_parity_weighted_weak_bonus=bool(args.use_ris_parity_weighted_weak_bonus),
        use_ideal_influence_group_bonus=bool(args.use_ideal_influence_group_bonus),
        use_cluster_coverage_bonus=bool(args.use_cluster_coverage_bonus),
        cluster_coverage_bonus_weight=float(args.cluster_coverage_bonus_weight),
        weak_group_overshoot_cap=float(args.weak_group_overshoot_cap),
        large_group_min_quota_ratio=float(args.large_group_min_quota_ratio),
        large_group_size_threshold=float(args.large_group_size_threshold),
        use_ingroup_ris_for_large_groups=bool(args.use_ingroup_ris_for_large_groups),
        ingroup_ris_score_weight=float(args.ingroup_ris_score_weight),
        ingroup_ris_num_rr_sets=int(args.ingroup_ris_num_rr_sets),
        swap_reject_spread_gain_if_fairness_collapses=bool(args.swap_reject_spread_gain_if_fairness_collapses),
        min_budget_node_ratio_warning=float(args.min_budget_node_ratio_warning),
        imbalance_ratio_warning_threshold=float(args.imbalance_ratio_warning_threshold),
        warn_if_communities_exceed_budget=bool(args.warn_if_communities_exceed_budget),
        min_seeds_per_group_warning=int(args.min_seeds_per_group_warning),
        scalability_mode=str(args.scalability_mode),
        max_candidate_pool_size=int(args.max_candidate_pool_size),
        candidate_pool_fraction=float(args.candidate_pool_fraction),
        adaptive_ris_rr_sets=bool(args.adaptive_ris_rr_sets),
        community_cache=bool(args.community_cache),
        ris_cache=bool(args.ris_cache),
        ris_cache_dir=_resolve_repo_path(args.ris_cache_dir) if args.ris_cache_dir else None,
        regenerate_ris_cache=bool(args.regenerate_ris_cache),
        evaluation_mode=evaluation_mode,
        ris_mc_sanity_check=bool(args.ris_mc_sanity_check),
        ris_mc_sanity_check_seeds=int(args.ris_mc_sanity_check_seeds),
        print_experiment_header=bool(args.print_experiment_header),
        print_budget_check=bool(args.print_budget_check),
        print_stack_summary=bool(args.print_stack_summary),
        print_runtime_breakdown=bool(args.print_runtime_breakdown),
        print_seed_diagnostics=bool(args.print_seed_diagnostics),
        print_group_influence=bool(args.print_group_influence),
        print_score_diagnostics=bool(args.print_score_diagnostics),
        print_optimizer_diagnostics=bool(args.print_optimizer_diagnostics),
        print_delta_vs_baseline=bool(args.print_delta_vs_baseline),
        print_decision_trace=bool(args.print_decision_trace),
        print_collapse_explanations=bool(args.print_collapse_explanations),
        print_raw_diagnostics=bool(args.print_raw_diagnostics),
        debug_diagnostics=bool(args.debug_diagnostics),
        spread_proxy_weight=spread_proxy_weight,
        clustering_method=str(args.clustering_method or "none"),
        num_clusters=args.num_clusters,
        use_clustering_features=bool(args.use_clustering_features),
        use_cluster_diversity_bonus=bool(args.use_cluster_diversity_bonus),
        cluster_diversity_weight=float(args.cluster_diversity_weight),
        cluster_balance_enabled=bool(args.cluster_balance_enabled),
        cluster_repair_enabled=bool(args.cluster_repair_enabled),
        ea_selection=str(args.ea_selection),
        ea_tournament_size=int(args.ea_tournament_size),
        ea_crossover_rate=float(args.ea_crossover_rate),
        ea_crossover_mode=str(args.ea_crossover_mode),
        ea_mutation_rate=float(args.ea_mutation_rate),
        ea_mutation_strength=args.ea_mutation_strength,
        ea_weak_group_mutation_bias=float(args.weak_group_mutation_bias),
        ea_elite_fraction=float(args.memetic_elite_fraction),
        ea_random_immigrant_rate=float(args.random_immigrant_rate),
        ea_diversity_preservation=bool(args.diversity_preservation),
        # Shortfall DCV settings.
        ideal_influence_mode=str(args.ideal_influence_mode),
        feasible_ceiling_factor=float(args.feasible_ceiling_factor),
        shortfall_dcv_weight=float(args.shortfall_dcv_weight),
        shortfall_dcv_worsen_tolerance=float(args.shortfall_dcv_worsen_tolerance),
        max_shortfall_dcv=float(args.max_shortfall_dcv),
        min_target_coverage_ratio=float(args.min_target_coverage_ratio),
        use_shortfall_repair=bool(args.use_shortfall_repair),
        shortfall_repair_rounds=int(args.shortfall_repair_rounds),
        shortfall_repair_candidate_limit=int(args.shortfall_repair_candidate_limit),
    )
    insight_thresholds = InsightThresholds(
        close_threshold=float(args.fairness_close_threshold if args.fairness_close_threshold is not None else args.close_threshold),
        dcv_collapse_threshold=float(args.max_dcv),
        mf_collapse_threshold=float(args.min_mf),
        min_f_score=float(args.min_f_score),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        scalability_required=bool(args.scalability_required),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        warn_only_fairness_gates=bool(args.warn_only_fairness_gates),
        max_shortfall_dcv=float(args.max_shortfall_dcv),
        min_target_coverage_ratio=float(args.min_target_coverage_ratio),
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
        ranking_policy=ranking_policy,
        save_json=bool(args.save_json),
        baseline_inclusion_status=baseline_inclusion_status,
    )
    if _output_mode == "compact":
        print(format_compact_experiment_summary(result.raw_comparison_frame, base_run_config))
    else:
        print(result.report_text.rstrip())
    if result.comparison_csv_path is not None:
        print("")
        print(f"Saved comparison CSV: {result.comparison_csv_path}")
        print(f"Saved normalized CSV: {result.normalized_csv_path}")
        print(f"Saved ranked CSV: {result.ranked_csv_path}")
        print(f"Saved delta-vs-baseline CSV: {result.delta_vs_baseline_csv_path}")
        print(f"Saved report: {result.report_path}")
        if result.json_path is not None:
            print(f"Saved JSON summary: {result.json_path}")


if __name__ == "__main__":
    main()
