"""Capability registry and reporting helpers for the FIM algorithm stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import available_baseline_methods, get_baseline_method_spec
from .clustering import available_clustering_methods, get_clustering_method_spec
from .diffusion import available_diffusion_models, get_diffusion_model_spec
from .embeddings.registry import get_method_spec as get_embedding_method_spec
from .embeddings.registry import missing_dependency_reason as embedding_missing_dependency_reason
from .embeddings.registry import available_embedding_methods
from .gnn_training import gnn_dependencies_available
from .ml_training import available_ranking_models, get_ranking_model_spec


CAPABILITY_COLUMNS = [
    "category",
    "name",
    "status",
    "integration_surface",
    "interface",
    "default",
    "optional_dependencies",
    "notes",
]


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    """One normalized capability row for reporting/export."""

    category: str
    name: str
    status: str
    integration_surface: str
    interface: str
    default: bool = False
    optional_dependencies: tuple[str, ...] = ()
    notes: str = ""


def _has_dependencies(dependencies: tuple[str, ...]) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in dependencies)


def _status_for_dependencies(dependencies: tuple[str, ...]) -> str:
    return "available" if _has_dependencies(dependencies) else "optional_dependency_missing"


def _diffusion_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_diffusion_models():
        spec = get_diffusion_model_spec(method_name)
        entries.append(
            CapabilityEntry(
                category="diffusion_model",
                name=spec.name,
                status="available",
                integration_surface="shared_evaluation",
                interface="simulate_diffusion(dataset, protected_group_report, seed_set, ...)",
                default=spec.default,
                notes=spec.description,
            )
        )
    return entries


def _spread_estimator_entries() -> list[CapabilityEntry]:
    return [
        CapabilityEntry(
            category="spread_estimator",
            name="monte_carlo",
            status="available",
            integration_surface="search_and_final_evaluation",
            interface="evaluate_seed_set(..., mc_runs=...)",
            default=True,
            notes="Trusted final evaluation path used for all reported spread/fairness metrics.",
        ),
        CapabilityEntry(
            category="spread_estimator",
            name="ris_guidance",
            status="available",
            integration_surface="search_guidance_only",
            interface="generate_ris_guidance(dataset, protected_group_report, propagation_probability, config)",
            notes="Reusable RR-set coverage prior for search-time guidance; final reporting remains Monte Carlo.",
        ),
    ]


def _seed_selection_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_baseline_methods():
        spec = get_baseline_method_spec(method_name)
        entries.append(
            CapabilityEntry(
                category="seed_selection",
                name=spec.name,
                status="available",
                integration_surface="experiment_runner",
                interface="select_baseline_seed_set(dataset, method, budget, ...)",
                notes=spec.description,
            )
        )
    return entries


def _fairness_fim_entries() -> list[CapabilityEntry]:
    return [
        CapabilityEntry(
            category="fairness_fim_method",
            name="cea_fim",
            status="available",
            integration_surface="experiment_runner",
            interface="run_loaded_experiment(..., only_methods=['cea_fim'])",
            notes="CEA-style comparator implemented as a community EA without swarm guidance or local search.",
        ),
        CapabilityEntry(
            category="fairness_fim_method",
            name="hybrid_siea",
            status="available",
            integration_surface="experiment_runner",
            interface="run_loaded_experiment(..., only_methods=['hybrid_siea'])",
            default=True,
            notes="Unified hybrid swarm-intelligence plus evolutionary optimizer.",
        ),
        CapabilityEntry(
            category="fairness_fim_method",
            name="fairness_weighted_greedy",
            status="available",
            integration_surface="experiment_runner_baseline",
            interface="baseline_methods=['fairness_weighted_greedy']",
            notes="Greedy comparator maximizing the trusted F-score objective.",
        ),
        CapabilityEntry(
            category="fairness_fim_method",
            name="maximin_greedy",
            status="available",
            integration_surface="experiment_runner_baseline",
            interface="baseline_methods=['maximin_greedy']",
            notes="Greedy comparator maximizing worst-group MF with spread-aware tie-breaking.",
        ),
    ]


def _community_and_clustering_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_clustering_methods():
        spec = get_clustering_method_spec(method_name)
        dependencies = ("igraph",) if getattr(spec, "requires_igraph", False) else ()
        entries.append(
            CapabilityEntry(
                category="community_detection" if spec.category == "graph_native" else "embedding_clustering",
                name=spec.name,
                status=_status_for_dependencies(dependencies),
                integration_surface=(
                    "experiment_runner_and_clustering_benchmark"
                    if spec.category == "graph_native"
                    else "clustering_benchmark_and_embedding_evaluation"
                ),
                interface="cluster_nodes(graph, method, embeddings=None, features=None, ...)",
                optional_dependencies=dependencies,
                default=spec.name in {"leiden", "kmeans"},
                notes=f"{spec.category} method with preferred input={spec.preferred_input_mode}.",
            )
        )
    return entries


def _embedding_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_embedding_methods():
        spec = get_embedding_method_spec(method_name)
        missing_reason = embedding_missing_dependency_reason(method_name)
        entries.append(
            CapabilityEntry(
                category="graph_embedding",
                name=spec.name,
                status="available" if missing_reason is None else "optional_dependency_missing",
                integration_surface="embedding_benchmark",
                interface="run_embedding_benchmark(..., methods=[...])",
                default=spec.name in {"deepwalk", "graphsage"},
                optional_dependencies=tuple(spec.optional_dependencies),
                notes=spec.description if missing_reason is None else missing_reason,
            )
        )
    return entries


def _ranking_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for model_name in available_ranking_models():
        spec = get_ranking_model_spec(model_name)
        dependencies = tuple(spec.optional_dependencies)
        status = _status_for_dependencies(dependencies)
        if spec.backend == "gnn" and not gnn_dependencies_available():
            status = "optional_dependency_missing"
        entries.append(
            CapabilityEntry(
                category="ranking_model",
                name=spec.name,
                status=status,
                integration_surface=(
                    "experiment_runner_ml"
                    if spec.backend in {"tabular", "gnn"}
                    else "search_guidance_only"
                ),
                interface=(
                    "train_node_utility_model(feature_frame, label_frame, budget, model_type=...)"
                    if spec.backend == "tabular"
                    else (
                        "train_gnn_node_utility_model(dataset, feature_frame, label_frame, budget, model_type=...)"
                        if spec.backend == "gnn"
                        else "generate_ris_guidance(dataset, protected_group_report, propagation_probability, config)"
                    )
                ),
                optional_dependencies=dependencies,
                notes=spec.description,
            )
        )
    return entries


def _optimization_entries() -> list[CapabilityEntry]:
    return [
        CapabilityEntry(
            category="optimization_mode",
            name="greedy_local_search",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., local_search_steps>0)",
            notes="Swap-based local search refinement inside the hybrid optimizer.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="evolutionary_algorithm",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., crossover_probability>0, mutation_probability>0)",
            notes="Population-based EA core with crossover, mutation, and repair.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="swarm_intelligence",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., disable_swarm_guidance=False)",
            notes="Leader-guidance / swarm-style population update component.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="hybrid_siea",
            status="available",
            integration_surface="experiment_runner",
            interface="method='hybrid_siea'",
            default=True,
            notes="Unified SI+EA optimizer with repair and local search.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="repair_heuristics",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer._repair_seed_set(...)",
            notes="Repair stage enforces valid, diverse, fairness-aware candidate seed sets.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="swap_local_search",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., local_search_swap_trials>0)",
            notes="Configurable swap-based neighborhood search.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="full",
            status="available",
            integration_surface="experiment_runner_cli",
            interface="--optimization-mode full",
            default=True,
            notes="Quality-oriented refinement mode.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="balanced",
            status="available",
            integration_surface="experiment_runner_cli",
            interface="--optimization-mode balanced",
            notes="Balanced runtime/quality trade-off.",
        ),
        CapabilityEntry(
            category="optimization_mode",
            name="fast",
            status="available",
            integration_surface="experiment_runner_cli",
            interface="--optimization-mode fast",
            notes="Runtime-oriented refinement mode.",
        ),
    ]


def _fairness_control_entries() -> list[CapabilityEntry]:
    return [
        CapabilityEntry(
            category="bias_fairness_control",
            name="fairness_first_initialization",
            status="available",
            integration_surface="experiment_runner",
            interface="--fairness-first-init-enabled",
            notes="Weak-group-aware initialization for hybrid search.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="worst_group_mutation_boost",
            status="available",
            integration_surface="experiment_runner",
            interface="--weakest-group-mutation-weight",
            notes="Boost mutation and local search toward under-covered groups.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="fairness_repair_weighting",
            status="available",
            integration_surface="experiment_runner",
            interface="--repair-fairness-weight",
            notes="Bias repair heuristics toward weak-group support.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="urgency_weighting",
            status="available",
            integration_surface="experiment_runner",
            interface="--urgency-weight-enabled",
            notes="Increase guidance weight on currently weak groups.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="group_robust_weighting",
            status="available",
            integration_surface="embedding_node_classification",
            interface="training_mode=group_robust / anti_collapse_group_robust",
            notes="Available in the GraphSAGE fairness training stack, not yet wired into FIM ranking training.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="adversarial_debiasing",
            status="available",
            integration_surface="embedding_node_classification",
            interface="training_mode=anti_collapse_group_robust_with_mild_adversarial",
            notes="Available in the GraphSAGE fairness training stack, not yet wired into FIM ranking training.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="class_weighted_loss",
            status="not_yet_wired",
            integration_surface="planned",
            interface="-",
            notes="Not currently exposed in the FIM ranking pipeline.",
        ),
        CapabilityEntry(
            category="bias_fairness_control",
            name="focal_loss",
            status="not_yet_wired",
            integration_surface="planned",
            interface="-",
            notes="Not currently exposed in the FIM ranking pipeline.",
        ),
    ]


def capability_entries() -> list[CapabilityEntry]:
    """Return the normalized capability registry entries."""

    return [
        *_diffusion_entries(),
        *_spread_estimator_entries(),
        *_seed_selection_entries(),
        *_fairness_fim_entries(),
        *_community_and_clustering_entries(),
        *_embedding_entries(),
        *_ranking_entries(),
        *_optimization_entries(),
        *_fairness_control_entries(),
    ]


def capability_frame() -> pd.DataFrame:
    """Return the capability registry as a normalized DataFrame."""

    frame = pd.DataFrame([asdict(entry) for entry in capability_entries()])
    if frame.empty:
        return pd.DataFrame(columns=CAPABILITY_COLUMNS)
    return frame.loc[:, [column for column in CAPABILITY_COLUMNS if column in frame.columns]]


def save_capability_report(output_dir: Path | str = "results") -> tuple[Path, Path]:
    """Write the capability registry to CSV and text report files."""

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    frame = capability_frame()
    csv_path = base_dir / "fim_algorithm_capabilities.csv"
    txt_path = base_dir / "fim_algorithm_capabilities.txt"
    frame.to_csv(csv_path, index=False)

    lines = ["FIM Algorithm Capability Summary", "=" * 80]
    for category, category_frame in frame.groupby("category", sort=False):
        lines.append(category)
        for row in category_frame.itertuples(index=False):
            dependency_note = (
                f" | deps={','.join(row.optional_dependencies)}"
                if row.optional_dependencies
                else ""
            )
            default_note = " | default" if bool(row.default) else ""
            notes = f" | notes={row.notes}" if row.notes else ""
            lines.append(
                f"- {row.name} [{row.status}] | surface={row.integration_surface}{default_note}"
                f"{dependency_note}{notes}"
            )
        lines.append("")
    txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return csv_path, txt_path
