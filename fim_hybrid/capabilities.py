"""Capability registry and reporting helpers for the unified FIM stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path

import pandas as pd

from .clustering import available_clustering_methods, get_clustering_method_spec
from .diffusion import available_diffusion_models, get_diffusion_model_spec
from .embeddings.registry import available_embedding_methods, get_method_spec as get_embedding_method_spec
from .embeddings.registry import missing_dependency_reason as embedding_missing_dependency_reason
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
                integration_surface="trusted_fim_pipeline",
                interface="evaluate_seed_set(..., diffusion_model=...)",
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
            notes="Trusted final evaluator for spread, extra_spread, MF, DCV, and F-score.",
        ),
        CapabilityEntry(
            category="spread_estimator",
            name="ris_guidance",
            status="available",
            integration_surface="search_guidance_only",
            interface="prepare_ris_guidance(dataset, protected_group_report, propagation_probability, ...)",
            notes="Search-time RIS guidance only; never replaces the final Monte Carlo path.",
        ),
        CapabilityEntry(
            category="spread_estimator",
            name="fairness_aware_ris",
            status="available",
            integration_surface="search_guidance_only",
            interface="prepare_ris_guidance(..., mode='weak_group_weighted')",
            notes="Fair-RIS weighting for search-time guidance only; final reporting remains Monte Carlo.",
        ),
    ]


def _community_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_clustering_methods():
        spec = get_clustering_method_spec(method_name)
        if spec.category != "graph_native":
            continue
        dependencies = ("igraph",) if getattr(spec, "requires_igraph", False) else ()
        entries.append(
            CapabilityEntry(
                category="community_method",
                name=spec.name,
                status=_status_for_dependencies(dependencies),
                integration_surface="trusted_fim_pipeline",
                interface="detect_communities(graph, method=..., input_mode='graph')",
                default=spec.name == "leiden",
                optional_dependencies=dependencies,
                notes=f"Graph-native community partition with preferred input={spec.preferred_input_mode}.",
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
                category="embedding_method",
                name=spec.name,
                status="available" if missing_reason is None else "optional_dependency_missing",
                integration_surface="stack_pipeline",
                interface="prepare_embedding_frame(dataset, method_name=..., output_dir=...)",
                default=spec.name == "graphsage",
                optional_dependencies=tuple(spec.optional_dependencies),
                notes=spec.description if missing_reason is None else missing_reason,
            )
        )
    return entries


def _clustering_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for method_name in available_clustering_methods():
        spec = get_clustering_method_spec(method_name)
        if spec.category != "embedding_space":
            continue
        status = "available"
        notes = f"Embedding-space clustering with preferred input={spec.preferred_input_mode}."
        if spec.name == "dbscan_or_hdbscan" and importlib.util.find_spec("hdbscan") is None:
            notes += " HDBSCAN is unavailable, so DBSCAN fallback is used."
        entries.append(
            CapabilityEntry(
                category="clustering_method",
                name=spec.name,
                status=status,
                integration_surface="stack_pipeline",
                interface="prepare_optional_clustering(dataset, method_name=..., embeddings=..., features=...)",
                default=spec.name == "kmeans",
                notes=notes,
            )
        )
    return entries


def _ranking_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for model_name in available_ranking_models():
        if model_name == "ris_guidance":
            continue
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
                integration_surface="stack_pipeline",
                interface="train_ranking_model(dataset, feature_frame, label_frame, budget, model_type=...)",
                default=spec.name == "graphsage",
                optional_dependencies=dependencies,
                notes=spec.description,
            )
        )
    return entries


def _optimizer_entries() -> list[CapabilityEntry]:
    return [
        CapabilityEntry(
            category="optimizer_mode",
            name="greedy",
            status="available",
            integration_surface="trusted_fim_pipeline",
            interface="select_baseline_seed_set(..., method='greedy')",
            notes="Spread-oriented greedy search over the trusted Monte Carlo search estimator.",
        ),
        CapabilityEntry(
            category="optimizer_mode",
            name="local_search",
            status="available",
            integration_surface="trusted_fim_pipeline",
            interface="_swap_local_search(dataset, protected_group_report, initial_seed_set, ...)",
            notes="Bounded swap local search over the shared search-time evaluator.",
        ),
        CapabilityEntry(
            category="optimizer_mode",
            name="evolutionary_algorithm",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., crossover_probability>0, mutation_probability>0)",
            notes="Population-based EA component in the unified hybrid optimizer.",
        ),
        CapabilityEntry(
            category="optimizer_mode",
            name="swarm_intelligence",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer(..., disable_swarm_guidance=False)",
            notes="Leader-guided swarm component in the unified hybrid optimizer.",
        ),
        CapabilityEntry(
            category="optimizer_mode",
            name="hybrid_si_ea",
            status="available",
            integration_surface="trusted_fim_pipeline",
            interface="HybridSIEAOptimizer(...).optimize()",
            default=True,
            notes="Unified SI+EA optimizer with repair and local search.",
        ),
        CapabilityEntry(
            category="optimizer_mode",
            name="repair_heuristics",
            status="available",
            integration_surface="hybrid_optimizer",
            interface="HybridSIEAOptimizer._repair_seed_set(...)",
            notes="Repair stage that keeps candidate seed sets valid and fairness-aware.",
        ),
    ]


def _debias_entries() -> list[CapabilityEntry]:
    gnn_status = "available" if gnn_dependencies_available() else "optional_dependency_missing"
    gnn_note = "GNN ranking backends only."
    return [
        CapabilityEntry(
            category="debias_mode",
            name="none",
            status="available",
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='none')",
            default=True,
            notes="No explicit debiasing; preserves the non-debiased ML baseline mode.",
        ),
        CapabilityEntry(
            category="debias_mode",
            name="class_weighted",
            status=gnn_status,
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='class_weighted')",
            notes=f"Priority-class weighted GNN objective. {gnn_note}",
        ),
        CapabilityEntry(
            category="debias_mode",
            name="focal_loss",
            status=gnn_status,
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='focal_loss')",
            notes=f"Auxiliary focal loss on top-budget priority targets. {gnn_note}",
        ),
        CapabilityEntry(
            category="debias_mode",
            name="worst_group_boost",
            status=gnn_status,
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='worst_group_boost')",
            notes=f"Worst-group boosted GNN loss; the strong GraphSAGE stack uses this mode. {gnn_note}",
        ),
        CapabilityEntry(
            category="debias_mode",
            name="group_dro",
            status=gnn_status,
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='group_dro')",
            notes=f"Group DRO-style robust weighting for GNN training. {gnn_note}",
        ),
        CapabilityEntry(
            category="debias_mode",
            name="adversarial",
            status=gnn_status,
            integration_surface="stack_pipeline",
            interface="train_ranking_model(..., debias_mode='adversarial')",
            notes=f"Adversarial debiasing head with gradient reversal. {gnn_note}",
        ),
    ]


def capability_entries() -> list[CapabilityEntry]:
    """Return the normalized capability registry entries."""

    return [
        *_diffusion_entries(),
        *_spread_estimator_entries(),
        *_community_entries(),
        *_embedding_entries(),
        *_clustering_entries(),
        *_ranking_entries(),
        *_optimizer_entries(),
        *_debias_entries(),
    ]


def capability_frame() -> pd.DataFrame:
    """Return the capability registry as a normalized DataFrame."""

    frame = pd.DataFrame([asdict(entry) for entry in capability_entries()])
    if frame.empty:
        return pd.DataFrame(columns=CAPABILITY_COLUMNS)
    for column in CAPABILITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, CAPABILITY_COLUMNS].copy()


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
