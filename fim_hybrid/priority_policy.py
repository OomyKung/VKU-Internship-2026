"""Shared professor-priority ranking and fairness-collapse gates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Mapping, Sequence

import pandas as pd


PROFESSOR_PRIORITY = "professor_priority"
FAIRNESS_FIRST_PRIORITY = "fairness_first_priority"
PRIORITY_POLICY_ALIASES = {PROFESSOR_PRIORITY, FAIRNESS_FIRST_PRIORITY}


@dataclass(frozen=True, slots=True)
class ProfessorPriorityConfig:
    """Thresholds and switches for the professor-priority policy."""

    fairness_close_threshold: float = 0.003
    scalability_required: bool = True
    runtime_tiebreak_only: bool = True
    min_f_score: float = 0.0
    min_mf: float = 0.0001
    max_dcv: float = 0.25
    min_fraction_groups_covered: float = 0.80
    warn_only_fairness_gates: bool = False


def normalize_ranking_policy(policy: object) -> str:
    """Normalize ranking-policy aliases while preserving non-priority policies."""

    value = str(policy or "").strip().lower()
    if value == FAIRNESS_FIRST_PRIORITY:
        return PROFESSOR_PRIORITY
    return value or "fim_default"


def is_professor_priority(policy: object) -> bool:
    """Return whether a policy token requests the professor-priority order."""

    return normalize_ranking_policy(policy) == PROFESSOR_PRIORITY


def professor_priority_config_from_object(
    source: object | None,
    base: ProfessorPriorityConfig = ProfessorPriorityConfig(),
) -> ProfessorPriorityConfig:
    """Build a policy config from any object exposing matching attributes."""

    if source is None:
        return base
    return ProfessorPriorityConfig(
        fairness_close_threshold=float(getattr(source, "fairness_close_threshold", base.fairness_close_threshold)),
        scalability_required=bool(getattr(source, "scalability_required", base.scalability_required)),
        runtime_tiebreak_only=bool(getattr(source, "runtime_tiebreak_only", base.runtime_tiebreak_only)),
        min_f_score=float(getattr(source, "min_f_score", base.min_f_score)),
        min_mf=float(getattr(source, "min_mf", base.min_mf)),
        max_dcv=float(getattr(source, "max_dcv", base.max_dcv)),
        min_fraction_groups_covered=float(
            getattr(source, "min_fraction_groups_covered", base.min_fraction_groups_covered)
        ),
        warn_only_fairness_gates=bool(getattr(source, "warn_only_fairness_gates", base.warn_only_fairness_gates)),
    )


def _first_present(row: Mapping[str, Any] | pd.Series, names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            value = row[name]
            if not pd.isna(value):
                return value
    return pd.NA


def _number(row: Mapping[str, Any] | pd.Series, names: Sequence[str], default: float) -> float:
    value = _first_present(row, names)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default if pd.isna(numeric) else numeric)


def _status_ok(row: Mapping[str, Any] | pd.Series) -> bool:
    status = str(row.get("status", "ok")).strip().lower()
    if not status:
        return True
    return status in {"ok", "success", "successful", "succeeded", "complete", "completed"}


def _scalability_ok(row: Mapping[str, Any] | pd.Series) -> bool:
    value = _first_present(row, ("scalability_pass", "scalability_status"))
    if pd.isna(value):
        return True
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(numeric) > 0.0
    text = str(value).strip().lower()
    return text not in {"false", "0", "no", "fail", "failed"}


def fairness_gate_failures(
    row: Mapping[str, Any] | pd.Series,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
) -> tuple[str, ...]:
    """Return fairness-collapse gate names failed by one result row."""

    failures: list[str] = []
    if not _status_ok(row):
        failures.append("status")
    if _number(row, ("f_score", "F-score", "fscore"), float("-inf")) < float(config.min_f_score):
        failures.append("min_f_score")
    if _number(row, ("mf", "MF"), 0.0) <= float(config.min_mf):
        failures.append("min_mf")
    if _number(row, ("dcv", "DCV"), float("inf")) >= float(config.max_dcv):
        failures.append("max_dcv")

    zero_groups = _first_present(row, ("zero_covered_groups_count", "zero_covered_groups"))
    if not pd.isna(zero_groups):
        if _number(row, ("zero_covered_groups_count", "zero_covered_groups"), 0.0) > 0.0:
            failures.append("zero_covered_groups")

    fraction = _first_present(row, ("fraction_groups_covered",))
    if not pd.isna(fraction):
        if _number(row, ("fraction_groups_covered",), 1.0) < float(config.min_fraction_groups_covered):
            failures.append("min_fraction_groups_covered")

    if bool(config.scalability_required) and not _scalability_ok(row):
        failures.append("scalability_required")

    return tuple(failures)


def fairness_gate_passes(
    row: Mapping[str, Any] | pd.Series,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
) -> bool:
    """Return whether one row passes collapse gates, respecting warn-only mode."""

    if bool(config.warn_only_fairness_gates):
        return _status_ok(row)
    return not fairness_gate_failures(row, config)


def fairness_valid_mask(
    frame: pd.DataFrame,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
) -> pd.Series:
    """Vectorized fairness-collapse validity mask for result frames."""

    if frame.empty:
        return pd.Series([], index=frame.index, dtype=bool)
    return pd.Series(
        [fairness_gate_passes(row, config) for _, row in frame.iterrows()],
        index=frame.index,
        dtype=bool,
    )


def fairness_gate_notes(
    frame: pd.DataFrame,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
) -> pd.Series:
    """Return semicolon-separated gate failures for every row."""

    if frame.empty:
        return pd.Series([], index=frame.index, dtype=object)
    return pd.Series(
        [";".join(fairness_gate_failures(row, config)) for _, row in frame.iterrows()],
        index=frame.index,
        dtype=object,
    )


def _better_number(
    left: float,
    right: float,
    *,
    higher_is_better: bool,
    threshold: float = 0.0,
) -> int:
    if pd.isna(left) and pd.isna(right):
        return 0
    if pd.isna(left):
        return 1
    if pd.isna(right):
        return -1
    delta = float(left) - float(right)
    if abs(delta) <= float(threshold):
        return 0
    if higher_is_better:
        return -1 if delta > 0.0 else 1
    return -1 if delta < 0.0 else 1


def _name(row: Mapping[str, Any] | pd.Series) -> str:
    return str(row.get("stack_name", row.get("method", "")))


def compare_professor_priority(
    left: Mapping[str, Any] | pd.Series,
    right: Mapping[str, Any] | pd.Series,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
    *,
    gate_validity_enabled: bool = True,
) -> int:
    """Comparator implementing fairness, scalability, spread, then runtime."""

    if gate_validity_enabled and not bool(config.warn_only_fairness_gates):
        left_valid = not fairness_gate_failures(left, config)
        right_valid = not fairness_gate_failures(right, config)
        if left_valid != right_valid:
            return -1 if left_valid else 1

    comparisons = (
        (_number(left, ("f_score", "F-score", "fscore"), float("-inf")), _number(right, ("f_score", "F-score", "fscore"), float("-inf")), True),
        (_number(left, ("mf", "MF"), float("-inf")), _number(right, ("mf", "MF"), float("-inf")), True),
        (_number(left, ("dcv", "DCV"), float("inf")), _number(right, ("dcv", "DCV"), float("inf")), False),
    )
    for left_value, right_value, higher_is_better in comparisons:
        result = _better_number(
            left_value,
            right_value,
            higher_is_better=higher_is_better,
            threshold=0.0,
        )
        if result != 0:
            return result

    left_scalable = _scalability_ok(left)
    right_scalable = _scalability_ok(right)
    if left_scalable != right_scalable:
        return -1 if left_scalable else 1

    for names, higher_is_better in (
        (("total_spread", "spread"), True),
        (("extra_spread", "extra"), True),
        (("runtime_seconds", "runtime"), False),
    ):
        default = float("-inf") if higher_is_better else float("inf")
        result = _better_number(
            _number(left, names, default),
            _number(right, names, default),
            higher_is_better=higher_is_better,
            threshold=0.0,
        )
        if result != 0:
            return result

    return -1 if _name(left) < _name(right) else (1 if _name(left) > _name(right) else 0)


def rank_professor_priority_frame(
    frame: pd.DataFrame,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
    *,
    gate_validity_enabled: bool = True,
) -> pd.DataFrame:
    """Return a frame ordered by the professor-priority comparator."""

    if frame.empty:
        return frame.copy()
    rows = [(index, row) for index, row in frame.iterrows()]
    ordered = sorted(
        rows,
        key=cmp_to_key(
            lambda left, right: compare_professor_priority(
                left[1],
                right[1],
                config,
                gate_validity_enabled=gate_validity_enabled,
            )
        ),
    )
    return frame.loc[[index for index, _ in ordered]].reset_index(drop=True)


def rank_frame_professor_priority(
    frame: pd.DataFrame,
    config: ProfessorPriorityConfig = ProfessorPriorityConfig(),
    *,
    gate_validity_enabled: bool = True,
) -> pd.DataFrame:
    """Compatibility alias for the shared professor-priority ranker."""

    return rank_professor_priority_frame(
        frame,
        config,
        gate_validity_enabled=gate_validity_enabled,
    )


def professor_priority_warning(frame: pd.DataFrame, config: ProfessorPriorityConfig) -> str | None:
    """Return the all-failed fairness-gate warning when applicable."""

    if frame.empty or bool(config.warn_only_fairness_gates):
        return None
    ok_rows = frame[
        frame.apply(_status_ok, axis=1)
    ] if "status" in frame.columns else frame
    if ok_rows.empty:
        return None
    valid_mask = fairness_valid_mask(ok_rows, config)
    if bool(valid_mask.any()):
        return None
    return (
        "No method passed professor-priority fairness gates; "
        "showing the least-bad method by F-score, MF, DCV, scalability, spread, extra spread, then runtime."
    )
