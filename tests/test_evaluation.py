"""Unit tests for shared seed-set evaluation and F(S) scoring."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, ProtectedGroupReport, verify_protected_groups
from fim_hybrid.evaluation import compute_f_score, evaluate_seed_set


def _toy_evaluation_dataset() -> tuple[LoadedDataset, ProtectedGroupReport]:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (5, 6),
            (5, 7),
            (5, 8),
        ]
    )
    for node_id, group_name in {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": graph.nodes[node_id]["group"]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_evaluation", graph=graph, node_attributes=node_attributes)
    report = verify_protected_groups(dataset, "group")
    return dataset, report


class EvaluationTestCase(unittest.TestCase):
    """Check the shared diffusion, fairness, and F(S) path."""

    def test_compute_f_score_matches_formula(self) -> None:
        self.assertEqual(compute_f_score(mf=0.4, dcv=0.1, lambda_weight=0.75), 0.275)

    def test_evaluate_seed_set_returns_expected_metrics(self) -> None:
        dataset, report = _toy_evaluation_dataset()

        evaluation = evaluate_seed_set(
            dataset=dataset,
            protected_group_report=report,
            seed_set=(1, 5),
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=7,
            lambda_weight=0.5,
            diffusion_model="ic",
        )

        self.assertEqual(evaluation.seed_set, (1, 5))
        self.assertEqual(evaluation.total_spread_mean, 2.0)
        self.assertEqual(evaluation.fairness.mf, 0.0)
        self.assertEqual(evaluation.fairness.dcv, 0.5)
        self.assertEqual(evaluation.f_score, -0.25)

    def test_evaluate_seed_set_handles_empty_seed_set(self) -> None:
        dataset, report = _toy_evaluation_dataset()

        evaluation = evaluate_seed_set(
            dataset=dataset,
            protected_group_report=report,
            seed_set=(),
            propagation_probability=0.5,
            mc_runs=3,
            random_seed=7,
            lambda_weight=0.5,
            diffusion_model="ic",
        )

        self.assertEqual(evaluation.seed_set, ())
        self.assertEqual(evaluation.total_spread_mean, 0.0)
        self.assertEqual(evaluation.fairness.mf, 0.0)
        self.assertEqual(evaluation.fairness.dcv, 0.0)
        self.assertEqual(evaluation.f_score, 0.0)

    def test_evaluate_seed_set_supports_linear_threshold(self) -> None:
        dataset, report = _toy_evaluation_dataset()

        evaluation = evaluate_seed_set(
            dataset=dataset,
            protected_group_report=report,
            seed_set=(1,),
            propagation_probability=1.0,
            mc_runs=3,
            random_seed=7,
            lambda_weight=0.5,
            diffusion_model="lt",
        )

        self.assertEqual(evaluation.seed_set, (1,))
        self.assertGreaterEqual(evaluation.total_spread_mean, 1.0)

    def test_evaluate_seed_set_rejects_unknown_diffusion_model(self) -> None:
        dataset, report = _toy_evaluation_dataset()

        with self.assertRaisesRegex(ValueError, "Unsupported diffusion_model"):
            evaluate_seed_set(
                dataset=dataset,
                protected_group_report=report,
                seed_set=(1,),
                diffusion_model="not_a_model",
            )


if __name__ == "__main__":
    unittest.main()
