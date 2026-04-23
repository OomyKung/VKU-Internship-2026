"""Tests for the FIM algorithm capability registry/reporting surface."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from fim_hybrid.capabilities import capability_frame, save_capability_report


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test-artifacts"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class _WorkspaceScratchDir:
    def __init__(self) -> None:
        self.path = TEST_TMP_ROOT / f"scratch_{uuid4().hex}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=False)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _workspace_tempdir() -> _WorkspaceScratchDir:
    return _WorkspaceScratchDir()


class CapabilityReportTestCase(unittest.TestCase):
    """Check that the capability registry is exportable and non-empty."""

    def test_capability_frame_contains_expected_categories_and_methods(self) -> None:
        frame = capability_frame()

        self.assertFalse(frame.empty)
        self.assertTrue({"category", "name", "status", "integration_surface"}.issubset(frame.columns))
        self.assertIn("diffusion_model", set(frame["category"]))
        self.assertIn("seed_selection", set(frame["category"]))
        self.assertIn("ranking_model", set(frame["category"]))
        self.assertIn("hybrid_siea", set(frame["name"]))
        self.assertIn("lt", set(frame["name"]))
        self.assertIn("greedy", set(frame["name"]))
        self.assertIn("mlp", set(frame["name"]))

    def test_save_capability_report_writes_csv_and_text(self) -> None:
        with _workspace_tempdir() as temp_dir:
            csv_path, txt_path = save_capability_report(output_dir=Path(temp_dir))

            self.assertTrue(csv_path.exists())
            self.assertTrue(txt_path.exists())
            frame = pd.read_csv(csv_path)
            self.assertIn("category", frame.columns)
            self.assertIn("name", frame.columns)
            self.assertIn("status", frame.columns)
            report_text = txt_path.read_text(encoding="utf-8")
            self.assertIn("FIM Algorithm Capability Summary", report_text)
            self.assertIn("diffusion_model", report_text)


if __name__ == "__main__":
    unittest.main()
