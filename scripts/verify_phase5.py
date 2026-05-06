"""Compatibility guard for the removed legacy optimizer verification script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.removed_swarm import REMOVED_SWARM_FEATURE_MESSAGE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(REMOVED_SWARM_FEATURE_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
