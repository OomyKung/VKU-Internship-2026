"""Write a normalized capability summary for the FIM algorithm stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.capabilities import capability_frame, save_capability_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results", help="Directory where the capability report will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_path, txt_path = save_capability_report(output_dir=output_dir)
    frame = capability_frame()
    print(f"Saved capability CSV to {csv_path}")
    print(f"Saved capability report to {txt_path}")
    if frame.empty:
        print("No capability entries were produced.")
        return
    print("")
    print("Capability counts by category:")
    counts = frame.groupby("category")["name"].count().sort_index()
    for category, count in counts.items():
        print(f"- {category}: {int(count)}")


if __name__ == "__main__":
    main()
