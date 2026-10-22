"""Aggregate completed GENCO case-study outputs into decision-map tables."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.manifest import load_config  # noqa: E402
from case_study.scoring import score_case_study  # noqa: E402
import pandas as pd


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
    outputs = config["outputs"]
    manifest = pd.read_csv(PROJECT_ROOT / outputs["scenario_manifest"])
    tables = score_case_study(
        config, manifest, PROJECT_ROOT / outputs["genco_directory"],
        PROJECT_ROOT / outputs["spatial_directory"], PROJECT_ROOT / outputs["indicators_directory"],
    )
    print(f"Created case-study map tables: {', '.join(tables)}")
    print(f"Output: {PROJECT_ROOT / outputs['indicators_directory']}")


if __name__ == "__main__":
    main()
