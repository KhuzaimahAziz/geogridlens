"""Write the GENCO-behaviour diagnostics (residual, error locality, pooled agreement)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.genco_behaviour import analyse_genco_behaviour  # noqa: E402
from case_study.manifest import load_config  # noqa: E402


def main() -> None:
    outputs = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")["outputs"]
    case = PROJECT_ROOT / outputs["root"]
    tables = analyse_genco_behaviour(
        pd.read_csv(case / "scenario_manifest.csv"),
        pd.read_csv(case / "genco_results" / "scenario_summary.csv"),
        pd.read_csv(case / "ac_validation" / "validation_summary.csv"),
        case / "genco_results", case / "ac_validation",
        PROJECT_ROOT / outputs["spatial_directory"], PROJECT_ROOT / outputs["indicators_directory"],
    )
    print(tables["agreement"].round(3).to_string(index=False))
    print(tables["spearman"].round(3).to_string())
    print(tables["by_distance"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
