"""Run the AC validation subset selected from full GENCO case-study results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.ac_validation import run_ac_validation_subset, select_validation_cases  # noqa: E402
from case_study.manifest import load_config  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
    outputs = config["outputs"]
    manifest = pd.read_csv(PROJECT_ROOT / outputs["scenario_manifest"])
    genco_root = PROJECT_ROOT / outputs["genco_directory"]
    selected = select_validation_cases(
        manifest,
        pd.read_csv(genco_root / "scenario_summary.csv"),
        pd.read_parquet(PROJECT_ROOT / outputs["indicators_directory"] / "candidate_final_scores.parquet"),
        pd.read_parquet(PROJECT_ROOT / outputs["indicators_directory"] / "scenario_metrics.parquet"),
    )
    with (PROJECT_ROOT / "configs" / "native_pfdelta.yaml").open(encoding="utf-8") as source:
        native = yaml.safe_load(source)
    summary = run_ac_validation_subset(
        selected,
        raw_directory=PROJECT_ROOT / "data" / "raw" / "pfdelta_task1.3" / "case118" / native["snapshot"]["relative_directory"],
        source_scenario_id=int(native["snapshot"]["scenario_id"]),
        genco_directory=genco_root,
        output_directory=PROJECT_ROOT / outputs["root"] / "ac_validation",
        validation_config_path=PROJECT_ROOT / "configs" / "validation.yaml",
    )
    print(f"Completed {len(summary)} AC-validation scenarios.")
    print(summary[["scenario_id", "validation_reason", "screening_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
