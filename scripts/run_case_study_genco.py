"""Run the resumable 2,160-intervention GENCO case-study screen."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.genco_screening import run_case_study_genco  # noqa: E402
from case_study.manifest import load_config, write_manifest  # noqa: E402


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "case_study.yaml"
    config = load_config(config_path)
    outputs = config["outputs"]
    manifest = write_manifest(config_path, PROJECT_ROOT / outputs["spatial_directory"], PROJECT_ROOT / outputs["scenario_manifest"])
    raw_root = PROJECT_ROOT / "data" / "raw" / "pfdelta_task1.3" / "case118"
    with (PROJECT_ROOT / "configs" / "native_pfdelta.yaml").open(encoding="utf-8") as source:
        native = yaml.safe_load(source)
    summary = run_case_study_genco(
        manifest,
        raw_directory=raw_root / native["snapshot"]["relative_directory"],
        source_scenario_id=int(native["snapshot"]["scenario_id"]),
        model_config_path=PROJECT_ROOT / "configs" / "pfdelta_task1_3_seed1_cpu.yaml",
        checkpoint_path=PROJECT_ROOT / "models" / "genco-pfdelta" / "task_1.3" / "seed1" / "last.pt",
        normalizer_stats_path=raw_root / "train" / "processed" / "data_stats_HeteroDataMVANormalizer.pt",
        output_directory=PROJECT_ROOT / outputs["genco_directory"],
    )
    print(f"Completed GENCO screening for {len(summary):,} rows.")
    print(f"Output: {PROJECT_ROOT / outputs['genco_directory']}")


if __name__ == "__main__":
    main()
