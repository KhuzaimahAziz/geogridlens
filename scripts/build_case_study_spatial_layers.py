"""Build geo-embedded network, hex cells, candidates and contingencies from real geodata."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.manifest import load_config  # noqa: E402
from case_study.spatial_layers import build_case_study_layers  # noqa: E402


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "case_study.yaml"
    outputs = load_config(config_path)["outputs"]
    with (PROJECT_ROOT / "configs" / "native_pfdelta.yaml").open(encoding="utf-8") as source:
        native = yaml.safe_load(source)
    counts = build_case_study_layers(
        config_path,
        PROJECT_ROOT / outputs["spatial_directory"],
        PROJECT_ROOT / outputs["raw_geodata_directory"],
        PROJECT_ROOT / "data" / "raw" / "pfdelta_task1.3" / "case118" / native["snapshot"]["relative_directory"],
        int(native["snapshot"]["scenario_id"]),
    )
    for name, count in counts.items():
        print(f"{name:26s} {count:>6,} features")
    print(f"Output: {PROJECT_ROOT / outputs['spatial_directory']}")


if __name__ == "__main__":
    main()
