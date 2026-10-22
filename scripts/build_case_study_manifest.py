"""Write the 2,160-intervention manifest from the built spatial layers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.manifest import load_config, write_manifest  # noqa: E402


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "case_study.yaml"
    outputs = load_config(config_path)["outputs"]
    manifest = write_manifest(config_path, PROJECT_ROOT / outputs["spatial_directory"], PROJECT_ROOT / outputs["scenario_manifest"])
    interventions = int((~manifest["is_baseline"]).sum())
    print(f"Wrote {interventions:,} interventions and {int(manifest['is_baseline'].sum())} baselines to {outputs['scenario_manifest']}")


if __name__ == "__main__":
    main()
