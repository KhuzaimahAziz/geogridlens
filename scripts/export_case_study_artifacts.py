"""Create documented consolidated Parquet outputs from per-scenario results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root = PROJECT_ROOT / "outputs" / "case_study" / "genco_results"
    manifest = pd.read_csv(root / "scenario_manifest.csv")
    summary = pd.read_csv(root / "scenario_summary.csv")
    buses: list[pd.DataFrame] = []
    branches: list[pd.DataFrame] = []
    for scenario_id in manifest["scenario_id"]:
        folder = root / str(scenario_id)
        buses.append(pd.read_csv(folder / "bus_results.csv"))
        branches.append(pd.read_csv(folder / "branch_results.csv"))
    summary.to_parquet(root / "scenario_summary.parquet", index=False)
    pd.concat(buses, ignore_index=True).to_parquet(root / "bus_results.parquet", index=False)
    pd.concat(branches, ignore_index=True).to_parquet(root / "branch_results.parquet", index=False)
    print(f"Exported {len(summary):,} scenario, {sum(len(frame) for frame in buses):,} bus, and {sum(len(frame) for frame in branches):,} branch records.")
    ac_root = PROJECT_ROOT / "outputs" / "case_study" / "ac_validation"
    if (ac_root / "validation_summary.csv").exists():
        ac_bus, ac_branch = [], []
        for folder in sorted(path for path in ac_root.iterdir() if path.is_dir() and (path / "bus_results.csv").exists()):
            ac_bus.append(pd.read_csv(folder / "bus_results.csv"))
            ac_branch.append(pd.read_csv(folder / "branch_results.csv"))
        if ac_bus:
            pd.concat(ac_bus, ignore_index=True).to_parquet(ac_root / "bus_results.parquet", index=False)
            pd.concat(ac_branch, ignore_index=True).to_parquet(ac_root / "branch_results.parquet", index=False)
            print(f"Exported AC results for {len(ac_bus)} scenarios.")


if __name__ == "__main__":
    main()
