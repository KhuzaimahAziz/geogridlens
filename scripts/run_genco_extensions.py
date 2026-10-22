"""Run the extra GENCO experiments: seed ensemble, all-branch outages, load and capacity sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.extensions import Context, run_all_branch_outages, run_capacity_sweep, run_load_sweep, run_seed_ensemble  # noqa: E402
from case_study.manifest import load_config  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
    with (PROJECT_ROOT / "configs" / "native_pfdelta.yaml").open(encoding="utf-8") as source:
        native = yaml.safe_load(source)
    ctx = Context(PROJECT_ROOT, config, native)
    manifest = pd.read_csv(ctx.case / "scenario_manifest.csv")
    final = pd.read_parquet(ctx.indicators / "candidate_final_scores.parquet")
    balanced = final.loc[final["weight_profile"].eq("balanced")].dropna(subset=["rank"]).sort_values("rank")
    best, worst = balanced.iloc[0], balanced.iloc[-1]

    print("== all-branch outage ranking")
    summary = run_all_branch_outages(ctx)
    print(summary.sort_values("outage_max_loading_increase_elsewhere", ascending=False).head(8).round(3).to_string(index=False))

    print("== load sweep on", best.candidate_id)
    sweep = run_load_sweep(ctx, best.candidate_id, int(best.linked_bus_id), [round(x, 2) for x in np.arange(0.5, 1.51, 0.05)])
    print(sweep[["load_multiplier", "ac_converged", "voltage_mae_pu", "loading_delta_mae", "loading_delta_max_error", "genco_residual_case"]].round(4).to_string(index=False))

    print("== capacity sweep for animation:", best.candidate_id, worst.candidate_id)
    normal = next(c for c in config["operating_conditions"] if c["id"] == "normal_demand")
    run_capacity_sweep(ctx, {best.candidate_id: int(best.linked_bus_id), worst.candidate_id: int(worst.linked_bus_id)}, [float(c) for c in range(0, 201, 10)], normal, ctx.case / "capacity_sweep")

    print("== seed ensemble")
    stability = run_seed_ensemble(ctx, manifest, [1, 2, 3])
    print(stability.sort_values("best_rank").head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
