"""Diagnostics on how the released GENCO checkpoint behaves on the case study.

Four questions, each answered with a table the dashboard and report reuse:

* Does GENCO's own power-balance residual predict when its deltas disagree
  with AC?  (If yes, the residual is a free trust flag for screening.)
* Where in the network do delta errors sit: near the injection bus and the
  outaged line, or everywhere?
* How well do pooled GENCO loading changes track AC loading changes?
* How does the residual move with capacity, operating condition and outages
  across the full 2,184-scenario matrix?
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def _hop_distances(branches: pd.DataFrame, sources: list[int]) -> dict[int, int]:
    graph = nx.Graph()
    for row in branches.itertuples(index=False):
        graph.add_edge(int(row.from_bus), int(row.to_bus))
    lengths = [nx.single_source_shortest_path_length(graph, s) for s in sources if s in graph]
    return {bus: min(length.get(bus, 99) for length in lengths) for bus in graph.nodes}


def analyse_genco_behaviour(
    manifest: pd.DataFrame,
    genco_summary: pd.DataFrame,
    validation: pd.DataFrame,
    genco_directory: Path,
    ac_directory: Path,
    spatial_directory: Path,
    output_directory: Path,
) -> dict[str, pd.DataFrame]:
    branches = gpd.read_parquet(spatial_directory / "branches.parquet").drop(columns="geometry")
    manifest = manifest.set_index("scenario_id")
    residual = genco_summary.set_index("scenario_id")["quality_or_residual_metric"]

    # A. residual versus validation error, per validated intervention scenario.
    per_scenario = validation.loc[validation["candidate_id"].notna() & validation["branch_active_power_delta_mae_mw"].notna()].copy()
    per_scenario["physics_residual"] = per_scenario["scenario_id"].map(residual)
    for column in ["capacity_mw", "operating_condition", "contingency_id"]:
        per_scenario[column] = per_scenario["scenario_id"].map(manifest[column])
    per_scenario = per_scenario[["scenario_id", "candidate_id", "capacity_mw", "operating_condition", "contingency_id", "physics_residual",
                                 "voltage_delta_mae_pu", "voltage_delta_max_abs_error_pu", "branch_active_power_delta_mae_mw",
                                 "branch_active_power_delta_max_abs_error_mw", "delta_validation_status"]]
    spearman = per_scenario[["physics_residual", "branch_active_power_delta_mae_mw", "branch_active_power_delta_max_abs_error_mw", "voltage_delta_mae_pu"]].corr(method="spearman").iloc[0]

    # B. per-branch delta error versus hop distance from the perturbation.
    rows, pooled = [], []
    for row in per_scenario.itertuples(index=False):
        g = pd.read_csv(genco_directory / row.scenario_id / "branch_results.csv").set_index("branch_id")
        a = pd.read_csv(ac_directory / row.scenario_id / "branch_results.csv").set_index("branch_id")
        joined = g[["loading_delta", "in_service"]].join(a[["loading_delta"]], lsuffix="_genco", rsuffix="_ac").loc[lambda d: d["in_service"]]
        sources = [int(manifest.loc[row.scenario_id, "linked_bus_id"])]
        outage = manifest.loc[row.scenario_id, "outage_branch_id"]
        active = branches.loc[branches["branch_id"].isin(joined.index)]
        if isinstance(outage, str):
            ends = branches.loc[branches["branch_id"].eq(outage), ["from_bus", "to_bus"]].iloc[0]
            sources += [int(ends.from_bus), int(ends.to_bus)]
        distance = _hop_distances(active, sources)
        hops = active.set_index("branch_id").apply(lambda b: min(distance.get(int(b.from_bus), 99), distance.get(int(b.to_bus), 99)), axis=1)
        joined["hops_from_perturbation"] = hops.reindex(joined.index)
        joined["abs_error"] = (joined["loading_delta_genco"] - joined["loading_delta_ac"]).abs()
        joined["scenario_id"] = row.scenario_id
        joined["is_outage_case"] = isinstance(outage, str)
        rows.append(joined.reset_index())
        pooled.append(joined[["loading_delta_genco", "loading_delta_ac"]].assign(scenario_id=row.scenario_id, is_outage_case=isinstance(outage, str)))
    per_branch = pd.concat(rows, ignore_index=True)
    by_distance = per_branch.groupby(["is_outage_case", "hops_from_perturbation"], as_index=False).agg(
        mean_abs_error=("abs_error", "mean"), p90_abs_error=("abs_error", lambda s: float(np.percentile(s, 90))), branches=("abs_error", "size"),
    )
    by_distance = by_distance.loc[by_distance["hops_from_perturbation"] <= 8]

    # C. pooled agreement of loading changes.
    pairs = pd.concat(pooled, ignore_index=True)
    x, y = pairs["loading_delta_ac"].to_numpy(), pairs["loading_delta_genco"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    agreement = pd.DataFrame([{
        "pairs": int(len(pairs)), "slope": float(slope), "intercept": float(intercept), "r2": r2,
        "mae": float(np.abs(y - x).mean()), "share_within_0_05": float((np.abs(y - x) <= 0.05).mean()),
        "sign_agreement_when_ac_change_gt_0_05": float((np.sign(y[np.abs(x) > 0.05]) == np.sign(x[np.abs(x) > 0.05])).mean()),
    }])

    # D. residual across the whole matrix.
    full = genco_summary[["scenario_id", "quality_or_residual_metric", "runtime_seconds"]].merge(
        manifest.reset_index()[["scenario_id", "capacity_mw", "operating_condition", "contingency_id", "is_baseline"]], on="scenario_id")
    residual_by = full.groupby(["contingency_id", "capacity_mw"], as_index=False)["quality_or_residual_metric"].median()
    residual_condition = full.groupby(["operating_condition", "capacity_mw"], as_index=False)["quality_or_residual_metric"].median()

    output_directory.mkdir(parents=True, exist_ok=True)
    per_scenario.to_parquet(output_directory / "genco_residual_vs_validation.parquet", index=False)
    by_distance.to_parquet(output_directory / "genco_error_by_hop_distance.parquet", index=False)
    pairs.to_parquet(output_directory / "genco_vs_ac_loading_delta_pairs.parquet", index=False)
    agreement.to_parquet(output_directory / "genco_delta_agreement.parquet", index=False)
    residual_by.to_parquet(output_directory / "genco_residual_by_contingency.parquet", index=False)
    residual_condition.to_parquet(output_directory / "genco_residual_by_condition.parquet", index=False)
    spearman.to_frame("spearman_vs_residual").to_parquet(output_directory / "genco_residual_spearman.parquet")
    return {"per_scenario": per_scenario, "by_distance": by_distance, "pairs": pairs, "agreement": agreement,
            "residual_by_contingency": residual_by, "residual_by_condition": residual_condition, "spearman": spearman.to_frame()}
