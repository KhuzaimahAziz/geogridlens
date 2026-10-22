"""Aggregate GENCO case-study output into the five decision-map tables.

Tables written to ``outputs/case_study/indicators``:

* ``scenario_metrics.parquet`` - one row per GENCO scenario (2,164).
* ``candidate_grid_scores.parquet`` - Map 3 robustness across the N-0 matrix.
* ``candidate_contingency_detail.parquet`` - every N-1 case per candidate.
* ``candidate_spatial_scores.parquet`` - Map 1 inputs from the real layers.
* ``candidate_final_scores.parquet`` - Map 5 for every weight profile.
* ``branch_hazard_consequence_scores.parquet`` - Map 4 bivariate table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


SPATIAL_COLUMNS = [
    "candidate_id", "linked_bus_id", "connection_distance_m", "lon", "lat", "pv_yield_kwh_per_kwp",
    "solar_resource_score", "land_suitability_score", "mean_slope_deg", "slope_constructability_score",
    "protected_fraction", "settlement_buffer_fraction", "frac_tree_cover", "frac_shrubland", "frac_grassland",
    "frac_cropland", "frac_built_up", "frac_water", "burned_fraction_2015_2024", "fuel_fraction",
    "wildfire_exposure_score", "corridor_wildfire_exposure", "spatial_opportunity_score", "spatially_excluded",
    "exclusion_reason",
]


def _severity(value: float, threshold: float) -> float:
    return float(np.clip(value / threshold, 0.0, 1.0))


def _read_pair(results_directory: Path, scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = results_directory / scenario_id
    return pd.read_csv(folder / "bus_results.csv"), pd.read_csv(folder / "branch_results.csv")


def _scenario_metrics(manifest: pd.DataFrame, results_directory: Path) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        bus, branch = _read_pair(results_directory, str(row.scenario_id))
        active = branch.loc[branch["in_service"]]
        rows.append({
            "scenario_id": row.scenario_id,
            "max_abs_voltage_delta_pu": float(bus["voltage_delta_pu"].abs().max()),
            "min_voltage_pu": float(bus["voltage_pu"].min()),
            "max_voltage_pu": float(bus["voltage_pu"].max()),
            "max_loading_delta": float(active["loading_delta"].clip(lower=0).max()),
            "max_loading_ratio": float(active["loading_ratio"].max()),
            "branches_loading_increase_gt_0_10": int((active["loading_delta"] > 0.10).sum()),
            "buses_voltage_shift_gt_0_03": int((bus["voltage_delta_pu"].abs() > 0.03).sum()),
            "worst_branch_id": str(active.loc[active["loading_delta"].idxmax(), "branch_id"]) if len(active) else "",
            "worst_bus_id": int(bus.loc[bus["voltage_delta_pu"].abs().idxmax(), "bus_id"]),
        })
    metrics = pd.DataFrame(rows)
    keep = ["scenario_id", "baseline_id", "candidate_id", "linked_bus_id", "capacity_mw", "active_power_mw",
            "operating_condition", "contingency_id", "contingency_kind", "outage_branch_id", "is_baseline"]
    return manifest[[c for c in keep if c in manifest.columns]].merge(metrics, on="scenario_id", validate="one_to_one")


def _classify(final_score: pd.Series, excluded: pd.Series) -> np.ndarray:
    return np.select(
        [excluded, final_score.ge(0.75), final_score.ge(0.50)],
        ["Spatially excluded", "Priority for deeper study", "Conditionally suitable"],
        default="Constrained",
    )


def _dominant_land(row: pd.Series) -> str:
    fractions = {
        "grassland": row.get("frac_grassland", 0), "cropland": row.get("frac_cropland", 0), "tree cover": row.get("frac_tree_cover", 0),
        "shrubland": row.get("frac_shrubland", 0), "built-up": row.get("frac_built_up", 0), "water": row.get("frac_water", 0),
    }
    return max(fractions, key=fractions.get)


def _reasons(row: pd.Series, thresholds: dict[str, Any], exposure_reference: float) -> str:
    bullets = []
    bullets.append(f"PVGIS yield {row['pv_yield_kwh_per_kwp']:.0f} kWh/kWp; dominant land cover {_dominant_land(row)}; mean slope {row['mean_slope_deg']:.1f} deg")
    if row["protected_fraction"] > 0:
        bullets.append(f"{row['protected_fraction']:.0%} of the cell overlaps a protected area")
    else:
        bullets.append("No protected-area overlap")
    if row["spatially_excluded"]:
        bullets.append(f"Excluded before grid analysis: {row['exclusion_reason']}")
    bullets.append(f"Bus {int(row['linked_bus_id'])} at {row['connection_distance_m'] / 1000:.1f} km")
    v, l = row["max_abs_voltage_delta_pu"], row["max_loading_delta"]
    if v < float(thresholds["warning_voltage_delta_pu"]) and l < float(thresholds["warning_loading_delta"]):
        bullets.append("Within warning thresholds in every N-0 capacity x condition case")
    else:
        bullets.append(f"Worst N-0 case: |dV| {v:.3f} pu, loading +{l:.2f} (warnings in {row['loading_warning_frequency']:.0%} of cases)")
    if row["worst_n1_loading_increase"] >= float(thresholds["max_loading_delta"]):
        bullets.append(f"Project adds +{row['worst_n1_loading_increase']:.2f} loading on {row['worst_n1_branch']} under {row['worst_n1_contingency']}: beyond the N-1 change limit")
    else:
        bullets.append(f"Largest project-attributable N-1 change +{row['worst_n1_loading_increase']:.2f} on {row['worst_n1_branch']} ({row['worst_n1_contingency']})")
    rel = row["corridor_wildfire_exposure"] / max(exposure_reference, 1e-9)
    if rel >= 0.67:
        bullets.append("High wildfire exposure along the connection corridor")
    elif rel >= 0.33:
        bullets.append("Moderate wildfire exposure along the connection corridor")
    else:
        bullets.append("Low wildfire exposure along the connection corridor")
    return " | ".join(bullets)


def score_case_study(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    results_directory: Path,
    spatial_directory: Path,
    output_directory: Path,
) -> dict[str, pd.DataFrame]:
    """Write the case-study analytical tables from completed GENCO output."""

    thresholds = config["screening_thresholds"]
    missing = next((results_directory / sid / "bus_results.csv" for sid in manifest["scenario_id"] if not (results_directory / sid / "bus_results.csv").exists()), None)
    if missing is not None:
        raise FileNotFoundError(f"Case-study GENCO result is missing: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = _scenario_metrics(manifest, results_directory)
    metrics.to_parquet(output_directory / "scenario_metrics.parquet", index=False)

    candidates = gpd.read_parquet(spatial_directory / "candidate_sites.parquet")
    hexes = gpd.read_parquet(spatial_directory / "hex_cells.parquet")
    exposure_reference = float(hexes["wildfire_exposure_score"].quantile(0.95))
    n0 = metrics.loc[~metrics["is_baseline"] & metrics["contingency_id"].eq("n0")]
    grid_rows, detail_rows, impact_frames = [], [], []
    for candidate_id, cases in n0.groupby("candidate_id"):
        voltage = cases["max_abs_voltage_delta_pu"].to_numpy()
        loading = cases["max_loading_delta"].to_numpy()
        v_risk = np.array([_severity(v, float(thresholds["max_voltage_delta_pu"])) for v in voltage])
        l_risk = np.array([_severity(v, float(thresholds["max_loading_delta"])) for v in loading])
        risk = 0.5 * v_risk + 0.5 * l_risk
        worst = cases.iloc[int(np.argmax(risk))]
        grid_rows.append({
            "candidate_id": candidate_id, "n0_case_count": int(len(cases)),
            "max_abs_voltage_delta_pu": float(voltage.max()), "max_loading_delta": float(loading.max()),
            "voltage_warning_frequency": float((voltage >= float(thresholds["warning_voltage_delta_pu"])).mean()),
            "loading_warning_frequency": float((loading >= float(thresholds["warning_loading_delta"])).mean()),
            "limit_breach_frequency": float(((voltage >= float(thresholds["max_voltage_delta_pu"])) | (loading >= float(thresholds["max_loading_delta"]))).mean()),
            "worst_n0_scenario": worst["scenario_id"], "worst_n0_branch": worst["worst_branch_id"],
            "grid_risk": float(risk.mean()), "grid_score": float(1 - risk.mean()),
        })
        n1 = metrics.loc[metrics["candidate_id"].eq(candidate_id) & metrics["contingency_kind"].eq("branch_outage")]
        for case in n1.itertuples(index=False):
            pair = cases.loc[cases["capacity_mw"].eq(case.capacity_mw) & cases["operating_condition"].eq(case.operating_condition)]
            if len(pair) != 1:
                raise ValueError(f"Cannot pair N-1 case '{case.scenario_id}' with its N-0 case.")
            n0_id = str(pair.iloc[0]["scenario_id"])
            outage_base_id = f"{case.baseline_id}__{case.contingency_id}"
            _, normal = _read_pair(results_directory, n0_id)
            base_bus, outage_base = _read_pair(results_directory, outage_base_id)
            case_bus, outage = _read_pair(results_directory, str(case.scenario_id))
            joined = (
                normal.set_index("branch_id")[["loading_ratio"]].rename(columns={"loading_ratio": "loading_n0_project"})
                .join(outage_base.set_index("branch_id")[["loading_ratio"]].rename(columns={"loading_ratio": "loading_n1_no_project"}))
                .join(outage.set_index("branch_id")[["loading_ratio", "in_service"]].rename(columns={"loading_ratio": "loading_n1_project"}))
            )
            joined = joined.loc[joined["in_service"]]
            # Project-attributable change under the outage (what the candidate adds on top of the outage itself).
            joined["project_increase_under_outage"] = (joined["loading_n1_project"] - joined["loading_n1_no_project"]).clip(lower=0)
            # Total redistribution a branch receives versus the candidate's own N-0 case (outage + project).
            joined["loading_increase"] = (joined["loading_n1_project"] - joined["loading_n0_project"]).clip(lower=0)
            top = joined["project_increase_under_outage"].idxmax()
            voltage_shift = float((case_bus.set_index("bus_id")["voltage_pu"] - base_bus.set_index("bus_id")["voltage_pu"]).abs().max())
            project_increase = float(joined["project_increase_under_outage"].max())
            post_outage_peak = float(joined["loading_n1_project"].max())
            severity = 0.5 * _severity(voltage_shift, float(thresholds["max_voltage_delta_pu"])) + 0.5 * _severity(project_increase, float(thresholds["max_loading_delta"]))
            detail_rows.append({
                "candidate_id": candidate_id, "scenario_id": case.scenario_id, "n0_scenario_id": n0_id, "outage_baseline_id": outage_base_id,
                "capacity_mw": case.capacity_mw, "operating_condition": case.operating_condition,
                "contingency_id": case.contingency_id, "outage_branch_id": case.outage_branch_id,
                "max_loading_increase": project_increase, "total_redistribution_max": float(joined["loading_increase"].max()),
                "post_outage_peak_loading": post_outage_peak, "most_affected_branch": str(top), "voltage_shift_pu": voltage_shift,
                "contingency_severity": severity,
            })
            impact = joined[["loading_increase", "project_increase_under_outage"]].reset_index()
            impact["scenario_id"], impact["candidate_id"], impact["outage_branch_id"] = case.scenario_id, candidate_id, case.outage_branch_id
            impact_frames.append(impact)
    grid = pd.DataFrame(grid_rows)
    detail = pd.DataFrame(detail_rows)
    resilience_cfg = config.get("resilience_weights", {"contingency": 0.7, "wildfire_corridor": 0.3})
    spatial = pd.DataFrame(candidates.drop(columns="geometry"))[[c for c in SPATIAL_COLUMNS if c in candidates.columns]].rename(columns={"spatial_opportunity_score": "spatial_score"})
    worst_n1 = detail.sort_values("max_loading_increase", ascending=False).drop_duplicates("candidate_id")[["candidate_id", "max_loading_increase", "most_affected_branch", "contingency_id"]]
    worst_n1 = worst_n1.rename(columns={"max_loading_increase": "worst_n1_loading_increase", "most_affected_branch": "worst_n1_branch", "contingency_id": "worst_n1_contingency"})
    resilience = detail.groupby("candidate_id", as_index=False).agg(
        n1_case_count=("scenario_id", "count"), contingency_risk=("contingency_severity", "mean"),
        n1_limit_breach_frequency=("max_loading_increase", lambda s: float((s >= float(thresholds["max_loading_delta"])).mean())),
    ).merge(worst_n1, on="candidate_id")
    combined = spatial.merge(grid, on="candidate_id", validate="one_to_one").merge(resilience, on="candidate_id", validate="one_to_one")
    combined["corridor_wildfire_relative"] = np.clip(combined["corridor_wildfire_exposure"] / max(exposure_reference, 1e-9), 0, 1)
    combined["resilience_score"] = 1 - (
        float(resilience_cfg["contingency"]) * combined["contingency_risk"]
        + float(resilience_cfg["wildfire_corridor"]) * combined["corridor_wildfire_relative"]
    )
    combined["reasons"] = combined.apply(lambda row: _reasons(row, thresholds, exposure_reference), axis=1)
    profiles = []
    for name, weights in config["weight_profiles"].items():
        table = combined.copy()
        table["weight_profile"] = name
        table["final_score"] = (
            float(weights["grid"]) * table["grid_score"] + float(weights["resilience"]) * table["resilience_score"] + float(weights["spatial"]) * table["spatial_score"]
        )
        table.loc[table["spatially_excluded"], "final_score"] = np.nan
        table["rank"] = table["final_score"].rank(ascending=False, method="min").astype("Int64")
        table["classification"] = _classify(table["final_score"], table["spatially_excluded"])
        profiles.append(table)
    final = pd.concat(profiles, ignore_index=True)
    # Rank stability across preference profiles.
    pivot = final.pivot(index="candidate_id", columns="weight_profile", values="rank")
    stability = pd.DataFrame({
        "candidate_id": pivot.index,
        "best_rank": pivot.min(axis=1).to_numpy(), "worst_rank": pivot.max(axis=1).to_numpy(),
        "top10_in_all_profiles": pivot.le(10).all(axis=1).to_numpy(),
    })
    final = final.merge(stability, on="candidate_id", how="left")

    impacts = pd.concat(impact_frames, ignore_index=True)
    branches = gpd.read_parquet(spatial_directory / "branches.parquet").drop(columns="geometry")
    redistribution = impacts.groupby("branch_id", as_index=False).agg(
        max_loading_increase=("loading_increase", "max"), mean_loading_increase=("loading_increase", "mean"),
        p95_loading_increase=("loading_increase", lambda s: float(np.percentile(s, 95))), affected_n1_cases=("scenario_id", "nunique"),
    )
    # Consequence of the outage itself: outage-only baseline versus N-0 baseline, per condition (no project involved).
    outage_rows = []
    for base in manifest.loc[manifest["is_baseline"] & manifest["contingency_kind"].eq("branch_outage")].itertuples(index=False):
        _, n0_branch = _read_pair(results_directory, str(base.baseline_id))
        _, n1_branch = _read_pair(results_directory, str(base.scenario_id))
        j = n0_branch.set_index("branch_id")[["loading_ratio"]].join(n1_branch.set_index("branch_id")[["loading_ratio", "in_service"]], lsuffix="_n0", rsuffix="_n1")
        j = j.loc[j["in_service"]]
        increase = (j["loading_ratio_n1"] - j["loading_ratio_n0"]).clip(lower=0)
        outage_rows.append({"branch_id": base.outage_branch_id, "operating_condition": base.operating_condition, "increase": float(increase.max()), "branch": str(increase.idxmax()), "lines_over_0_10": int((increase > 0.10).sum())})
    outage_table = pd.DataFrame(outage_rows)
    outage_consequence = outage_table.groupby("branch_id", as_index=False).agg(
        outage_max_loading_increase_elsewhere=("increase", "max"), outage_mean_loading_increase_elsewhere=("increase", "mean"),
        outage_lines_over_0_10=("lines_over_0_10", "max"),
    )
    outage_table.to_parquet(output_directory / "outage_consequence_by_condition.parquet", index=False)
    hazard = branches.merge(redistribution, on="branch_id", how="left").merge(outage_consequence, on="branch_id", how="left")
    hazard[["max_loading_increase", "mean_loading_increase", "p95_loading_increase"]] = hazard[["max_loading_increase", "mean_loading_increase", "p95_loading_increase"]].fillna(0.0)
    hazard["affected_n1_cases"] = hazard["affected_n1_cases"].fillna(0).astype(int)
    hazard["electrical_consequence_score"] = hazard["p95_loading_increase"].map(lambda v: _severity(v, float(thresholds["max_loading_delta"])))
    hazard["outage_consequence_score"] = hazard["outage_max_loading_increase_elsewhere"].map(lambda v: _severity(v, float(thresholds["max_loading_delta"])) if pd.notna(v) else np.nan)
    hazard["priority_score"] = hazard["wildfire_exposure_score"] * hazard["electrical_consequence_score"]
    e_low, e_high = hazard["wildfire_exposure_score"].quantile([1 / 3, 2 / 3])
    c_low, c_high = hazard["electrical_consequence_score"].quantile([1 / 3, 2 / 3])
    hazard["exposure_tercile"] = np.where(hazard["wildfire_exposure_score"] < e_low, "low", np.where(hazard["wildfire_exposure_score"] < e_high, "medium", "high"))
    hazard["consequence_tercile"] = np.where(hazard["electrical_consequence_score"] < c_low, "low", np.where(hazard["electrical_consequence_score"] < c_high, "medium", "high"))
    hazard["bivariate_class"] = hazard["exposure_tercile"] + " exposure / " + hazard["consequence_tercile"] + " consequence"
    hazard["is_screened_outage"] = hazard["branch_id"].isin(manifest["outage_branch_id"].dropna().unique())

    grid.to_parquet(output_directory / "candidate_grid_scores.parquet", index=False)
    detail.to_parquet(output_directory / "candidate_contingency_detail.parquet", index=False)
    spatial.to_parquet(output_directory / "candidate_spatial_scores.parquet", index=False)
    final.to_parquet(output_directory / "candidate_final_scores.parquet", index=False)
    hazard.to_parquet(output_directory / "branch_hazard_consequence_scores.parquet", index=False)
    plan = final.loc[final["weight_profile"].eq("balanced") & ~final["spatially_excluded"]].nsmallest(10, "rank")
    plan[["candidate_id", "rank", "classification"]].assign(validation_reason="Top-ranked GENCO-screened candidate; validate with AC").to_csv(output_directory / "ac_validation_plan.csv", index=False)
    return {"metrics": metrics, "grid": grid, "detail": detail, "spatial": spatial, "final": final, "branch_hazard": hazard}
