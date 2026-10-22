"""Select and AC-validate representative scenarios from the GENCO matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged
import yaml

from case_study.manifest import scenario_definition
from genco.native_pfdelta_scenarios import solve_ac_snapshot_scenario, load_snapshot, modify_snapshot
from genco.scenario_inference import raw_branch_mapping
from validation.compare_power_flow import compare_deltas, compare_result_pairs, meets_criteria, meets_delta_criteria


def select_validation_cases(
    manifest: pd.DataFrame,
    genco_summary: pd.DataFrame,
    final_scores: pd.DataFrame,
    scenario_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a documented mixed validation subset without cherry-picking."""

    selected: dict[str, str] = {}
    for row in manifest.loc[manifest["is_baseline"]].itertuples():
        selected[str(row.scenario_id)] = "Operating-condition baseline" if row.contingency_kind == "none" else "Outage-only baseline (N-1, no project)"
    balanced = final_scores.loc[
        final_scores["weight_profile"].eq("balanced") & ~final_scores["spatially_excluded"]
    ].nsmallest(10, "rank")
    top_ids = balanced["candidate_id"].tolist()
    for candidate_id in top_ids:
        case = manifest.loc[
            manifest["candidate_id"].eq(candidate_id)
            & manifest["capacity_mw"].eq(100.0)
            & manifest["operating_condition"].eq("normal_demand")
            & manifest["contingency_id"].eq("n0")
        ].iloc[0]
        selected[str(case.scenario_id)] = "Top-ranked candidate, normal N-0"
    # Every N-1 outage is represented for the three highest balanced candidates.
    for candidate_id in top_ids[:3]:
        cases = manifest.loc[
            manifest["candidate_id"].eq(candidate_id)
            & manifest["capacity_mw"].eq(100.0)
            & manifest["operating_condition"].eq("normal_demand")
            & manifest["contingency_kind"].eq("branch_outage")
        ]
        for row in cases.itertuples():
            selected[str(row.scenario_id)] = "Top-ranked candidate, N-1 coverage"
    # Stable evenly spaced holdouts avoid a subjective random sample.
    holdouts = manifest.loc[
        ~manifest["is_baseline"] & manifest["contingency_id"].eq("n0")
    ].sort_values("scenario_id").iloc[::45].head(8)
    for row in holdouts.itertuples():
        selected[str(row.scenario_id)] = "Deterministic N-0 holdout"
    near_limit = genco_summary.loc[~genco_summary["is_baseline"].fillna(False)].nlargest(8, "max_loading_ratio")
    for row in near_limit.itertuples():
        selected[str(row.scenario_id)] = "Highest predicted absolute loading diagnostic"
    if scenario_metrics is not None:
        # Near-limit by predicted *change*, the quantity screening actually uses.
        largest_delta = scenario_metrics.loc[~scenario_metrics["is_baseline"]].nlargest(6, "max_loading_delta")
        for row in largest_delta.itertuples():
            selected.setdefault(str(row.scenario_id), "Largest predicted loading-change diagnostic")
        # Lowest-ranked non-excluded candidate, so the subset is not only winners.
        worst = final_scores.loc[final_scores["weight_profile"].eq("balanced") & ~final_scores["spatially_excluded"]].nlargest(2, "rank")
        for candidate_id in worst["candidate_id"]:
            case = manifest.loc[manifest["candidate_id"].eq(candidate_id) & manifest["capacity_mw"].eq(100.0) & manifest["operating_condition"].eq("normal_demand") & manifest["contingency_id"].eq("n0")].iloc[0]
            selected.setdefault(str(case.scenario_id), "Lowest-ranked candidate, normal N-0")
    result = manifest.loc[manifest["scenario_id"].isin(selected)].copy()
    result["validation_reason"] = result["scenario_id"].map(selected)
    return result.sort_values(["is_baseline", "scenario_id"], ascending=[False, True]).reset_index(drop=True)


def _add_ac_deltas(results: dict[str, tuple[pd.DataFrame, pd.DataFrame]], selected: pd.DataFrame) -> None:
    for row in selected.itertuples():
        scenario_id = str(row.scenario_id)
        baseline_id = str(row.baseline_id)
        buses, branches = results[scenario_id]
        base_bus, base_branch = results[baseline_id]
        buses["voltage_delta_pu"] = buses["voltage_pu"] - buses["bus_id"].map(base_bus.set_index("bus_id")["voltage_pu"])
        branches["loading_delta"] = branches["loading_ratio"] - branches["branch_id"].map(base_branch.set_index("branch_id")["loading_ratio"])


def run_ac_validation_subset(
    selected: pd.DataFrame,
    *,
    raw_directory: Path,
    source_scenario_id: int,
    genco_directory: Path,
    output_directory: Path,
    validation_config_path: Path,
) -> pd.DataFrame:
    """Run matching AC cases and write absolute plus condition-delta comparisons."""

    config = yaml.safe_load(validation_config_path.read_text(encoding="utf-8")) or {}
    source = load_snapshot(raw_directory, source_scenario_id)
    branch_base = raw_branch_mapping(pn.case118(), source.branch)
    ac_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    ac_failures: dict[str, str] = {}
    ac_summaries: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        definition = scenario_definition(pd.Series(row._asdict()))
        modified = modify_snapshot(source, definition)
        lookup = branch_base.copy()
        lookup["in_service"] = modified.branch["br_status"].astype(bool).to_numpy()
        try:
            buses, branches, summary = solve_ac_snapshot_scenario(modified, definition, lookup)
        except LoadflowNotConverged as error:
            ac_failures[str(row.scenario_id)] = str(error)
            continue
        ac_results[str(row.scenario_id)] = (buses, branches)
        ac_summaries.append(summary)
    solvable = selected.loc[selected["scenario_id"].isin(ac_results)].copy()
    # A failed condition baseline prevents delta validation of only the cases
    # belonging to that condition; it does not hide the rest of the subset.
    solvable = solvable.loc[solvable["baseline_id"].isin(ac_results)]
    _add_ac_deltas(ac_results, solvable)
    output_directory.mkdir(parents=True, exist_ok=True)
    for scenario_id, (buses, branches) in ac_results.items():
        folder = output_directory / scenario_id
        folder.mkdir(parents=True, exist_ok=True)
        buses.to_csv(folder / "bus_results.csv", index=False)
        branches.to_csv(folder / "branch_results.csv", index=False)
    pd.DataFrame(ac_summaries).to_csv(output_directory / "scenario_summary.csv", index=False)

    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        if scenario_id not in ac_results:
            rows.append({
                "scenario_id": scenario_id,
                "baseline_id": row.baseline_id,
                "candidate_id": row.candidate_id,
                "validation_reason": row.validation_reason,
                "validation_status": "Infeasible",
                "delta_validation_status": "Needs verification",
                "screening_status": "AC reference did not converge",
                "note": ac_failures.get(scenario_id, "AC reference unavailable."),
            })
            continue
        ac_bus, ac_branch = ac_results[scenario_id]
        genco_bus = pd.read_csv(genco_directory / scenario_id / "bus_results.csv")
        genco_branch = pd.read_csv(genco_directory / scenario_id / "branch_results.csv")
        absolute = compare_result_pairs(ac_bus, ac_branch, genco_bus, genco_branch)
        is_baseline = bool(row.is_baseline)
        record: dict[str, Any] = {
            "scenario_id": scenario_id,
            "baseline_id": row.baseline_id,
            "candidate_id": row.candidate_id,
            "validation_reason": row.validation_reason,
            "validation_status": "AC verified" if meets_criteria(absolute, config["acceptance_criteria"]) else "GENCO screened",
            **absolute,
        }
        if is_baseline:
            record["delta_validation_status"] = "Baseline / not applicable"
            record["screening_status"] = "GENCO baseline diagnostic"
        else:
            base_id = str(row.baseline_id)
            if base_id not in ac_results:
                record["delta_validation_status"] = "Needs verification"
                record["screening_status"] = "AC baseline did not converge"
                record["note"] = "Delta comparison unavailable because the matching AC baseline did not converge."
                rows.append(record)
                continue
            base_ac_bus, base_ac_branch = ac_results[base_id]
            base_genco_bus = pd.read_csv(genco_directory / base_id / "bus_results.csv")
            base_genco_branch = pd.read_csv(genco_directory / base_id / "branch_results.csv")
            delta = compare_deltas(base_ac_bus, base_ac_branch, base_genco_bus, base_genco_branch, ac_bus, ac_branch, genco_bus, genco_branch)
            passed = meets_delta_criteria(delta, config["delta_acceptance_criteria"])
            record.update(delta)
            record["delta_validation_status"] = "Delta validated" if passed else "Delta screened"
            record["screening_status"] = "GENCO delta-validated" if passed else "GENCO screened"
        rows.append(record)
    summary = pd.DataFrame(rows)
    selected.to_csv(output_directory / "validation_selection.csv", index=False)
    summary.to_csv(output_directory / "validation_summary.csv", index=False)
    summary.to_parquet(output_directory / "ac_validation_results.parquet", index=False)
    return summary
