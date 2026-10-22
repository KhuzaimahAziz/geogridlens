"""Resumable GENCO-only screening for the documented case-study matrix.

AC power flow is intentionally not run for all 2,160 interventions.  The
separate native validation workflow remains the conventional-solver check for
priority, near-limit, and holdout scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandapower.networks as pn

from case_study.manifest import scenario_definition
from genco.native_pfdelta_scenarios import build_native_graph, load_snapshot, modify_snapshot
from genco.scenario_inference import (
    ScenarioPrediction,
    load_inference_components,
    output_tables,
    raw_branch_mapping,
    run_model,
    save_genco_scenarios,
)


def _apply_condition_deltas(
    predictions: list[tuple[pd.Series, ScenarioPrediction]],
) -> None:
    """Set every result's deltas against its matching operating-condition base."""

    bases = {
        str(row["scenario_id"]): prediction
        for row, prediction in predictions
        if bool(row["is_baseline"])
    }
    for row, prediction in predictions:
        baseline_id = str(row["baseline_id"])
        if baseline_id not in bases:
            raise ValueError(f"Missing base case '{baseline_id}' for {prediction.scenario_id}.")
        base = bases[baseline_id]
        base_voltage = base.bus_results.set_index("bus_id")["voltage_pu"]
        base_loading = base.branch_results.set_index("branch_id")["loading_ratio"]
        prediction.bus_results["voltage_delta_pu"] = (
            prediction.bus_results["voltage_pu"]
            - prediction.bus_results["bus_id"].map(base_voltage)
        )
        prediction.branch_results["loading_delta"] = (
            prediction.branch_results["loading_ratio"]
            - prediction.branch_results["branch_id"].map(base_loading)
        )


def _discard_stale_results(manifest: pd.DataFrame, output_directory: Path) -> None:
    """Delete cached results whose scenario definition changed since they were written.

    Resume keys on ``scenario_id`` only, so a candidate re-linked to another bus
    or an N-1 set re-selected from new geodata would otherwise be served stale
    predictions.  The comparison uses the manifest copy saved beside results.
    """

    import shutil

    saved_path = output_directory / "scenario_manifest.csv"
    if not saved_path.exists():
        return
    keys = ["scenario_id", "linked_bus_id", "active_power_mw", "load_multiplier", "pfdelta_branch_index"]
    saved = pd.read_csv(saved_path)
    if any(column not in saved.columns for column in keys):
        shutil.rmtree(output_directory)
        print("Discarded GENCO results written by an older manifest schema.")
        return
    def normalised(frame: pd.DataFrame) -> pd.DataFrame:
        table = frame[keys].copy()
        for column in keys[1:]:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(6).fillna(-1).astype(str)
        return table.set_index("scenario_id")

    current = normalised(manifest)
    previous = normalised(saved)
    stale = [sid for sid in current.index if sid not in previous.index or not current.loc[sid].equals(previous.loc[sid])]
    orphaned = [sid for sid in previous.index if sid not in current.index]
    for scenario_id in stale + orphaned:
        folder = output_directory / str(scenario_id)
        if folder.exists():
            shutil.rmtree(folder)
    if stale or orphaned:
        print(f"Discarded {len(stale)} changed and {len(orphaned)} orphaned cached GENCO scenario folders.")


def run_case_study_genco(
    manifest: pd.DataFrame,
    *,
    raw_directory: Path,
    source_scenario_id: int,
    model_config_path: Path,
    checkpoint_path: Path,
    normalizer_stats_path: Path,
    output_directory: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """Run released GENCO inference for the manifest and write mapped results.

    ``resume`` skips a scenario only when all three generated result files are
    present.  This makes an interrupted long screening run safely restartable.
    """

    required = {"scenario_id", "baseline_id", "is_baseline", "change"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    _discard_stale_results(manifest, output_directory)
    ordered = manifest.sort_values(["is_baseline", "scenario_id"], ascending=[False, True])
    source = load_snapshot(raw_directory, source_scenario_id)
    branch_lookup_base = raw_branch_mapping(pn.case118(), source.branch)
    args, normalizer, task = load_inference_components(
        model_config_path, checkpoint_path, normalizer_stats_path,
    )
    predictions: list[tuple[pd.Series, ScenarioPrediction]] = []
    for ordinal, (_, row) in enumerate(ordered.iterrows(), start=1):
        scenario_id = str(row["scenario_id"])
        folder = output_directory / scenario_id
        cached = [folder / "bus_results.csv", folder / "branch_results.csv"]
        if resume and all(path.exists() for path in cached):
            # Reconstructing predictions is necessary to recalculate stable
            # deltas if a prior run was interrupted before summary generation.
            bus = pd.read_csv(cached[0])
            branch = pd.read_csv(cached[1])
            predictions.append((row, ScenarioPrediction(scenario_id, bus, branch, 0.0, float("nan"), None)))
            continue
        modified = modify_snapshot(source, scenario_definition(row))
        branch_lookup = branch_lookup_base.copy()
        branch_lookup["in_service"] = modified.branch["br_status"].astype(bool).to_numpy()
        graph = build_native_graph(modified, branch_lookup)
        output, graph, runtime = run_model(graph, args, normalizer, task)
        buses, branches, quality = output_tables(
            scenario_id, graph, output, branch_lookup, normalizer,
        )
        predictions.append((row, ScenarioPrediction(scenario_id, buses, branches, runtime, quality, None)))
        if ordinal % 50 == 0:
            print(f"GENCO screened {ordinal:,}/{len(ordered):,} scenarios")

    _apply_condition_deltas(predictions)
    summary = save_genco_scenarios([prediction for _, prediction in predictions], output_directory)
    manifest_columns = [
        "scenario_id", "baseline_id", "candidate_id", "linked_bus_id", "capacity_mw",
        "active_power_mw", "operating_condition", "contingency_id", "contingency_kind",
        "is_baseline",
    ]
    available = [column for column in manifest_columns if column in manifest.columns]
    summary = summary.merge(manifest[available], on="scenario_id", how="left", validate="one_to_one")
    summary.to_csv(output_directory / "scenario_summary.csv", index=False)
    manifest.to_csv(output_directory / "scenario_manifest.csv", index=False)
    return summary
