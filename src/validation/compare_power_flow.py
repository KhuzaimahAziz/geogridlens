"""Metrics comparing GENCO scenario tables with pandapower AC results.

Absolute agreement and agreement on scenario-minus-baseline changes are kept
separate on purpose: the released checkpoint carries an absolute flow bias on
the project snapshot while its predicted changes track AC closely.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _agreement(reference: pd.Series, prediction: pd.Series) -> float:
    if len(reference) == 0:
        return float("nan")
    return float((reference.astype(bool) == prediction.astype(bool)).mean())


def compare_result_pairs(
    reference_bus: pd.DataFrame,
    reference_branch: pd.DataFrame,
    genco_bus: pd.DataFrame,
    genco_branch: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate configured AC-reference comparison metrics for one scenario."""

    bus = reference_bus.merge(
        genco_bus,
        on="bus_id",
        how="outer",
        suffixes=("_ac", "_genco"),
        indicator=True,
    )
    branch = reference_branch.merge(
        genco_branch,
        on="branch_id",
        how="outer",
        suffixes=("_ac", "_genco"),
        indicator=True,
    )
    bus_overlap = bus.loc[bus["_merge"] == "both"]
    branch_overlap = branch.loc[branch["_merge"] == "both"]
    if bus_overlap.empty or branch_overlap.empty:
        raise ValueError("GENCO and AC outputs have no common buses or branches.")

    voltage_error = bus_overlap["voltage_pu_genco"] - bus_overlap["voltage_pu_ac"]
    active_power_error = (
        branch_overlap["p_flow_mw_genco"] - branch_overlap["p_flow_mw_ac"]
    )
    return {
        "bus_coverage": float(len(bus_overlap) / len(reference_bus)),
        "branch_coverage": float(len(branch_overlap) / len(reference_branch)),
        "voltage_mae_pu": float(voltage_error.abs().mean()),
        "voltage_max_abs_error_pu": float(voltage_error.abs().max()),
        "branch_active_power_mae_mw": float(active_power_error.abs().mean()),
        "branch_active_power_max_abs_error_mw": float(active_power_error.abs().max()),
        "overloaded_branch_agreement": _agreement(
            branch_overlap["overload_flag_ac"],
            branch_overlap["overload_flag_genco"],
        ),
        "voltage_violation_agreement": _agreement(
            bus_overlap["voltage_violation_ac"],
            bus_overlap["voltage_violation_genco"],
        ),
        "unmatched_ac_bus_count": int((bus["_merge"] == "left_only").sum()),
        "unmatched_genco_bus_count": int((bus["_merge"] == "right_only").sum()),
        "unmatched_ac_branch_count": int((branch["_merge"] == "left_only").sum()),
        "unmatched_genco_branch_count": int((branch["_merge"] == "right_only").sum()),
    }


def meets_criteria(metrics: dict[str, Any], criteria: dict[str, float]) -> bool:
    lower_is_better = {
        "voltage_mae_pu",
        "voltage_max_abs_error_pu",
        "branch_active_power_mae_mw",
        "branch_active_power_max_abs_error_mw",
    }
    higher_is_better = {"overloaded_branch_agreement", "voltage_violation_agreement"}
    for metric in lower_is_better:
        if metrics[metric] > float(criteria[metric]):
            return False
    for metric in higher_is_better:
        if metrics[metric] < float(criteria[metric]):
            return False
    return metrics["bus_coverage"] == 1.0 and metrics["branch_coverage"] == 1.0


def compare_deltas(
    base_ac_bus: pd.DataFrame,
    base_ac_branch: pd.DataFrame,
    base_genco_bus: pd.DataFrame,
    base_genco_branch: pd.DataFrame,
    scenario_ac_bus: pd.DataFrame,
    scenario_ac_branch: pd.DataFrame,
    scenario_genco_bus: pd.DataFrame,
    scenario_genco_branch: pd.DataFrame,
) -> dict[str, float]:
    """Compare intervention changes after removing each engine's baseline bias."""

    def delta(frame: pd.DataFrame, base: pd.DataFrame, key: str, value: str) -> pd.Series:
        current = frame.set_index(key)[value]
        baseline = base.set_index(key)[value]
        if set(current.index) != set(baseline.index):
            raise ValueError(f"Scenario and base do not share {key} identifiers.")
        return current - baseline

    ac_voltage = delta(scenario_ac_bus, base_ac_bus, "bus_id", "voltage_pu")
    genco_voltage = delta(scenario_genco_bus, base_genco_bus, "bus_id", "voltage_pu")
    ac_power = delta(scenario_ac_branch, base_ac_branch, "branch_id", "p_flow_mw")
    genco_power = delta(scenario_genco_branch, base_genco_branch, "branch_id", "p_flow_mw")
    voltage_error = genco_voltage - ac_voltage
    power_error = genco_power - ac_power
    return {
        "voltage_delta_mae_pu": float(voltage_error.abs().mean()),
        "voltage_delta_max_abs_error_pu": float(voltage_error.abs().max()),
        "branch_active_power_delta_mae_mw": float(power_error.abs().mean()),
        "branch_active_power_delta_max_abs_error_mw": float(power_error.abs().max()),
    }


def meets_delta_criteria(metrics: dict[str, float], criteria: dict[str, float]) -> bool:
    return all(metrics[name] <= float(limit) for name, limit in criteria.items())
