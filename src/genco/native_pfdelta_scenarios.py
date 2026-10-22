"""Scenario edits on a released PFDelta operating snapshot.

The same feasible snapshot is modified for both engines: GENCO receives the
native Graphkit graph, pandapower receives an equivalent MATPOWER case for the
independent AC check.  Keeping one source of truth avoids comparing a
pandapower default case against a GENCO training-distribution snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandapower as pp
import torch
from pandapower.converter.pypower.from_ppc import from_ppc
from pandapower.pypower.idx_brch import ANGMAX, ANGMIN, BR_B, BR_R, BR_STATUS, BR_X, F_BUS, RATE_A, RATE_B, RATE_C, SHIFT, TAP, T_BUS
from pandapower.pypower.idx_bus import BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE, PD, QD, VA, VM, VMAX, VMIN, ZONE
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG
from torch_geometric.data import HeteroData

from genco.scenario_inference import (
    BUS_FEATURES,
    FORWARD_BRANCH_FEATURES,
    REVERSE_BRANCH_FEATURES,
    VOLTAGE_VIOLATION_MAX_PU,
    VOLTAGE_VIOLATION_MIN_PU,
    VOLTAGE_WARNING_MAX_PU,
    VOLTAGE_WARNING_MIN_PU,
)


GEN_FEATURES = [
    "p_mw", "min_p_mw", "max_p_mw", "cp0_eur", "cp1_eur_per_mw",
    "cp2_eur_per_mw2", "in_service",
]


@dataclass
class NativeSnapshot:
    bus: pd.DataFrame
    gen: pd.DataFrame
    branch: pd.DataFrame
    source_scenario_id: int


def load_snapshot(raw_directory: Path, scenario_id: int) -> NativeSnapshot:
    """Read one released PFDelta scenario without altering its operating point."""

    tables = {
        name: pd.read_parquet(raw_directory / f"{name}_data.parquet")
        for name in ("bus", "gen", "branch")
    }
    selected = {
        name: table.loc[table["scenario"] == scenario_id].copy()
        for name, table in tables.items()
    }
    expected = {"bus": 118, "gen": 54, "branch": 186}
    for name, count in expected.items():
        if len(selected[name]) != count:
            raise ValueError(
                f"PFDelta scenario {scenario_id} has {len(selected[name])} {name} rows; expected {count}.",
            )
    return NativeSnapshot(
        bus=selected["bus"].sort_values("bus").reset_index(drop=True),
        gen=selected["gen"].sort_values("idx").reset_index(drop=True),
        branch=selected["branch"].sort_values("idx").reset_index(drop=True),
        source_scenario_id=scenario_id,
    )


def modify_snapshot(snapshot: NativeSnapshot, definition: dict[str, Any]) -> NativeSnapshot:
    """Apply condition load, fixed-P solar injection, and optional N-1 outage.

    ``load_multiplier`` defaults to one so the original five-case workflow is
    unchanged.  The case-study matrix uses it to represent low, normal, peak,
    and high-renewable operating conditions from the same released snapshot.
    """

    bus = snapshot.bus.copy()
    gen = snapshot.gen.copy()
    branch = snapshot.branch.copy()
    load_multiplier = float(definition.get("load_multiplier", 1.0))
    if load_multiplier <= 0:
        raise ValueError("load_multiplier must be positive.")
    bus["Pd"] *= load_multiplier
    bus["Qd"] *= load_multiplier
    change = definition["change"]
    if change in {"none", "branch_outage"}:
        pass
    elif change in {"generation_injection", "generation_injection_with_branch_outage"}:
        # Fixed-P sgen at a PQ bus is equivalent to a reduction in Pd.
        internal_bus = int(definition["bus_id"]) - 1
        target = bus["bus"] == internal_bus
        if target.sum() != 1:
            raise ValueError(f"Cannot locate configured Bus {definition['bus_id']} in PFDelta snapshot.")
        bus.loc[target, "Pd"] -= float(definition["active_power_mw"])
    else:
        raise NotImplementedError(f"Unsupported native PFDelta change '{change}'.")
    if change in {"generation_injection_with_branch_outage", "branch_outage"}:
        branch_index = int(definition["pfdelta_branch_index"])
        target = branch["idx"] == branch_index
        if target.sum() != 1:
            raise ValueError(f"Cannot locate PFDelta branch {branch_index} for N-1 scenario.")
        branch.loc[target, "br_status"] = 0.0
    return NativeSnapshot(bus=bus, gen=gen, branch=branch, source_scenario_id=snapshot.source_scenario_id)


def build_native_graph(snapshot: NativeSnapshot, branch_lookup: pd.DataFrame) -> HeteroData:
    """Build precisely the graph structure used by Graphkit's on-disk dataset."""

    data = HeteroData()
    aggregate_limits = (
        snapshot.gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]]
        .sum()
        .reset_index()
    )
    bus_features = snapshot.bus.merge(aggregate_limits, on="bus", how="left").fillna(0)
    data["bus"].x = torch.tensor(bus_features[BUS_FEATURES].to_numpy(), dtype=torch.float)
    data["bus"].y = data["bus"].x[:, :5].clone()
    data["gen"].x = torch.tensor(snapshot.gen[GEN_FEATURES].to_numpy(), dtype=torch.float)
    data["gen"].y = data["gen"].x[:, :1].clone()
    branch = snapshot.branch.copy()
    # Power flow is intentionally masked by Graphkit; zeroing makes absence of
    # target flow data explicit before the published masking transform runs.
    branch[["pf", "qf", "pt", "qt"]] = 0.0
    forward_edges = torch.tensor(branch[["from_bus", "to_bus"]].to_numpy().T.copy(), dtype=torch.long)
    reverse_edges = torch.tensor(branch[["to_bus", "from_bus"]].to_numpy().T.copy(), dtype=torch.long)
    data["bus", "connects", "bus"].edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
    forward = torch.tensor(branch[FORWARD_BRANCH_FEATURES].to_numpy(), dtype=torch.float)
    reverse = torch.tensor(branch[REVERSE_BRANCH_FEATURES].to_numpy(), dtype=torch.float)
    data["bus", "connects", "bus"].edge_attr = torch.cat([forward, reverse], dim=0)
    data["bus", "connects", "bus"].y = torch.zeros((2 * len(branch), 2), dtype=torch.float)
    generator_indices = np.arange(len(snapshot.gen), dtype=np.int64)
    generator_buses = snapshot.gen["bus"].to_numpy(dtype=np.int64)
    gen_to_bus = torch.tensor(np.vstack([generator_indices, generator_buses]), dtype=torch.long)
    data["gen", "connected_to", "bus"].edge_index = gen_to_bus
    data["bus", "connected_to", "gen"].edge_index = gen_to_bus.flip(0)
    data["scenario_id"] = torch.tensor([snapshot.source_scenario_id], dtype=torch.long)
    if not np.array_equal(branch_lookup["in_service"].to_numpy(), branch["br_status"].astype(bool).to_numpy()):
        raise ValueError("Branch lookup and native scenario statuses are inconsistent.")
    return data


def ppc_from_snapshot(snapshot: NativeSnapshot) -> dict[str, Any]:
    """Create a MATPOWER case for the same native input snapshot."""

    bus = np.zeros((len(snapshot.bus), 13), dtype=float)
    bus[:, BUS_I] = snapshot.bus["bus"]
    bus[:, BUS_TYPE] = np.select(
        [snapshot.bus["REF"].astype(bool), snapshot.bus["PV"].astype(bool)],
        [3, 2],
        default=1,
    )
    bus[:, PD] = snapshot.bus["Pd"]
    bus[:, QD] = snapshot.bus["Qd"]
    bus[:, 4] = snapshot.bus["GS"]
    bus[:, BS] = snapshot.bus["BS"]
    bus[:, BUS_AREA] = 1
    bus[:, VM] = snapshot.bus["Vm"]
    bus[:, VA] = snapshot.bus["Va"]
    bus[:, BASE_KV] = snapshot.bus["vn_kv"]
    bus[:, ZONE] = 1
    bus[:, VMAX] = snapshot.bus["max_vm_pu"]
    bus[:, VMIN] = snapshot.bus["min_vm_pu"]

    gen = np.zeros((len(snapshot.gen), 21), dtype=float)
    gen[:, GEN_BUS] = snapshot.gen["bus"]
    gen[:, PG] = snapshot.gen["p_mw"]
    gen[:, QG] = snapshot.gen["q_mvar"]
    gen[:, QMAX] = snapshot.gen["max_q_mvar"]
    gen[:, QMIN] = snapshot.gen["min_q_mvar"]
    vm_by_bus = snapshot.bus.set_index("bus")["Vm"]
    gen[:, VG] = snapshot.gen["bus"].map(vm_by_bus)
    gen[:, MBASE] = 100.0
    gen[:, GEN_STATUS] = snapshot.gen["in_service"]
    gen[:, PMAX] = snapshot.gen["max_p_mw"]
    gen[:, PMIN] = snapshot.gen["min_p_mw"]

    branch = np.zeros((len(snapshot.branch), 13), dtype=float)
    branch[:, F_BUS] = snapshot.branch["from_bus"]
    branch[:, T_BUS] = snapshot.branch["to_bus"]
    branch[:, BR_R] = snapshot.branch["r"]
    branch[:, BR_X] = snapshot.branch["x"]
    branch[:, BR_B] = snapshot.branch["b"]
    branch[:, RATE_A] = snapshot.branch["rate_a"]
    branch[:, RATE_B] = snapshot.branch["rate_a"]
    branch[:, RATE_C] = snapshot.branch["rate_a"]
    branch[:, TAP] = snapshot.branch["tap"]
    branch[:, SHIFT] = snapshot.branch["shift"]
    branch[:, BR_STATUS] = snapshot.branch["br_status"]
    branch[:, ANGMIN] = snapshot.branch["ang_min"]
    branch[:, ANGMAX] = snapshot.branch["ang_max"]
    return {"version": "2", "baseMVA": 100.0, "bus": bus, "gen": gen, "branch": branch}


def solve_ac_snapshot_scenario(
    snapshot: NativeSnapshot,
    definition: dict[str, Any],
    branch_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Solve a matching pandapower case and return the common output schema."""

    net = from_ppc(ppc_from_snapshot(snapshot), f_hz=60, validate_conversion=False)
    started_at = pd.Timestamp.now().timestamp()
    # The released snapshot includes a solved voltage state, so retain it as
    # the Newton initial point rather than replacing it with a DC/flat guess.
    pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=False, max_iteration=30)
    runtime_seconds = pd.Timestamp.now().timestamp() - started_at
    if not net.converged:
        raise RuntimeError(f"Native AC scenario '{definition['id']}' did not converge.")
    solved = net._ppc
    solved_bus = solved["bus"]
    solved_gen = solved["gen"]
    bus_id = snapshot.bus["bus"].to_numpy(dtype=int) + 1
    generation = np.zeros(len(bus_id), dtype=float)
    reactive_generation = np.zeros(len(bus_id), dtype=float)
    for row in solved_gen:
        generation[int(row[GEN_BUS])] += row[PG]
        reactive_generation[int(row[GEN_BUS])] += row[QG]
    bus_results = pd.DataFrame(
        {
            "scenario_id": str(definition["id"]),
            "bus_id": bus_id,
            "voltage_pu": solved_bus[:, VM],
            "voltage_delta_pu": 0.0,
            "active_injection_mw": generation - snapshot.bus["Pd"].to_numpy(),
            "reactive_injection_mvar": reactive_generation - snapshot.bus["Qd"].to_numpy(),
        },
    )
    bus_results["voltage_warning"] = (bus_results["voltage_pu"] < VOLTAGE_WARNING_MIN_PU) | (bus_results["voltage_pu"] > VOLTAGE_WARNING_MAX_PU)
    bus_results["voltage_violation"] = (bus_results["voltage_pu"] < VOLTAGE_VIOLATION_MIN_PU) | (bus_results["voltage_pu"] > VOLTAGE_VIOLATION_MAX_PU)
    # pandapower re-orders (and may merge) branches internally, so its solved
    # branch array cannot be read back in PFDelta order.  Reconstruct sending-end
    # flows from the solved bus voltages with the snapshot's own branch
    # admittances instead; that keeps every flow on its original branch.
    voltage = solved_bus[:, VM] * np.exp(1j * np.deg2rad(solved_bus[:, VA]))
    from_index = snapshot.branch["from_bus"].to_numpy(dtype=int)
    to_index = snapshot.branch["to_bus"].to_numpy(dtype=int)
    y_ff = snapshot.branch["Yff_r"].to_numpy() + 1j * snapshot.branch["Yff_i"].to_numpy()
    y_ft = snapshot.branch["Yft_r"].to_numpy() + 1j * snapshot.branch["Yft_i"].to_numpy()
    sending = voltage[from_index] * np.conj(y_ff * voltage[from_index] + y_ft * voltage[to_index]) * 100.0
    branch_results = branch_lookup.copy()
    branch_results["p_flow_mw"] = sending.real * branch_results["flow_sign"]
    branch_results["q_flow_mvar"] = sending.imag * branch_results["flow_sign"]
    branch_results.loc[~branch_results["in_service"], ["p_flow_mw", "q_flow_mvar"]] = 0.0
    branch_results["loading_ratio"] = np.hypot(branch_results["p_flow_mw"], branch_results["q_flow_mvar"]) / branch_results["rate_a"].replace(0, np.nan)
    branch_results.loc[~branch_results["in_service"], "loading_ratio"] = 0.0
    branch_results["loading_delta"] = 0.0
    branch_results["overload_flag"] = branch_results["in_service"] & (branch_results["loading_ratio"] > 1.0)
    branch_results.insert(0, "scenario_id", str(definition["id"]))
    branch_results = branch_results[["scenario_id", "branch_id", "from_bus", "to_bus", "p_flow_mw", "q_flow_mvar", "loading_ratio", "loading_delta", "in_service", "overload_flag"]]
    summary = {
        "scenario_id": str(definition["id"]),
        "max_loading_ratio": float(branch_results["loading_ratio"].max()),
        "min_voltage_pu": float(bus_results["voltage_pu"].min()),
        "max_voltage_pu": float(bus_results["voltage_pu"].max()),
        "overloaded_line_count": int(branch_results["overload_flag"].sum()),
        "voltage_violation_count": int(bus_results["voltage_violation"].sum()),
        "runtime_seconds": runtime_seconds,
        "quality_or_residual_metric": None,
        "validation_status": "Needs verification",
        "reference_solver_status": "AC solved",
        "outage_branch_id": definition.get("branch_id"),
    }
    return bus_results, branch_results, summary


