"""Run the released GENCO PFDelta checkpoint and map its output to project tables.

Graphkit's released PowerFlow task has no ``predict_step``, so this module
loads the checkpoint, applies the published normalisation and masking
transforms, runs the model once per graph, and reconstructs branch flows from
the predicted voltages and the PFDelta branch admittances.  It also owns the
stable mapping between PFDelta branch order and the project's ``branch_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pandapower as pp
import torch
import yaml
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import BS, GS, PD_H, QD_H, QG_H
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task,
    get_task_transforms,
    load_normalizer,
)
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)


# Screening bands for bus voltage magnitude (per unit).
VOLTAGE_WARNING_MIN_PU = 0.95
VOLTAGE_WARNING_MAX_PU = 1.05
VOLTAGE_VIOLATION_MIN_PU = 0.90
VOLTAGE_VIOLATION_MAX_PU = 1.10


BUS_FEATURES = [
    "Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF", "min_vm_pu",
    "max_vm_pu", "min_q_mvar", "max_q_mvar", "GS", "BS", "vn_kv",
]
FORWARD_BRANCH_FEATURES = [
    "pf", "qf", "Yff_r", "Yff_i", "Yft_r", "Yft_i", "tap", "ang_min",
    "ang_max", "rate_a", "br_status",
]
REVERSE_BRANCH_FEATURES = [
    "pt", "qt", "Ytt_r", "Ytt_i", "Ytf_r", "Ytf_i", "tap", "ang_min",
    "ang_max", "rate_a", "br_status",
]


@dataclass
class ScenarioPrediction:
    """In-memory GENCO result for one configured intervention."""

    scenario_id: str
    bus_results: pd.DataFrame
    branch_results: pd.DataFrame
    runtime_seconds: float
    quality_metric: float
    outage_branch_id: str | None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def raw_branch_mapping(net: Any, branches: pd.DataFrame) -> pd.DataFrame:
    """Map PFDelta/MATPOWER branch order to pandapower component identifiers.

    The two case118 representations contain the same 186 physical branches,
    but their line ordering differs.  Some pandapower transformer conversions
    slightly alter equivalent r/x/b values, so endpoint pairs provide the
    portable key for correct N-1 masking and AC-result joins.
    """

    # pandapower exposes its stable component-to-PYPOWER ordering after case
    # conversion.  ``runpp`` is used only to obtain that ordering; no res_bus,
    # res_line, voltage, angle, or flow value is read into the GENCO graph.
    pp.runpp(net, algorithm="nr", init="dc", enforce_q_lims=True)
    ppc_branch = net._ppc["branch"]
    if len(ppc_branch) != len(branches):
        raise ValueError("pandapower and PFDelta IEEE-118 branch counts differ.")
    component_lookup = {
        index: (
            f"line:{index}",
            bool(row["in_service"]),
            int(row["from_bus"]),
            int(row["to_bus"]),
        )
        for index, row in net.line.iterrows()
    }
    component_lookup.update(
        {
            173 + index: (
                f"trafo:{index}",
                bool(row["in_service"]),
                int(row["hv_bus"]),
                int(row["lv_bus"]),
            )
            for index, row in net.trafo.iterrows()
        },
    )
    # Deterministic assignment.  Parallel circuits share an endpoint pair, so
    # endpoints alone cannot say which pandapower component is which PFDelta
    # branch; a bipartite matching would pick arbitrarily and could differ from
    # one process to the next, silently swapping parallel lines between cached
    # and fresh results.  Resolve ties by nearest series impedance instead.
    has_impedance = {"r", "x"}.issubset(branches.columns)
    used: set[int] = set()
    assignment: dict[int, int] = {}
    for raw_position, (_, branch) in enumerate(branches.iterrows()):
        endpoints = {int(branch["from_bus"]), int(branch["to_bus"])}
        candidates = [
            index for index, ppc_row in enumerate(ppc_branch)
            if {int(ppc_row[0]), int(ppc_row[1])} == endpoints and index not in used
        ]
        if not candidates:
            raise ValueError(
                "Could not map PFDelta branch "
                f"{int(branch['idx'])} to a pandapower component.",
            )
        if has_impedance and len(candidates) > 1:
            candidates.sort(key=lambda index: abs(float(ppc_branch[index][2]) - float(branch["r"])) + abs(float(ppc_branch[index][3]) - float(branch["x"])))
        chosen = candidates[0]
        used.add(chosen)
        assignment[raw_position] = chosen
    if len(assignment) != len(branches):
        raise ValueError("Could not construct a complete PFDelta-to-pandapower branch mapping.")

    records: list[dict[str, Any]] = []
    for raw_position, (_, branch) in enumerate(branches.iterrows()):
        component_index = assignment[raw_position]
        branch_id, in_service, component_from, component_to = component_lookup[component_index]
        raw_from = int(branch["from_bus"])
        raw_to = int(branch["to_bus"])
        if {raw_from, raw_to} != {component_from, component_to}:
            raise ValueError("Mapped components do not share a physical endpoint pair.")
        records.append(
            {
                "branch_id": branch_id,
                "from_bus": int(net.bus.at[component_from, "name"]),
                "to_bus": int(net.bus.at[component_to, "name"]),
                "rate_a": float(branch["rate_a"]),
                "in_service": in_service,
                "flow_sign": 1.0 if raw_from == component_from else -1.0,
            },
        )
    return pd.DataFrame(records)


def load_inference_components(
    model_config_path: Path,
    checkpoint_path: Path,
    normalizer_stats_path: Path,
):
    config = _read_yaml(model_config_path)
    args = NestedNamespace(**config)
    normalizer = load_normalizer(args)
    stats = torch.load(normalizer_stats_path, map_location="cpu", weights_only=True)
    normalizer.fit_from_dict(stats)
    task = get_task(args, [normalizer])
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    task.load_state_dict(state_dict)
    task.eval()
    return args, normalizer, task


def run_model(graph: HeteroData, args: Any, normalizer: Any, task: Any) -> tuple[torch.Tensor, HeteroData, float]:
    """Apply released preprocessing and execute the checkpoint once on CPU."""

    normalizer.transform(graph)
    transform = get_task_transforms(args)
    graph = transform(graph)
    started_at = perf_counter()
    with torch.no_grad():
        output = task(
            graph.x_dict,
            graph.edge_index_dict,
            graph.edge_attr_dict,
            graph.mask_dict,
        )["bus"].detach().cpu()
    return output, graph, perf_counter() - started_at


def output_tables(
    scenario_id: str,
    graph: HeteroData,
    output: torch.Tensor,
    branch_lookup: pd.DataFrame,
    normalizer: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Denormalize GENCO outputs and reconstruct bus/branch screening tables."""

    base_mva = float(normalizer.baseMVA)
    output = output.clone()
    output[:, 2:4] *= base_mva
    edge_attributes = graph["bus", "connects", "bus"].edge_attr.detach().cpu().clone()
    # This mirrors HeteroDataMVANormalizer.inverse_transform for the Y terms.
    edge_attributes[:, 2:6] *= base_mva
    edge_index = graph["bus", "connects", "bus"].edge_index.detach().cpu()
    flows = ComputeBranchFlow()(output, edge_index, edge_attributes)
    p_flow = flows[0].detach().cpu().numpy()
    q_flow = flows[1].detach().cpu().numpy()
    active_count = int(branch_lookup["in_service"].sum())
    if len(p_flow) != 2 * active_count:
        raise ValueError("GENCO branch-edge count does not match active physical branches.")

    active_lookup = branch_lookup.loc[branch_lookup["in_service"]].reset_index(drop=True)
    active_results = active_lookup.copy()
    active_results["p_flow_mw"] = p_flow[:active_count] * active_results["flow_sign"]
    active_results["q_flow_mvar"] = q_flow[:active_count] * active_results["flow_sign"]
    apparent_power = np.hypot(active_results["p_flow_mw"], active_results["q_flow_mvar"])
    active_results["loading_ratio"] = apparent_power / active_results["rate_a"].replace(0, np.nan)
    active_results["overload_flag"] = active_results["loading_ratio"] > 1.0
    inactive_results = branch_lookup.loc[~branch_lookup["in_service"]].copy()
    inactive_results["p_flow_mw"] = 0.0
    inactive_results["q_flow_mvar"] = 0.0
    inactive_results["loading_ratio"] = 0.0
    inactive_results["overload_flag"] = False
    branch_results = pd.concat([active_results, inactive_results], ignore_index=True).sort_index()
    branch_results.insert(0, "scenario_id", scenario_id)
    branch_results["loading_delta"] = 0.0
    branch_results = branch_results[
        [
            "scenario_id", "branch_id", "from_bus", "to_bus", "p_flow_mw", "q_flow_mvar",
            "loading_ratio", "loading_delta", "in_service", "overload_flag",
        ]
    ]

    raw_bus = graph["bus"].x.detach().cpu().clone()
    raw_bus[:, [PD_H, QD_H, QG_H]] *= base_mva
    raw_bus[:, [GS, BS]] *= base_mva
    node_injection = ComputeNodeInjection()(*flows, edge_index, output.shape[0])
    residual_p, residual_q = ComputeNodeResiduals()(node_injection[0], node_injection[1], output, raw_bus)
    quality_metric = float(torch.sqrt(residual_p.square() + residual_q.square()).mean().item())
    bus_ids = np.arange(1, output.shape[0] + 1)
    bus_results = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "bus_id": bus_ids,
            "voltage_pu": output[:, 0].numpy(),
            "voltage_delta_pu": 0.0,
            "active_injection_mw": (output[:, 2] - raw_bus[:, PD_H]).numpy(),
            "reactive_injection_mvar": (output[:, 3] - raw_bus[:, QD_H]).numpy(),
        },
    )
    bus_results["voltage_warning"] = (
        (bus_results["voltage_pu"] < VOLTAGE_WARNING_MIN_PU)
        | (bus_results["voltage_pu"] > VOLTAGE_WARNING_MAX_PU)
    )
    bus_results["voltage_violation"] = (
        (bus_results["voltage_pu"] < VOLTAGE_VIOLATION_MIN_PU)
        | (bus_results["voltage_pu"] > VOLTAGE_VIOLATION_MAX_PU)
    )
    return bus_results, branch_results, quality_metric


def save_genco_scenarios(predictions: list[ScenarioPrediction], output_directory: Path) -> pd.DataFrame:
    """Write stable per-scenario GENCO result files and a summary table."""

    summaries: list[dict[str, Any]] = []
    for prediction in predictions:
        folder = output_directory / prediction.scenario_id
        folder.mkdir(parents=True, exist_ok=True)
        prediction.bus_results.to_csv(folder / "bus_results.csv", index=False)
        prediction.branch_results.to_csv(folder / "branch_results.csv", index=False)
        summaries.append(
            {
                "scenario_id": prediction.scenario_id,
                "max_loading_ratio": float(prediction.branch_results["loading_ratio"].max()),
                "min_voltage_pu": float(prediction.bus_results["voltage_pu"].min()),
                "max_voltage_pu": float(prediction.bus_results["voltage_pu"].max()),
                "overloaded_line_count": int(prediction.branch_results["overload_flag"].sum()),
                "voltage_violation_count": int(prediction.bus_results["voltage_violation"].sum()),
                "runtime_seconds": prediction.runtime_seconds,
                "quality_or_residual_metric": prediction.quality_metric,
                "validation_status": "GENCO screened",
                "outage_branch_id": prediction.outage_branch_id,
            },
        )
    summary = pd.DataFrame(summaries).sort_values("scenario_id")
    summary.to_csv(output_directory / "scenario_summary.csv", index=False)
    return summary
