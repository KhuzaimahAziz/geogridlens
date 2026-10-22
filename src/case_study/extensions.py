"""Extra GENCO experiments that probe the model beyond the core screening matrix.

* seed ensemble   - the same 2,184 scenarios through all released seeds; spread
                    per scenario and rank stability across seeds.
* outage ranking  - an outage-only baseline for every connectivity-preserving
                    branch under every condition: GENCO's own consequence table
                    for all lines, not just the five screened ones.
* load sweep      - one site at 100 MW across a wide load-multiplier range, with
                    AC at every step: error versus distance from the training
                    operating point.
* capacity sweep  - one good and one poor site from 0 to 200 MW in fine steps,
                    used for the animated hero figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged

from case_study.genco_screening import run_case_study_genco
from genco.native_pfdelta_scenarios import build_native_graph, load_snapshot, modify_snapshot, solve_ac_snapshot_scenario
from genco.scenario_inference import load_inference_components, output_tables, raw_branch_mapping, run_model


class Context:
    def __init__(self, root: Path, config: dict[str, Any], native: dict[str, Any]) -> None:
        self.root = root
        self.config = config
        self.raw_root = root / "data" / "raw" / "pfdelta_task1.3" / "case118"
        self.raw_directory = self.raw_root / native["snapshot"]["relative_directory"]
        self.source_scenario_id = int(native["snapshot"]["scenario_id"])
        self.model_config = root / "configs" / "pfdelta_task1_3_seed1_cpu.yaml"
        self.stats = self.raw_root / "train" / "processed" / "data_stats_HeteroDataMVANormalizer.pt"
        self.case = root / config["outputs"]["root"]
        self.indicators = root / config["outputs"]["indicators_directory"]
        self.spatial = root / config["outputs"]["spatial_directory"]

    def checkpoint(self, seed: int) -> Path:
        return self.root / "models" / "genco-pfdelta" / "task_1.3" / f"seed{seed}" / "last.pt"


def _scenario_metrics(directory: Path, scenario_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in scenario_ids:
        bus = pd.read_csv(directory / sid / "bus_results.csv")
        branch = pd.read_csv(directory / sid / "branch_results.csv")
        active = branch.loc[branch["in_service"]]
        rows.append({"scenario_id": sid, "max_abs_voltage_delta_pu": float(bus["voltage_delta_pu"].abs().max()),
                     "max_loading_delta": float(active["loading_delta"].clip(lower=0).max()),
                     "worst_branch_id": str(active.loc[active["loading_delta"].idxmax(), "branch_id"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- seed ensemble


def run_seed_ensemble(ctx: Context, manifest: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    thresholds = ctx.config["screening_thresholds"]
    per_seed = []
    for seed in seeds:
        directory = ctx.case / "genco_results" if seed == 1 else ctx.case / f"genco_results_seed{seed}"
        if seed != 1:
            if not ctx.checkpoint(seed).exists():
                print(f"seed {seed}: checkpoint missing, skipped")
                continue
            run_case_study_genco(
                manifest, raw_directory=ctx.raw_directory, source_scenario_id=ctx.source_scenario_id,
                model_config_path=ctx.model_config, checkpoint_path=ctx.checkpoint(seed),
                normalizer_stats_path=ctx.stats, output_directory=directory,
            )
        metrics = _scenario_metrics(directory, manifest["scenario_id"].tolist())
        metrics["seed"] = seed
        per_seed.append(metrics)
    table = pd.concat(per_seed, ignore_index=True)
    table = table.merge(manifest[["scenario_id", "candidate_id", "capacity_mw", "operating_condition", "contingency_id", "is_baseline"]], on="scenario_id")
    table.to_parquet(ctx.indicators / "seed_ensemble_scenarios.parquet", index=False)
    spread = table.groupby("scenario_id").agg(
        loading_delta_mean=("max_loading_delta", "mean"), loading_delta_std=("max_loading_delta", "std"),
        loading_delta_range=("max_loading_delta", lambda s: float(s.max() - s.min())),
        voltage_delta_std=("max_abs_voltage_delta_pu", "std"), seeds=("seed", "nunique"),
        worst_branch_agreement=("worst_branch_id", lambda s: float(s.nunique() == 1)),
    ).reset_index()
    spread.to_parquet(ctx.indicators / "seed_ensemble_spread.parquet", index=False)
    # Grid score per seed and rank stability across seeds.
    n0 = table.loc[~table["is_baseline"] & table["contingency_id"].eq("n0")].copy()
    n0["risk"] = 0.5 * np.clip(n0["max_abs_voltage_delta_pu"] / float(thresholds["max_voltage_delta_pu"]), 0, 1) + 0.5 * np.clip(n0["max_loading_delta"] / float(thresholds["max_loading_delta"]), 0, 1)
    grid = n0.groupby(["candidate_id", "seed"], as_index=False)["risk"].mean()
    grid["grid_score"] = 1 - grid["risk"]
    grid["rank"] = grid.groupby("seed")["grid_score"].rank(ascending=False, method="min").astype(int)
    pivot = grid.pivot(index="candidate_id", columns="seed", values="rank")
    stability = pd.DataFrame({"candidate_id": pivot.index, "best_rank": pivot.min(axis=1).to_numpy(), "worst_rank": pivot.max(axis=1).to_numpy(),
                              "rank_range": (pivot.max(axis=1) - pivot.min(axis=1)).to_numpy(), "top5_in_all_seeds": pivot.le(5).all(axis=1).to_numpy()})
    stability = stability.merge(grid.pivot(index="candidate_id", columns="seed", values="grid_score").add_prefix("grid_score_seed").reset_index(), on="candidate_id")
    stability.to_parquet(ctx.indicators / "seed_rank_stability.parquet", index=False)
    return stability


# ---------------------------------------------------------- outage ranking


def run_all_branch_outages(ctx: Context) -> pd.DataFrame:
    import geopandas as gpd

    branches = gpd.read_parquet(ctx.spatial / "branches.parquet").drop(columns="geometry")
    graph = nx.MultiGraph()
    for row in branches.loc[branches["in_service"]].itertuples():
        graph.add_edge(int(row.from_bus), int(row.to_bus), key=row.branch_id)
    conditions = ctx.config["operating_conditions"]
    records = []
    for cond in conditions:
        records.append({"scenario_id": f"base__{cond['id']}", "baseline_id": f"base__{cond['id']}", "candidate_id": pd.NA, "linked_bus_id": pd.NA,
                        "capacity_mw": 0.0, "active_power_mw": 0.0, "operating_condition": cond["id"], "load_multiplier": float(cond["load_multiplier"]),
                        "solar_capacity_factor": float(cond["solar_capacity_factor"]), "contingency_id": "base", "contingency_kind": "none",
                        "pfdelta_branch_index": pd.NA, "outage_branch_id": pd.NA, "change": "none", "is_baseline": True})
    keeps_connected = {}
    for row in branches.loc[branches["in_service"]].itertuples():
        trial = graph.copy()
        trial.remove_edge(int(row.from_bus), int(row.to_bus), key=row.branch_id)
        keeps_connected[row.branch_id] = nx.is_connected(trial)
        if not keeps_connected[row.branch_id]:
            continue
        for cond in conditions:
            records.append({"scenario_id": f"base__{cond['id']}__out_{int(row.pfdelta_branch_index):03d}", "baseline_id": f"base__{cond['id']}",
                            "candidate_id": pd.NA, "linked_bus_id": pd.NA, "capacity_mw": 0.0, "active_power_mw": 0.0, "operating_condition": cond["id"],
                            "load_multiplier": float(cond["load_multiplier"]), "solar_capacity_factor": float(cond["solar_capacity_factor"]),
                            "contingency_id": f"out_{int(row.pfdelta_branch_index):03d}", "contingency_kind": "branch_outage",
                            "pfdelta_branch_index": int(row.pfdelta_branch_index), "outage_branch_id": row.branch_id, "change": "branch_outage", "is_baseline": True})
    manifest = pd.DataFrame(records)
    directory = ctx.case / "genco_results_all_outages"
    run_case_study_genco(manifest, raw_directory=ctx.raw_directory, source_scenario_id=ctx.source_scenario_id, model_config_path=ctx.model_config,
                         checkpoint_path=ctx.checkpoint(1), normalizer_stats_path=ctx.stats, output_directory=directory)
    rows = []
    bases = {sid: pd.read_csv(directory / sid / "branch_results.csv").set_index("branch_id") for sid in manifest.loc[manifest["contingency_kind"].eq("none"), "scenario_id"]}
    for row in manifest.loc[manifest["contingency_kind"].eq("branch_outage")].itertuples(index=False):
        n1 = pd.read_csv(directory / row.scenario_id / "branch_results.csv").set_index("branch_id")
        n0 = bases[row.baseline_id]
        j = n0[["loading_ratio"]].join(n1[["loading_ratio", "in_service"]], lsuffix="_n0", rsuffix="_n1")
        j = j.loc[j["in_service"]]
        increase = (j["loading_ratio_n1"] - j["loading_ratio_n0"]).clip(lower=0)
        rows.append({"branch_id": row.outage_branch_id, "operating_condition": row.operating_condition, "max_loading_increase_elsewhere": float(increase.max()),
                     "most_affected_branch": str(increase.idxmax()), "lines_over_0_10": int((increase > 0.10).sum()), "lines_over_0_20": int((increase > 0.20).sum()),
                     "post_outage_max_loading": float(j["loading_ratio_n1"].max())})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("branch_id", as_index=False).agg(outage_max_loading_increase_elsewhere=("max_loading_increase_elsewhere", "max"),
                                                              outage_mean_loading_increase_elsewhere=("max_loading_increase_elsewhere", "mean"),
                                                              outage_lines_over_0_10=("lines_over_0_10", "max"), outage_lines_over_0_20=("lines_over_0_20", "max"))
    disconnecting = pd.DataFrame({"branch_id": [b for b, ok in keeps_connected.items() if not ok]})
    disconnecting["note"] = "removal splits the network; not screened"
    summary = pd.concat([summary, disconnecting], ignore_index=True)
    detail.to_parquet(ctx.indicators / "outage_consequence_all_branches_by_condition.parquet", index=False)
    summary.to_parquet(ctx.indicators / "outage_consequence_all_branches.parquet", index=False)
    return summary


# --------------------------------------------------------------- sweeps


def _inference(ctx: Context):
    source = load_snapshot(ctx.raw_directory, ctx.source_scenario_id)
    lookup = raw_branch_mapping(pn.case118(), source.branch)
    args, normalizer, task = load_inference_components(ctx.model_config, ctx.checkpoint(1), ctx.stats)

    def genco(definition: dict[str, Any]):
        modified = modify_snapshot(source, definition)
        l = lookup.copy()
        l["in_service"] = modified.branch["br_status"].astype(bool).to_numpy()
        graph = build_native_graph(modified, l)
        output, graph, _ = run_model(graph, args, normalizer, task)
        bus, branch, residual = output_tables(definition["id"], graph, output, l, normalizer)
        return bus, branch, residual

    def ac(definition: dict[str, Any]):
        modified = modify_snapshot(source, definition)
        l = lookup.copy()
        l["in_service"] = modified.branch["br_status"].astype(bool).to_numpy()
        return solve_ac_snapshot_scenario(modified, definition, l)

    return genco, ac


def run_load_sweep(ctx: Context, candidate_id: str, bus_id: int, multipliers: list[float], capacity_mw: float = 100.0, capacity_factor: float = 0.8) -> pd.DataFrame:
    genco, ac = _inference(ctx)
    rows = []
    for mult in multipliers:
        base_def = {"id": f"sweep_base_{mult:.2f}", "change": "none", "load_multiplier": mult}
        case_def = {"id": f"sweep_{candidate_id}_{mult:.2f}", "change": "generation_injection", "load_multiplier": mult, "bus_id": bus_id, "active_power_mw": capacity_mw * capacity_factor}
        g_base_bus, g_base_br, r_base = genco(base_def)
        g_bus, g_br, residual = genco(case_def)
        record = {"candidate_id": candidate_id, "load_multiplier": mult, "genco_residual_base": r_base, "genco_residual_case": residual}
        try:
            a_base_bus, a_base_br, _ = ac(base_def)
            a_bus, a_br, _ = ac(case_def)
        except (LoadflowNotConverged, RuntimeError) as error:
            record.update({"ac_converged": False, "note": str(error)})
            rows.append(record)
            continue
        g_dl = (g_br.set_index("branch_id")["loading_ratio"] - g_base_br.set_index("branch_id")["loading_ratio"])
        a_dl = (a_br.set_index("branch_id")["loading_ratio"] - a_base_br.set_index("branch_id")["loading_ratio"])
        g_dv = (g_bus.set_index("bus_id")["voltage_pu"] - g_base_bus.set_index("bus_id")["voltage_pu"])
        a_dv = (a_bus.set_index("bus_id")["voltage_pu"] - a_base_bus.set_index("bus_id")["voltage_pu"])
        record.update({
            "ac_converged": True,
            "voltage_mae_pu": float((g_bus.set_index("bus_id")["voltage_pu"] - a_bus.set_index("bus_id")["voltage_pu"]).abs().mean()),
            "branch_p_mae_mw": float((g_br.set_index("branch_id")["p_flow_mw"] - a_br.set_index("branch_id")["p_flow_mw"]).abs().mean()),
            "loading_delta_mae": float((g_dl - a_dl).abs().mean()), "loading_delta_max_error": float((g_dl - a_dl).abs().max()),
            "voltage_delta_mae_pu": float((g_dv - a_dv).abs().mean()),
            "genco_max_loading_delta": float(g_dl.clip(lower=0).max()), "ac_max_loading_delta": float(a_dl.clip(lower=0).max()),
            "ac_min_voltage_pu": float(a_bus["voltage_pu"].min()),
        })
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_parquet(ctx.indicators / "load_sweep.parquet", index=False)
    return table


def run_capacity_sweep(ctx: Context, sites: dict[str, int], capacities: list[float], condition: dict[str, Any], output_directory: Path) -> pd.DataFrame:
    genco, _ = _inference(ctx)
    output_directory.mkdir(parents=True, exist_ok=True)
    base_bus, base_br, _ = genco({"id": "capacity_base", "change": "none", "load_multiplier": float(condition["load_multiplier"])})
    rows = []
    for candidate_id, bus_id in sites.items():
        for capacity in capacities:
            sid = f"{candidate_id}__{int(capacity):03d}mw"
            bus, branch, residual = genco({"id": sid, "change": "generation_injection", "load_multiplier": float(condition["load_multiplier"]), "bus_id": bus_id,
                                           "active_power_mw": capacity * float(condition["solar_capacity_factor"])})
            bus["voltage_delta_pu"] = bus["voltage_pu"] - bus["bus_id"].map(base_bus.set_index("bus_id")["voltage_pu"])
            branch["loading_delta"] = branch["loading_ratio"] - branch["branch_id"].map(base_br.set_index("branch_id")["loading_ratio"])
            folder = output_directory / sid
            folder.mkdir(exist_ok=True)
            bus.to_csv(folder / "bus_results.csv", index=False)
            branch.to_csv(folder / "branch_results.csv", index=False)
            rows.append({"scenario_id": sid, "candidate_id": candidate_id, "bus_id": bus_id, "capacity_mw": capacity, "max_loading_delta": float(branch["loading_delta"].clip(lower=0).max()),
                         "max_abs_voltage_delta_pu": float(bus["voltage_delta_pu"].abs().max()), "residual": residual})
    table = pd.DataFrame(rows)
    table.to_csv(output_directory / "capacity_sweep.csv", index=False)
    return table
