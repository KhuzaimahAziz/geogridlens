"""Fast checks for the case-study spatial rules, manifest and scoring logic."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from case_study import spatial_layers  # noqa: E402
from case_study.geodata import StudyArea  # noqa: E402
from case_study.manifest import build_manifest, load_config, scenario_definition  # noqa: E402

CONFIG = load_config(ROOT / "configs" / "case_study.yaml")
SPATIAL = ROOT / CONFIG["outputs"]["spatial_directory"]


def test_study_area_grid_is_metric_and_100m():
    area = StudyArea(-8.45, 37.80, -7.30, 38.70)
    transform, height, width, bounds = area.metric_grid()
    assert transform.a == 100 and transform.e == -100
    assert 950 < width < 1100 and 950 < height < 1100  # ~100 km box
    assert bounds[2] > bounds[0] and bounds[3] > bounds[1]


def test_hex_grid_covers_bounds_without_overlap():
    hexes = spatial_layers._hex_grid((0, 0, 10_000, 10_000), 1000)
    assert len(hexes) > 20
    union_area = hexes.geometry.union_all().area
    assert abs(union_area - hexes.geometry.area.sum()) / union_area < 1e-6


def test_exposure_terciles_are_balanced():
    values = pd.Series(np.linspace(0, 1, 90))
    classes = spatial_layers._exposure_class(values)
    assert sorted(set(classes)) == ["high", "low", "medium"]
    assert 25 <= (classes == "low").sum() <= 35


def test_candidate_selection_rule_gives_distinct_buses():
    rng = np.random.default_rng(1)
    n = 400
    cx, cy = rng.uniform(0, 50_000, n), rng.uniform(0, 50_000, n)
    hexes = gpd.GeoDataFrame({
        "hex_id": [f"H{i}" for i in range(n)], "cx": cx, "cy": cy, "spatially_excluded": rng.random(n) < 0.2,
        "spatial_opportunity_score": rng.random(n),
    }, geometry=[Point(x, y).buffer(500) for x, y in zip(cx, cy)])
    buses = gpd.GeoDataFrame({"bus_id": range(1, 41)}, geometry=[Point(x, y) for x, y in rng.uniform(0, 50_000, (40, 2))])
    config = {"candidate_sites": {"count": 12, "connection_radius_m": 15_000, "pool_score_quantile": 0.5, "distinct_buses": True}}
    hexes = spatial_layers._attach_nearest_bus(hexes, buses)
    chosen = spatial_layers._select_candidates(config, hexes, buses)
    assert len(chosen) == 12
    assert chosen["linked_bus_id"].is_unique
    assert not chosen["spatially_excluded"].any()
    assert (chosen["connection_distance_m"] <= 15_000).all()
    assert list(chosen["candidate_id"]) == [f"A-{i:02d}" for i in range(1, 13)]


@pytest.mark.skipif(not (SPATIAL / "candidate_sites.parquet").exists(), reason="spatial layers not built")
def test_manifest_matrix_counts_and_definitions():
    manifest = build_manifest(CONFIG, SPATIAL)
    assert (~manifest["is_baseline"]).sum() == 2160
    assert manifest["is_baseline"].sum() == 4 + 4 * 5
    assert manifest["scenario_id"].is_unique
    n1 = manifest.loc[manifest["contingency_kind"].eq("branch_outage") & ~manifest["is_baseline"]].iloc[0]
    definition = scenario_definition(n1)
    assert definition["change"] == "generation_injection_with_branch_outage"
    assert definition["bus_id"] == int(n1["linked_bus_id"]) and "pfdelta_branch_index" in definition
    outage_only = manifest.loc[manifest["is_baseline"] & manifest["contingency_kind"].eq("branch_outage")].iloc[0]
    assert scenario_definition(outage_only)["change"] == "branch_outage"


@pytest.mark.skipif(not (SPATIAL / "candidate_sites.parquet").exists(), reason="spatial layers not built")
def test_built_layers_are_consistent():
    candidates = gpd.read_parquet(SPATIAL / "candidate_sites.parquet")
    buses = gpd.read_parquet(SPATIAL / "buses.parquet")
    branches = gpd.read_parquet(SPATIAL / "branches.parquet")
    hexes = gpd.read_parquet(SPATIAL / "hex_cells.parquet")
    assert candidates.crs == buses.crs == branches.crs == hexes.crs
    assert set(candidates["linked_bus_id"]).issubset(set(buses["bus_id"]))
    assert len(buses) == 118 and len(branches) == 186
    assert branches["pfdelta_branch_index"].notna().all() and branches["pfdelta_branch_index"].is_unique
    bus_ids = set(buses["bus_id"])
    assert set(branches["from_bus"]).issubset(bus_ids) and set(branches["to_bus"]).issubset(bus_ids)
    assert candidates["spatial_opportunity_score"].between(0, 1).all()
    assert hexes["spatially_excluded"].mean() < 0.6
    contingencies = pd.read_csv(SPATIAL / "contingencies.csv")
    assert (contingencies["kind"] == "branch_outage").sum() == CONFIG["contingencies"]["n1_selection"]["k"]


PFDELTA = ROOT / "data" / "raw" / "pfdelta_task1.3" / "case118" / "test" / "case118" / "n" / "feasible" / "raw"


@pytest.mark.skipif(not PFDELTA.exists(), reason="PFDelta split not downloaded")
def test_branch_mapping_is_deterministic_and_resolves_parallel_lines_by_impedance():
    import pandapower.networks as pn
    from genco.native_pfdelta_scenarios import load_snapshot
    from genco.scenario_inference import raw_branch_mapping

    snapshot = load_snapshot(PFDELTA, 14558)
    first = raw_branch_mapping(pn.case118(), snapshot.branch)
    second = raw_branch_mapping(pn.case118(), snapshot.branch)
    assert first.equals(second)
    assert first["branch_id"].is_unique and len(first) == 186
    net = pn.case118()
    import pandapower as pp
    pp.runpp(net, algorithm="nr", init="dc", enforce_q_lims=True)
    ppc = net._ppc["branch"]
    component_index = {f"line:{i}": i for i in net.line.index}
    component_index.update({f"trafo:{i}": 173 + i for i in net.trafo.index})
    # Every PFDelta branch must be assigned to the pandapower component with the
    # nearest series reactance among those sharing its endpoints.
    for position, (_, raw) in enumerate(snapshot.branch.iterrows()):
        chosen = component_index[first.iloc[position]["branch_id"]]
        same_endpoints = [i for i, row in enumerate(ppc) if {int(row[0]), int(row[1])} == {int(raw["from_bus"]), int(raw["to_bus"])}]
        best = min(same_endpoints, key=lambda i: abs(float(ppc[i][3]) - float(raw["x"])))
        assert abs(float(ppc[chosen][3]) - float(raw["x"])) <= abs(float(ppc[best][3]) - float(raw["x"])) + 1e-9


@pytest.mark.skipif(not PFDELTA.exists(), reason="PFDelta split not downloaded")
def test_ac_reference_reproduces_snapshot_flows_in_pfdelta_order():
    import pandapower.networks as pn
    from genco.native_pfdelta_scenarios import load_snapshot, solve_ac_snapshot_scenario
    from genco.scenario_inference import raw_branch_mapping

    snapshot = load_snapshot(PFDELTA, 14558)
    lookup = raw_branch_mapping(pn.case118(), snapshot.branch)
    _, branches, _ = solve_ac_snapshot_scenario(snapshot, {"id": "base", "change": "none"}, lookup)
    recovered = branches["p_flow_mw"].to_numpy() * lookup["flow_sign"].to_numpy()
    # The released snapshot is a solved case: re-solving it must give back its own flows, branch by branch.
    assert np.abs(recovered - snapshot.branch["pf"].to_numpy()).max() < 2.0
