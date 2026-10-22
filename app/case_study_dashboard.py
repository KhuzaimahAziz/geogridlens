"""GeoGridLens case-study dashboard: real Alentejo geodata x GENCO screening.

Read-only viewer over the artifacts written by the case-study pipeline.  Maps
use a real basemap because terrain, land cover, protected areas, burned areas,
solar resource and settlements are real; the IEEE-118 network drawn on top is
a schematic embedding and every electrical result is illustrative.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen
from matplotlib import colormaps
from matplotlib.colors import to_hex
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / "data" / "spatial" / "case_study"
CASE = ROOT / "outputs" / "case_study"
INDICATORS = CASE / "indicators"
RESULTS = CASE / "genco_results"
AC = CASE / "ac_validation"

AUTHOR, ROLE, TEAM, EMAIL = "Khuzaima Aziz", "Co-Founder, Spatial Data Scientist, Polisense AI", "Polisense AI", "khuzaimaaziz89@gmail.com"
DISCLAIMER = (
    "Geographic layers (terrain, land cover, Natura 2000 and national protected areas, MODIS burned areas, "
    "PVGIS solar yield, OpenStreetMap settlements) are real open data for the Evora-Beja-Alqueva area. "
    "The transmission network is the IEEE-118 benchmark laid out schematically inside that area; bus and line "
    "positions, candidate-to-bus links and all power-flow numbers are illustrative and say nothing about "
    "Portugal's grid or any real interconnection decision."
)
BIVARIATE = {
    ("low", "low"): "#d9d9d9", ("low", "medium"): "#9ecae1", ("low", "high"): "#3182bd",
    ("medium", "low"): "#fff7bc", ("medium", "medium"): "#fe9929", ("medium", "high"): "#ec7014",
    ("high", "low"): "#fee391", ("high", "medium"): "#e31a1c", ("high", "high"): "#800026",
}
ROBUSTNESS_CLASSES = [
    (0.75, "Robustly grid-feasible", "#1a9850"), (0.60, "Feasible, condition-sensitive", "#fee08b"),
    (0.45, "Grid constrained", "#f46d43"), (-1.0, "Frequent thermal/voltage issues", "#d73027"),
]


# ------------------------------------------------------------------ loading


@st.cache_data(show_spinner=False)
def load_layers() -> dict[str, gpd.GeoDataFrame]:
    names = ["study_area", "buses", "branches", "hex_cells", "candidate_sites", "candidate_connections", "protected_areas",
             "burned_area", "settlements", "settlement_buffers", "osm_substations_context"]
    return {name: gpd.read_parquet(SPATIAL / f"{name}.parquet").to_crs("EPSG:4326") for name in names}


@st.cache_data(show_spinner=False)
def load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "metrics": pd.read_parquet(INDICATORS / "scenario_metrics.parquet"),
        "grid": pd.read_parquet(INDICATORS / "candidate_grid_scores.parquet"),
        "detail": pd.read_parquet(INDICATORS / "candidate_contingency_detail.parquet"),
        "spatial": pd.read_parquet(INDICATORS / "candidate_spatial_scores.parquet"),
        "final": pd.read_parquet(INDICATORS / "candidate_final_scores.parquet"),
        "hazard": pd.read_parquet(INDICATORS / "branch_hazard_consequence_scores.parquet"),
        "manifest": pd.read_csv(CASE / "scenario_manifest.csv"),
        "contingencies": pd.read_csv(SPATIAL / "contingencies.csv"),
        "genco_summary": pd.read_csv(RESULTS / "scenario_summary.csv") if (RESULTS / "scenario_summary.csv").exists() else pd.read_parquet(RESULTS / "scenario_summary.parquet"),
    }
    if (AC / "validation_summary.csv").exists():
        tables["validation"] = pd.read_csv(AC / "validation_summary.csv")
        tables["ac_summary"] = pd.read_csv(AC / "scenario_summary.csv")
    behaviour = {
        "residual_vs_validation": "genco_residual_vs_validation.parquet", "error_by_hops": "genco_error_by_hop_distance.parquet",
        "delta_pairs": "genco_vs_ac_loading_delta_pairs.parquet", "delta_agreement": "genco_delta_agreement.parquet",
        "residual_by_contingency": "genco_residual_by_contingency.parquet", "residual_by_condition": "genco_residual_by_condition.parquet",
    }
    if all((INDICATORS / name).exists() for name in behaviour.values()):
        for key, name in behaviour.items():
            tables[key] = pd.read_parquet(INDICATORS / name)
    for key, name in {"load_sweep": "load_sweep.parquet", "seed_spread": "seed_ensemble_spread.parquet", "seed_ranks": "seed_rank_stability.parquet",
                      "outages_all": "outage_consequence_all_branches.parquet"}.items():
        if (INDICATORS / name).exists():
            tables[key] = pd.read_parquet(INDICATORS / name)
    return tables


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((SPATIAL / "spatial_metadata.json").read_text(encoding="utf-8"))


FORCE_CONSOLIDATED = os.environ.get("GEOGRIDLENS_FORCE_CONSOLIDATED") == "1"


@st.cache_data(show_spinner=False)
def _consolidated(root: Path, kind: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"{kind}_results.parquet")


def _scenario_tables(root: Path, scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scenario CSVs when the full run is present, else the consolidated Parquet shipped with the repo."""

    folder = root / scenario_id
    if not FORCE_CONSOLIDATED and (folder / "bus_results.csv").exists():
        return pd.read_csv(folder / "bus_results.csv"), pd.read_csv(folder / "branch_results.csv")
    bus = _consolidated(root, "bus")
    branch = _consolidated(root, "branch")
    return bus.loc[bus["scenario_id"].eq(scenario_id)].reset_index(drop=True), branch.loc[branch["scenario_id"].eq(scenario_id)].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_scenario(scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _scenario_tables(RESULTS, scenario_id)


@st.cache_data(show_spinner=False)
def load_ac_scenario(scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _scenario_tables(AC, scenario_id)


# ------------------------------------------------------------------ helpers


def colour(value: float, cmap: str, vmin: float, vmax: float) -> str:
    if pd.isna(value):
        return "#bdbdbd"
    t = 0.0 if vmax == vmin else float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    return to_hex(colormaps[cmap](t))


def robustness_class(score: float) -> tuple[str, str]:
    for threshold, label, hex_colour in ROBUSTNESS_CLASSES:
        if score >= threshold:
            return label, hex_colour
    return ROBUSTNESS_CLASSES[-1][1:]


def base_map(layers: dict[str, gpd.GeoDataFrame], tiles: str = "OpenStreetMap") -> folium.Map:
    bounds = layers["study_area"].total_bounds
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=9, tiles=tiles, control_scale=True)
    folium.GeoJson(layers["study_area"], name="Study area", style_function=lambda _: {"color": "#333", "weight": 1.5, "fill": False, "dashArray": "6 4"}).add_to(m)
    Fullscreen().add_to(m)
    return m


def add_context_layers(m: folium.Map, layers: dict[str, gpd.GeoDataFrame], show_protected: bool = True, show_burned: bool = True, show_substations: bool = False, show_settlements: bool = True) -> None:
    if show_protected:
        folium.GeoJson(
            layers["protected_areas"], name="Protected areas (Natura 2000 / national, real)",
            style_function=lambda _: {"color": "#1b7837", "weight": 1, "fillColor": "#1b7837", "fillOpacity": 0.18},
            tooltip=folium.GeoJsonTooltip(fields=["site_name", "designation"], aliases=["Site", "Designation"]),
        ).add_to(m)
    if show_burned and len(layers["burned_area"]):
        folium.GeoJson(layers["burned_area"], name="MODIS burned area 2015-2024 (real)", style_function=lambda _: {"color": "#6a1b9a", "weight": 0.5, "fillColor": "#6a1b9a", "fillOpacity": 0.55}).add_to(m)
    if show_settlements:
        towns = layers["settlements"].loc[layers["settlements"]["place"].isin(["city", "town"])]
        group = folium.FeatureGroup(name="Cities and towns (OSM, real)")
        for row in towns.itertuples():
            folium.CircleMarker([row.geometry.y, row.geometry.x], radius=3, color="#222", fill=True, fill_opacity=1, tooltip=f"{row.name} ({row.place})").add_to(group)
        group.add_to(m)
    if show_substations:
        subs = layers["osm_substations_context"].loc[layers["osm_substations_context"]["substation"].eq("transmission")]
        group = folium.FeatureGroup(name="Real REN transmission substations (OSM, context only, NOT modelled)", show=False)
        for row in subs.itertuples():
            folium.RegularPolygonMarker([row.geometry.y, row.geometry.x], number_of_sides=3, radius=7, color="#00838f", fill=True, fill_opacity=0.9, tooltip=f"{row.name} {row.voltage} (context only)").add_to(group)
        group.add_to(m)


def add_network(m: folium.Map, layers: dict[str, gpd.GeoDataFrame], branch_style, bus_style=None, branch_tooltip=None, bus_tooltip=None, name: str = "IEEE-118 (schematic)") -> None:
    group = folium.FeatureGroup(name=f"{name} branches")
    for row in layers["branches"].itertuples():
        style = branch_style(row)
        coords = [(y, x) for x, y in row.geometry.coords]
        folium.PolyLine(coords, tooltip=branch_tooltip(row) if branch_tooltip else row.branch_id, **style).add_to(group)
    group.add_to(m)
    if bus_style is not None:
        bus_group = folium.FeatureGroup(name=f"{name} buses")
        for row in layers["buses"].itertuples():
            style = bus_style(row)
            folium.CircleMarker([row.geometry.y, row.geometry.x], tooltip=bus_tooltip(row) if bus_tooltip else f"Bus {row.bus_id}", **style).add_to(bus_group)
        bus_group.add_to(m)


def add_candidates(m: folium.Map, layers: dict[str, gpd.GeoDataFrame], values: pd.DataFrame, value_column: str, cmap: str, vmin: float, vmax: float, popup_builder=None, title: str = "Candidates", fixed_colour=None) -> None:
    merged = layers["candidate_sites"].merge(values, on="candidate_id", how="left", suffixes=("", "_v"))
    group = folium.FeatureGroup(name=title)
    for row in merged.itertuples():
        score = getattr(row, value_column, np.nan)
        fill = fixed_colour(row) if fixed_colour else colour(score, cmap, vmin, vmax)
        popup = folium.Popup(popup_builder(row), max_width=420) if popup_builder else None
        folium.GeoJson(
            row.geometry.__geo_interface__, style_function=lambda _, c=fill: {"color": "#1f2937", "weight": 2, "fillColor": c, "fillOpacity": 0.9},
            tooltip=f"{row.candidate_id} -> bus {int(row.linked_bus_id)}", popup=popup,
        ).add_to(group)
        folium.Marker([row.lat + 0.012, row.lon], icon=folium.DivIcon(html=f'<div style="font:600 11px sans-serif;color:#111;text-shadow:0 0 3px #fff">{row.candidate_id}</div>')).add_to(group)
    group.add_to(m)
    connections = folium.FeatureGroup(name="Candidate -> bus connectors (schematic)")
    for row in layers["candidate_connections"].itertuples():
        folium.PolyLine([(y, x) for x, y in row.geometry.coords], color="#1f2937", weight=1.2, dash_array="4 4", opacity=0.8).add_to(connections)
    connections.add_to(m)


def legend(m: folium.Map, title: str, entries: list[tuple[str, str]]) -> None:
    items = "".join(f'<div><span style="display:inline-block;width:14px;height:14px;background:{c};margin-right:6px;border:1px solid #777"></span>{label}</div>' for label, c in entries)
    html = f"""{{% macro html(this, kwargs) %}}<div style="position:fixed;bottom:30px;left:12px;z-index:9999;background:rgba(255,255,255,.93);padding:8px 10px;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)"><b>{title}</b>{items}</div>{{% endmacro %}}"""
    macro = MacroElement()
    macro._template = Template(html)
    m.get_root().add_child(macro)


def render(m: folium.Map, height: int = 640) -> None:
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, use_container_width=True, height=height, returned_objects=[])


def candidate_popup(row, extra: str = "") -> str:
    reasons = "".join(f"<li>{item.strip()}</li>" for item in str(getattr(row, "reasons", "")).split("|") if item.strip())
    return (
        f"<b>{row.candidate_id}</b> - 100 MW solar + optional 50 MW battery<br>"
        f"Linked bus {int(row.linked_bus_id)} ({row.connection_distance_m / 1000:.1f} km)<br>"
        f"Spatial {getattr(row, 'spatial_score', float('nan')):.2f} | Grid {getattr(row, 'grid_score', float('nan')):.2f} | "
        f"Resilience {getattr(row, 'resilience_score', float('nan')):.2f} | Final {getattr(row, 'final_score', float('nan')):.2f}<br>"
        f"<b>{getattr(row, 'classification', '')}</b>{extra}<ul style='padding-left:16px;margin:4px 0'>{reasons}</ul>"
    )


# --------------------------------------------------------------------- tabs


def tab_overview(layers, tables, meta) -> None:
    final = tables["final"].loc[tables["final"]["weight_profile"].eq("balanced")]
    hexes = layers["hex_cells"]
    st.markdown(
        "#### Where can a 100 MW solar project connect with low grid stress, on plausible land, away from wildfire-exposed corridors?\n"
        "GENCO screens the whole candidate x capacity x condition x contingency matrix in minutes; pandapower AC validates a documented subset; "
        "real GIS layers decide where candidates can exist at all."
    )
    k = st.columns(6)
    k[0].metric("Hex cells screened", f"{len(hexes):,}")
    k[1].metric("Excluded by real constraints", f"{hexes['spatially_excluded'].mean():.0%}")
    k[2].metric("Candidate sites", f"{len(final):,}")
    k[3].metric("GENCO scenarios", f"{(~tables['manifest']['is_baseline']).sum():,} + 4 baselines")
    if "validation" in tables:
        v = tables["validation"].copy()
    if "note" not in v.columns:
        v["note"] = ""
        k[4].metric("AC-validated cases", f"{len(v):,}", help="Independent pandapower runs on baselines, top candidates, N-1, holdouts, near-limit")
        delta_ok = v["delta_validation_status"].eq("Delta validated").sum()
        k[5].metric("Delta criteria met", f"{delta_ok}/{(v['delta_validation_status'] != 'Baseline / not applicable').sum()}")
    runtime = tables["genco_summary"]["runtime_seconds"]
    if runtime.gt(0).any() and "ac_summary" in tables:
        st.caption(f"Median GENCO inference {runtime[runtime > 0].median() * 1000:.0f} ms per scenario versus median pandapower Newton-Raphson {tables['ac_summary']['runtime_seconds'].median() * 1000:.0f} ms (same CPU, including GENCO graph build).")
    m = base_map(layers)
    add_context_layers(m, layers, show_substations=True)
    add_network(m, layers, lambda r: {"color": "#555", "weight": 1.5, "opacity": 0.8})
    for row in layers["buses"].itertuples():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=2.5, color="#222", fill=True, fill_opacity=1, tooltip=f"Bus {row.bus_id} ({row.bus_role})").add_to(m)
    add_candidates(m, layers, final, "final_score", "YlGn", 0.4, 0.9, candidate_popup, "Candidates (balanced final score)")
    legend(m, "Overview", [("Protected area (real)", "#1b7837"), ("Burned 2015-2024 (real)", "#6a1b9a"), ("IEEE-118 branch (schematic)", "#555"), ("Candidate, higher final score", "#238443"), ("Candidate, lower final score", "#f7fcb9")])
    render(m, 600)
    hero = ROOT / "docs" / "figures"
    c1, c2 = st.columns(2)
    if (hero / "capacity_sweep.gif").exists():
        c1.image(str(hero / "capacity_sweep.gif"), caption="What GENCO sees as a plant grows from 0 to 200 MW at a strong and a weak site (one frame = one 44 ms inference)")
    if (hero / "terrain_3d_candidates.png").exists():
        c2.image(str(hero / "terrain_3d_candidates.png"), caption="The top three sites in 3D: real 30 m terrain and land cover, the candidate cell, its connector and bus, nearby lines by consequence class, protected land and real substations (context). Path-traced with forge3d.")
        with c2.expander("Whole study area in 3D"):
            st.image(str(hero / "terrain_3d_oblique.png"))
    top = final.sort_values("rank").head(5)[["rank", "candidate_id", "linked_bus_id", "spatial_score", "grid_score", "resilience_score", "final_score", "classification", "top10_in_all_profiles"]]
    st.markdown("**Top five under the balanced profile (0.55 grid / 0.25 resilience / 0.20 spatial)**")
    st.dataframe(top.round(3), hide_index=True, width="stretch")
    st.info(DISCLAIMER, icon="ℹ️")


def tab_spatial(layers, tables) -> None:
    st.markdown("**Map 1 - Spatial opportunity.** Where is a project geographically plausible before any grid analysis? Score = 0.40 solar resource + 0.35 land cover + 0.25 slope - 0.50 protected-area overlap; hard exclusions for protected areas (>25 %), water (>10 %), built-up land (>20 %) and settlement buffers.")
    hexes = layers["hex_cells"]
    show = st.radio("Colour cells by", ["spatial_opportunity_score", "solar_resource_score", "land_suitability_score", "slope_constructability_score", "wildfire_exposure_score"], horizontal=True, format_func=lambda s: s.replace("_", " "))
    m = base_map(layers)
    cmap = {"wildfire_exposure_score": "Reds"}.get(show, "YlOrBr")
    vmax = float(hexes[show].quantile(0.98)) if show == "wildfire_exposure_score" else 1.0
    fields = ["hex_id", "spatial_opportunity_score", "solar_resource_score", "pv_yield_kwh_per_kwp", "land_suitability_score", "mean_slope_deg", "protected_fraction", "wildfire_exposure_score", "exclusion_reason", "nearest_bus_id", "nearest_bus_distance_m"]
    style = lambda feature: {  # noqa: E731
        "weight": 0, "fillOpacity": 0.75,
        "fillColor": "#9e9e9e" if feature["properties"]["spatially_excluded"] else colour(feature["properties"][show], cmap, 0, vmax),
    }
    folium.GeoJson(hexes[fields + ["spatially_excluded", "geometry"]].round(3), name="Hex cells", style_function=style,
                   tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[f.replace("_", " ") for f in fields], localize=True)).add_to(m)
    add_context_layers(m, layers, show_burned=show == "wildfire_exposure_score")
    folium.GeoJson(layers["settlement_buffers"], name="Settlement buffers", style_function=lambda _: {"color": "#444", "weight": 0.6, "fill": True, "fillOpacity": 0.08}, show=False).add_to(m)
    add_candidates(m, layers, tables["spatial"], "spatial_score", "YlOrBr", 0, 1, None, "Candidate cells", fixed_colour=lambda r: "#ffffff00")
    legend(m, show.replace("_", " "), [("Excluded (grey)", "#9e9e9e"), ("Low", colour(0.1, cmap, 0, 1)), ("Medium", colour(0.5, cmap, 0, 1)), ("High", colour(0.9, cmap, 0, 1)), ("Protected area", "#1b7837")])
    render(m)
    c1, c2 = st.columns([1, 1])
    with c1:
        counts = hexes["exclusion_reason"].replace("", "not excluded").value_counts().reset_index()
        counts.columns = ["reason", "cells"]
        st.plotly_chart(px.bar(counts, x="cells", y="reason", orientation="h", title="Why cells are excluded (real constraints)", template="plotly_white", height=320), width="stretch")
    with c2:
        st.plotly_chart(px.histogram(hexes.loc[~hexes["spatially_excluded"]], x="spatial_opportunity_score", nbins=40, title="Spatial opportunity of non-excluded cells", template="plotly_white", height=320), width="stretch")
    st.markdown("**Candidate selection rule.** Pool = non-excluded cells within 10 km of a bus and above the 60th percentile of spatial opportunity; 30 sites picked by farthest-point sampling with one site per connection bus; each linked to its nearest bus.")
    st.dataframe(tables["spatial"].drop(columns=[c for c in ["lon", "lat"] if c in tables["spatial"].columns]).round(3), hide_index=True, width="stretch")


def tab_stress(layers, tables) -> None:
    st.markdown("**Map 2 - Grid stress.** GENCO-predicted change caused by one project under one operating scenario, relative to the same condition's baseline (delta mode).")
    manifest = tables["manifest"].loc[~tables["manifest"]["is_baseline"]]
    c = st.columns(4)
    candidate = c[0].selectbox("Candidate", sorted(manifest["candidate_id"].unique()))
    capacity = c[1].selectbox("Capacity (MW)", sorted(manifest["capacity_mw"].unique()), index=1)
    condition = c[2].selectbox("Operating condition", list(manifest["operating_condition"].unique()), index=1)
    contingency = c[3].selectbox("Contingency", list(manifest["contingency_id"].unique()), format_func=lambda s: s if s == "n0" else f"{s} (outage of {tables['contingencies'].set_index('contingency_id').loc[s, 'branch_id']})")
    row = manifest.loc[manifest["candidate_id"].eq(candidate) & manifest["capacity_mw"].eq(capacity) & manifest["operating_condition"].eq(condition) & manifest["contingency_id"].eq(contingency)].iloc[0]
    scenario_id = row["scenario_id"]
    bus, branch = load_scenario(scenario_id)
    metric = tables["metrics"].set_index("scenario_id").loc[scenario_id]
    k = st.columns(5)
    k[0].metric("Injection", f"{row['active_power_mw']:.0f} MW at bus {int(row['linked_bus_id'])}")
    k[1].metric("Max |ΔV|", f"{metric['max_abs_voltage_delta_pu']:.4f} pu")
    k[2].metric("Max Δloading", f"{metric['max_loading_delta']:+.3f}", help=f"on {metric['worst_branch_id']}")
    k[3].metric("Branches Δloading > 0.10", int(metric["branches_loading_increase_gt_0_10"]))
    status = "GENCO screened (not in AC subset)"
    if "validation" in tables and scenario_id in set(tables["validation"]["scenario_id"]):
        status = tables["validation"].set_index("scenario_id").loc[scenario_id, "screening_status"]
    k[4].metric("Validation status", status)
    bus_view = layers["buses"].merge(bus[["bus_id", "voltage_pu", "voltage_delta_pu"]], on="bus_id")
    branch_view = layers["branches"].drop(columns=["in_service"]).merge(branch[["branch_id", "loading_ratio", "loading_delta", "in_service", "p_flow_mw"]], on="branch_id")
    layers_view = dict(layers, buses=bus_view, branches=branch_view)
    vmax_l = max(0.1, float(branch_view["loading_delta"].abs().quantile(0.99)))
    m = base_map(layers_view)
    add_context_layers(m, layers, show_settlements=False)

    def branch_style(r):
        if not r.in_service:
            return {"color": "#000", "weight": 4, "dash_array": "6 6", "opacity": 1}
        return {"color": colour(r.loading_delta, "RdBu_r", -vmax_l, vmax_l), "weight": 1.5 + 6 * min(float(r.loading_ratio), 1.2), "opacity": 0.95}

    def bus_style(r):
        return {"radius": 5, "color": "#333", "weight": 0.8, "fill": True, "fill_opacity": 1, "fill_color": colour(r.voltage_delta_pu, "PuOr_r", -0.02, 0.02)}

    add_network(m, layers_view, branch_style, bus_style,
                lambda r: f"{r.branch_id} {int(r.from_bus)}-{int(r.to_bus)}<br>loading {r.loading_ratio:.2f} (Δ {r.loading_delta:+.3f})<br>P {r.p_flow_mw:.0f} MW" + (" <b>OUT OF SERVICE</b>" if not r.in_service else ""),
                lambda r: f"Bus {r.bus_id}: V {r.voltage_pu:.4f} pu (Δ {r.voltage_delta_pu:+.4f})")
    site = layers["candidate_sites"].loc[layers["candidate_sites"]["candidate_id"].eq(candidate)]
    folium.GeoJson(site, style_function=lambda _: {"color": "#111", "weight": 3, "fillColor": "#ffd54f", "fillOpacity": 0.9}, tooltip=f"{candidate}: {row['active_power_mw']:.0f} MW").add_to(m)
    conn = layers["candidate_connections"].loc[layers["candidate_connections"]["candidate_id"].eq(candidate)]
    folium.GeoJson(conn, style_function=lambda _: {"color": "#111", "weight": 2, "dashArray": "5 5"}).add_to(m)
    legend(m, "Δ loading / ΔV", [("Loading decrease", colour(-vmax_l, "RdBu_r", -vmax_l, vmax_l)), ("No change", colour(0, "RdBu_r", -1, 1)), ("Loading increase", colour(vmax_l, "RdBu_r", -vmax_l, vmax_l)), ("Outaged branch", "#000"), ("Bus ΔV < 0", colour(-0.02, "PuOr_r", -0.02, 0.02)), ("Bus ΔV > 0", colour(0.02, "PuOr_r", -0.02, 0.02))])
    render(m)
    st.caption("Line width = absolute loading ratio; colour = change versus the condition baseline. Buses coloured by voltage change. Screening uses these changes so every candidate is judged against the same baseline.")
    top = branch_view.drop(columns="geometry").loc[branch_view["in_service"]].nlargest(8, "loading_delta")[["branch_id", "from_bus", "to_bus", "loading_ratio", "loading_delta", "wildfire_exposure_score"]]
    st.dataframe(top.round(3), hide_index=True, width="stretch")


def tab_robustness(layers, tables) -> None:
    st.markdown("**Map 3 - Candidate grid robustness.** Each candidate is summarised over its 12 N-0 cases (3 capacities x 4 conditions). Grid score = 1 - mean(0.5 voltage severity + 0.5 loading severity), severities scaled to the 0.05 pu and +0.20 loading-change limits.")
    grid = tables["grid"].merge(tables["spatial"][["candidate_id", "spatially_excluded", "linked_bus_id", "connection_distance_m"]], on="candidate_id")
    grid["class"] = [robustness_class(s)[0] if not e else "Spatially excluded" for s, e in zip(grid["grid_score"], grid["spatially_excluded"])]
    m = base_map(layers)
    add_context_layers(m, layers, show_burned=False)
    add_network(m, layers, lambda r: {"color": "#777", "weight": 1.2, "opacity": 0.7})
    add_candidates(m, layers, grid, "grid_score", "RdYlGn", 0.3, 0.9, lambda r: f"<b>{r.candidate_id}</b> bus {int(r.linked_bus_id)}<br>Grid score {r.grid_score:.2f} ({r._asdict().get('class', '')})<br>max |ΔV| {r.max_abs_voltage_delta_pu:.3f} pu, max Δloading {r.max_loading_delta:+.2f}<br>Loading warnings in {r.loading_warning_frequency:.0%} of N-0 cases<br>Worst: {r.worst_n0_scenario} on {r.worst_n0_branch}",
                   "Candidates (grid score)", fixed_colour=lambda r: "#9e9e9e" if r.spatially_excluded else robustness_class(r.grid_score)[1])
    legend(m, "Grid robustness", [(label, c) for _, label, c in ROBUSTNESS_CLASSES] + [("Spatially excluded", "#9e9e9e")])
    render(m)
    n0 = tables["metrics"].loc[~tables["metrics"]["is_baseline"] & tables["metrics"]["contingency_id"].eq("n0")].copy()
    n0["case"] = n0["capacity_mw"].astype(int).astype(str) + " MW / " + n0["operating_condition"]
    heat = n0.pivot(index="candidate_id", columns="case", values="max_loading_delta")
    st.plotly_chart(px.imshow(heat, color_continuous_scale="YlOrRd", zmin=0, zmax=0.4, aspect="auto", title="Max loading increase per candidate and N-0 case (GENCO delta)", height=620, template="plotly_white"), width="stretch")
    st.dataframe(grid.drop(columns=["spatially_excluded"]).sort_values("grid_score", ascending=False).round(3), hide_index=True, width="stretch")


def tab_wildfire(layers, tables) -> None:
    st.markdown("**Map 4 - Wildfire exposure x electrical consequence.** Exposure = 0.6 x corridor fraction burned 2015-2024 (MODIS) + 0.4 x corridor tree/shrub fuel fraction (WorldCover), 1 km corridor. Consequence = 95th-percentile loading increase a branch receives across the 1,800 N-1 cases. Terciles within the study area; the five screened outages were chosen as the most exposed connectivity-preserving lines.")
    hazard = tables["hazard"].copy()
    if "outages_all" in tables:
        allo = tables["outages_all"][["branch_id", "outage_max_loading_increase_elsewhere", "outage_lines_over_0_10"]].rename(columns={"outage_max_loading_increase_elsewhere": "outage_all_max_increase", "outage_lines_over_0_10": "outage_all_lines_over_0_10"})
        hazard = hazard.merge(allo, on="branch_id", how="left")
        st.caption("Every connectivity-preserving line also has a GENCO outage-only consequence (max loading increase elsewhere when it is lost); the table below shows both views.")
    view = layers["branches"].merge(hazard[["branch_id", "priority_score", "bivariate_class", "exposure_tercile", "consequence_tercile", "electrical_consequence_score", "p95_loading_increase", "max_loading_increase", "is_screened_outage", "outage_consequence_score"] + [c for c in ["outage_all_max_increase"] if c in hazard.columns]], on="branch_id")
    m = base_map(dict(layers, branches=view))
    add_context_layers(m, layers, show_settlements=False)
    add_network(m, dict(layers, branches=view), lambda r: {"color": BIVARIATE[(r.exposure_tercile, r.consequence_tercile)], "weight": 6 if r.is_screened_outage else 3.5, "opacity": 0.95},
                None, lambda r: f"{r.branch_id} {int(r.from_bus)}-{int(r.to_bus)}<br>{r.bivariate_class}<br>exposure {r.wildfire_exposure_score:.2f}, consequence {r.electrical_consequence_score:.2f}<br>p95 loading increase received +{r.p95_loading_increase:.3f}" + (f"<br><b>Screened N-1 outage</b>: max increase elsewhere {r.max_loading_increase:.2f}" if r.is_screened_outage else ""))
    for row in layers["buses"].itertuples():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=2, color="#222", fill=True, fill_opacity=1).add_to(m)
    legend(m, "exposure / consequence", [(f"{e} / {c}", col) for (e, c), col in BIVARIATE.items()])
    render(m)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(px.scatter(hazard, x="wildfire_exposure_score", y="electrical_consequence_score", color="bivariate_class", color_discrete_map={f"{e} exposure / {c} consequence": col for (e, c), col in BIVARIATE.items()}, hover_name="branch_id", symbol="is_screened_outage", title="Branches: exposure vs consequence", template="plotly_white", height=420), width="stretch")
    with c2:
        st.markdown("**Priority branches (exposure x consequence)**")
        cols = ["branch_id", "from_bus", "to_bus", "wildfire_exposure_score", "corridor_burned_fraction", "corridor_fuel_fraction", "electrical_consequence_score", "priority_score", "bivariate_class", "is_screened_outage"] + [c for c in ["outage_all_max_increase", "outage_all_lines_over_0_10"] if c in hazard.columns]
        st.dataframe(hazard.sort_values("priority_score", ascending=False).head(15)[cols].round(3), hide_index=True, width="stretch", height=420)
    st.markdown("**Screened N-1 outages and their consequence**")
    st.dataframe(tables["contingencies"].merge(hazard[["branch_id", "outage_consequence_score", "outage_max_loading_increase_elsewhere"]], on="branch_id", how="left").round(3), hide_index=True, width="stretch")


def tab_final(layers, tables) -> None:
    st.markdown("**Map 5 - Grid-aware suitability.** Final = w_grid x grid + w_resilience x resilience + w_spatial x spatial. Weights are preferences, not truth; move them and watch the ranking.")
    final_all = tables["final"]
    base = final_all.loc[final_all["weight_profile"].eq("balanced")].drop(columns=["weight_profile", "final_score", "rank", "classification"])
    c = st.columns([1, 1, 1, 2])
    preset = c[3].selectbox("Preset profile", ["balanced", "grid_first", "resilience_first", "custom"])
    presets = {"grid_first": (0.70, 0.20, 0.10), "balanced": (0.55, 0.25, 0.20), "resilience_first": (0.40, 0.40, 0.20)}
    wg, wr, ws = presets.get(preset, (0.55, 0.25, 0.20))
    wg = c[0].slider("Grid weight", 0.0, 1.0, wg, 0.05, disabled=preset != "custom")
    wr = c[1].slider("Resilience weight", 0.0, 1.0, wr, 0.05, disabled=preset != "custom")
    ws = c[2].slider("Spatial weight", 0.0, 1.0, ws, 0.05, disabled=preset != "custom")
    total = max(wg + wr + ws, 1e-9)
    wg, wr, ws = wg / total, wr / total, ws / total
    table = base.copy()
    table["final_score"] = wg * table["grid_score"] + wr * table["resilience_score"] + ws * table["spatial_score"]
    table.loc[table["spatially_excluded"], "final_score"] = np.nan
    table["rank"] = table["final_score"].rank(ascending=False, method="min").astype("Int64")
    table["classification"] = np.select([table["spatially_excluded"], table["final_score"].ge(0.75), table["final_score"].ge(0.50)], ["Spatially excluded", "Priority for deeper study", "Conditionally suitable"], default="Constrained")
    st.caption(f"Effective weights: grid {wg:.2f}, resilience {wr:.2f}, spatial {ws:.2f}")
    m = base_map(layers)
    add_context_layers(m, layers)
    add_network(m, layers, lambda r: {"color": "#777", "weight": 1.2, "opacity": 0.7})
    extra_fn = lambda r: f"<br>Rank {int(r.rank) if pd.notna(r.rank) else '-'} of {int(table['rank'].max())}"  # noqa: E731
    add_candidates(m, layers, table, "final_score", "YlGn", 0.4, 0.9, lambda r: candidate_popup(r, extra_fn(r)), "Candidates (final score)",
                   fixed_colour=lambda r: "#9e9e9e" if r.spatially_excluded else colour(r.final_score, "YlGn", 0.4, 0.9))
    legend(m, "Final score", [("Priority (>= 0.75)", colour(0.85, "YlGn", 0.4, 0.9)), ("Conditionally suitable", colour(0.62, "YlGn", 0.4, 0.9)), ("Constrained (< 0.50)", colour(0.45, "YlGn", 0.4, 0.9)), ("Spatially excluded", "#9e9e9e")])
    render(m)
    ranked = table.sort_values("rank")[["rank", "candidate_id", "linked_bus_id", "spatial_score", "grid_score", "resilience_score", "final_score", "classification", "best_rank", "worst_rank", "top10_in_all_profiles", "reasons"]]
    st.dataframe(ranked.round(3), hide_index=True, width="stretch", height=420)
    st.markdown("**Rank stability across the three documented profiles**")
    bump = final_all.pivot(index="candidate_id", columns="weight_profile", values="rank")[["grid_first", "balanced", "resilience_first"]].reset_index().melt(id_vars="candidate_id", var_name="profile", value_name="rank")
    fig = px.line(bump.dropna(), x="profile", y="rank", color="candidate_id", markers=True, template="plotly_white", height=520, title="Candidate rank by weight profile (lower is better)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Why does each candidate rank where it does?** Component contributions under the current weights.")
    contrib = table.loc[~table["spatially_excluded"]].sort_values("final_score", ascending=False)
    fig = go.Figure()
    for name, w, col in (("grid", wg, "#1f78b4"), ("resilience", wr, "#e31a1c"), ("spatial", ws, "#ff7f00")):
        fig.add_bar(name=f"{name} x {w:.2f}", x=contrib["candidate_id"], y=w * contrib[f"{name}_score"], marker_color=col)
    fig.update_layout(barmode="stack", template="plotly_white", height=380, yaxis_title="weighted contribution")
    st.plotly_chart(fig, width="stretch")


def tab_validation(layers, tables, meta) -> None:
    st.markdown("**Electrical validation.** pandapower Newton-Raphson AC power flow on the same modified PFDelta snapshot GENCO received, with branch flows read back in PFDelta order. Absolute agreement says how close GENCO's operating point is; change agreement says how well it predicts what a project does. Both are reported.")
    if "validation" not in tables:
        st.warning("Run scripts/validate_case_study_subset.py to populate this tab.")
        return
    v = tables["validation"].copy()
    if "note" not in v.columns:
        v["note"] = ""
    k = st.columns(4)
    k[0].metric("AC cases attempted", len(v))
    k[1].metric("AC converged", int(v["validation_status"].ne("Infeasible").sum()))
    k[2].metric("Delta validated", int(v["delta_validation_status"].eq("Delta validated").sum()))
    k[3].metric("Median |ΔV| error", f"{v['voltage_delta_mae_pu'].median():.5f} pu")
    st.plotly_chart(px.histogram(v.dropna(subset=["branch_active_power_delta_mae_mw"]), x="branch_active_power_delta_mae_mw", color="delta_validation_status", nbins=25, title="Branch active-power delta MAE (GENCO vs AC), MW", template="plotly_white", height=320), width="stretch")
    st.dataframe(v[["scenario_id", "validation_reason", "validation_status", "delta_validation_status", "screening_status", "voltage_mae_pu", "branch_active_power_mae_mw", "voltage_delta_mae_pu", "voltage_delta_max_abs_error_pu", "branch_active_power_delta_mae_mw", "branch_active_power_delta_max_abs_error_mw", "note"]].round(4), hide_index=True, width="stretch", height=420)
    choices = v.loc[v["validation_status"].ne("Infeasible") & v["candidate_id"].notna(), "scenario_id"].tolist()
    if choices:
        scenario_id = st.selectbox("Inspect one validated scenario", choices)
        g_bus, g_branch = load_scenario(scenario_id)
        a_bus, a_branch = load_ac_scenario(scenario_id)
        c1, c2 = st.columns(2)
        with c1:
            d = g_bus[["bus_id", "voltage_delta_pu"]].merge(a_bus[["bus_id", "voltage_delta_pu"]], on="bus_id", suffixes=("_genco", "_ac"))
            fig = px.scatter(d, x="voltage_delta_pu_ac", y="voltage_delta_pu_genco", hover_name="bus_id", title="Bus ΔV: GENCO vs AC", template="plotly_white", height=380)
            lim = float(d[["voltage_delta_pu_ac", "voltage_delta_pu_genco"]].abs().max().max()) or 0.01
            fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"dash": "dash", "color": "#888"})
            st.plotly_chart(fig, width="stretch")
        with c2:
            d = g_branch[["branch_id", "loading_delta"]].merge(a_branch[["branch_id", "loading_delta"]], on="branch_id", suffixes=("_genco", "_ac"))
            fig = px.scatter(d, x="loading_delta_ac", y="loading_delta_genco", hover_name="branch_id", title="Branch Δloading: GENCO vs AC", template="plotly_white", height=380)
            lim = float(d[["loading_delta_ac", "loading_delta_genco"]].abs().max().max()) or 0.01
            fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"dash": "dash", "color": "#888"})
            st.plotly_chart(fig, width="stretch")
    st.markdown("**Spatial validation and provenance**")
    st.write({"CRS": meta["crs"], "Study area (WGS84)": meta["study_area_wgs84"], "Network geometry": "schematic embedding", "Geodata": "real open data", "Layer counts": {k: v["feature_count"] for k, v in meta["layers"].items()}})
    sources = pd.DataFrame(meta.get("geodata_sources", []))
    if len(sources):
        st.dataframe(sources[[c for c in ["layer", "title", "native_resolution", "derived", "license", "accessed_utc"] if c in sources.columns]], hide_index=True, width="stretch")
    st.markdown("**Model and limitations**")
    st.markdown(
        "- GENCO checkpoint `gridfm/genco-pfdelta` Task 1.3 seed 1 on the released PFDelta IEEE-118 split; baseline snapshot = feasible test scenario 14558.\n"
        "- Interventions are fixed-P injections (reduced Pd) at the linked bus; conditions scale all loads; N-1 masks one branch.\n"
        "- All screening uses scenario-minus-baseline changes so every candidate is judged against the same operating point; absolute and change agreement with AC are both shown above.\n"
        "- Several AC N-1 cases do not converge; those are kept visible as high-consequence outcomes rather than dropped.\n"
        "- Exposure is a 10-year MODIS burned-area record plus WorldCover fuel; it is not a fire-weather or ignition model.\n"
        f"- {DISCLAIMER}"
    )


def tab_genco(layers, tables) -> None:
    st.markdown("**How GENCO behaved on this study.** Diagnostics that explain when the neural solver can be trusted as a screen, measured against pandapower on the validated subset and across all 2,184 scenarios.")
    if "delta_agreement" not in tables:
        st.warning("Run scripts/analyze_genco_behaviour.py to populate this tab.")
        return
    agreement = tables["delta_agreement"].iloc[0]
    runtime = tables["genco_summary"]["runtime_seconds"]
    k = st.columns(5)
    k[0].metric("GENCO per scenario", f"{runtime[runtime > 0].median() * 1000:.0f} ms", help="median, CPU, including graph build")
    if "ac_summary" in tables:
        k[1].metric("pandapower per case", f"{tables['ac_summary']['runtime_seconds'].median() * 1000:.0f} ms", help="median Newton-Raphson solve")
    k[2].metric("Loading-change agreement R²", f"{agreement['r2']:.2f}", help=f"{int(agreement['pairs']):,} branch-scenario pairs, slope {agreement['slope']:.2f}")
    k[3].metric("Within ±0.05 of AC", f"{agreement['share_within_0_05']:.0%}", help="share of branch loading changes")
    k[4].metric("Sign agreement, |AC change| > 0.05", f"{agreement['sign_agreement_when_ac_change_gt_0_05']:.0%}")
    c1, c2 = st.columns(2)
    with c1:
        hops = tables["error_by_hops"].copy()
        hops["case"] = np.where(hops["is_outage_case"], "N-1 (distance to injection bus or outaged line)", "N-0 (distance to injection bus)")
        fig = px.line(hops, x="hops_from_perturbation", y="mean_abs_error", color="case", markers=True, template="plotly_white", height=380,
                      title="Where GENCO's loading-change error sits", labels={"hops_from_perturbation": "network hops from the perturbation", "mean_abs_error": "mean |Δloading error|"})
        fig.add_scatter(x=hops.loc[hops["is_outage_case"], "hops_from_perturbation"], y=hops.loc[hops["is_outage_case"], "p90_abs_error"], mode="lines", line={"dash": "dot", "color": "#888"}, name="N-1 90th percentile")
        st.plotly_chart(fig, width="stretch")
        st.caption("Errors are small everywhere and largest within one or two hops of the injection bus or the removed line, so AC confirmation can concentrate on that neighbourhood.")
    with c2:
        pairs = tables["delta_pairs"]
        sample = pairs.sample(min(6000, len(pairs)), random_state=0)
        sample["case"] = np.where(sample["is_outage_case"], "N-1", "N-0")
        fig = px.scatter(sample, x="loading_delta_ac", y="loading_delta_genco", color="case", opacity=0.45, template="plotly_white", height=380,
                         title="Branch loading change: GENCO vs AC (validated scenarios)", labels={"loading_delta_ac": "AC Δloading", "loading_delta_genco": "GENCO Δloading"})
        lim = float(np.nanpercentile(np.abs(pairs[["loading_delta_ac", "loading_delta_genco"]].to_numpy()), 99.5))
        fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"dash": "dash", "color": "#666"})
        fig.update_xaxes(range=[-lim, lim])
        fig.update_yaxes(range=[-lim, lim])
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        rv = tables["residual_vs_validation"]
        rho = rv[["physics_residual", "branch_active_power_delta_mae_mw"]].corr(method="spearman").iloc[0, 1]
        fig = px.scatter(rv, x="physics_residual", y="branch_active_power_delta_mae_mw", color="delta_validation_status", symbol="contingency_id", hover_name="scenario_id", template="plotly_white", height=380,
                         title=f"GENCO's own power-balance residual vs error against AC (Spearman {rho:.2f})", labels={"physics_residual": "mean bus power-balance residual (MW)", "branch_active_power_delta_mae_mw": "branch ΔP MAE vs AC (MW)"})
        st.plotly_chart(fig, width="stretch")
        st.caption("The residual exists for every scenario without running AC. A positive correlation means it can rank scenarios for confirmation.")
    with c4:
        rc = tables["residual_by_contingency"].pivot(index="contingency_id", columns="capacity_mw", values="quality_or_residual_metric")
        fig = px.imshow(rc, color_continuous_scale="YlOrRd", aspect="auto", template="plotly_white", height=380, title="Median power-balance residual by contingency and capacity (all 2,184 scenarios)", labels={"color": "MW"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Residuals rise with injected capacity and under outages: the model is asked to extrapolate further from the released snapshot distribution.")
    if "load_sweep" in tables:
        st.markdown("**Operating envelope.** One site at 100 MW, load level swept from 0.5 to 1.5 of the released snapshot, AC at every step.")
        sweep = tables["load_sweep"]
        c5, c6 = st.columns(2)
        with c5:
            fig = go.Figure()
            ok = sweep[sweep["ac_converged"]]
            fig.add_scatter(x=ok["load_multiplier"], y=ok["branch_p_mae_mw"], mode="lines+markers", name="branch P error vs AC (MW)", line={"color": "#1f78b4"})
            fig.add_scatter(x=sweep["load_multiplier"], y=sweep["genco_residual_case"], mode="lines+markers", name="GENCO residual (MW per bus)", line={"color": "#e31a1c"}, yaxis="y2")
            failed = sweep[~sweep["ac_converged"]]
            if len(failed):
                fig.add_vrect(x0=float(failed["load_multiplier"].min()) - 0.025, x1=float(sweep["load_multiplier"].max()) + 0.025, fillcolor="#999", opacity=0.25, line_width=0, annotation_text="AC does not converge")
            fig.update_layout(template="plotly_white", height=380, title="Error and self-reported residual vs load level", xaxis_title="load multiplier", yaxis={"title": "MW vs AC"}, yaxis2={"title": "residual", "overlaying": "y", "side": "right"}, legend={"orientation": "h", "y": -0.25})
            st.plotly_chart(fig, width="stretch")
        with c6:
            fig = px.line(ok, x="load_multiplier", y=["genco_max_loading_delta", "ac_max_loading_delta"], markers=True, template="plotly_white", height=380, title="Screening quantity along the sweep: max loading change, GENCO vs AC", labels={"value": "max loading change", "variable": ""})
            st.plotly_chart(fig, width="stretch")
    if "seed_ranks" in tables:
        st.markdown("**Seed ensemble.** The same scenarios through all three released checkpoints.")
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(px.histogram(tables["seed_spread"], x="loading_delta_range", nbins=30, template="plotly_white", height=360, title="Range of max loading change across seeds, per scenario", labels={"loading_delta_range": "max minus min across seeds"}), width="stretch")
        with c8:
            ranks = tables["seed_ranks"].sort_values("best_rank").head(15)
            fig = go.Figure(go.Bar(y=ranks["candidate_id"], x=ranks["worst_rank"] - ranks["best_rank"] + 0.6, base=ranks["best_rank"] - 0.3, orientation="h", marker_color=np.where(ranks["top5_in_all_seeds"], "#1a9850", "#fdae61")))
            fig.update_layout(template="plotly_white", height=360, title="Grid-score rank across seeds (green = top 5 in every seed)", xaxis_title="rank", yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, width="stretch")
        st.dataframe(tables["seed_ranks"].sort_values("best_rank").round(3), hide_index=True, width="stretch", height=300)
    st.markdown("**What this means for the workflow**")
    st.markdown(
        "- Build screening on scenario-minus-baseline changes with explicit baselines; GENCO reproduces those changes almost one-for-one against AC.\n"
        "- Send to AC the cases GENCO itself flags: high residual, 200 MW cases, outage cases near the removed line.\n"
        "- Keep the bookkeeping between solvers exact: parallel circuits resolved by impedance, AC flows read back in PFDelta order. Both bugs found during this study were bookkeeping, not model error, and both are now regression-tested."
    )


# --------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="GeoGridLens | Alentejo case study", layout="wide", page_icon="🗺️")
    st.title("GeoGridLens - grid-aware siting with GENCO and real geodata")
    st.caption("Alentejo, Portugal study area (real GIS layers) x IEEE-118 benchmark (schematic embedding) x GENCO neural power flow (2,160 scenarios) x pandapower AC validation")
    required = [INDICATORS / "candidate_final_scores.parquet", INDICATORS / "scenario_metrics.parquet", SPATIAL / "hex_cells.parquet"]
    if any(not p.exists() for p in required):
        st.error("Case-study artifacts are missing. Run scripts/run_case_study_pipeline.ps1 (or the individual scripts in README) first.")
        return
    layers, tables, meta = load_layers(), load_tables(), load_metadata()
    tabs = st.tabs(["Overview", "1 Spatial opportunity", "2 Grid stress", "3 Grid robustness", "4 Wildfire x consequence", "5 Final suitability", "How GENCO behaved", "Validation & provenance"])
    with tabs[0]:
        tab_overview(layers, tables, meta)
    with tabs[1]:
        tab_spatial(layers, tables)
    with tabs[2]:
        tab_stress(layers, tables)
    with tabs[3]:
        tab_robustness(layers, tables)
    with tabs[4]:
        tab_wildfire(layers, tables)
    with tabs[5]:
        tab_final(layers, tables)
    with tabs[6]:
        tab_genco(layers, tables)
    with tabs[7]:
        tab_validation(layers, tables, meta)
    st.caption(f"GeoGridLens by {AUTHOR} ({ROLE}), {TEAM}. Contact {EMAIL}. Read-only view over pipeline outputs; geometry of the IEEE-118 network is schematic.")


if __name__ == "__main__":
    main()
