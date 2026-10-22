"""Build the geo-embedded IEEE-118 layers and real-data candidate cells.

Inputs are the cached real layers from :mod:`case_study.geodata`.  Outputs are
GeoParquet + GeoJSON under ``data/spatial/case_study`` with one metadata file
that states, per layer, whether it is real geography or a schematic embedding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pandapower.networks as pn
import rasterio
import rasterio.features
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, box, shape

from case_study.geodata import WORLDCOVER_CLASSES, StudyArea, study_area_from_config
from case_study.manifest import load_config
from scipy.stats import rankdata
from spatial.schematic_embedding import network_graph, build_branch_layer, build_bus_layer


# ----------------------------------------------------------------- utilities


def _read_raster(path: Path) -> tuple[np.ndarray, Any]:
    with rasterio.open(path) as source:
        return source.read(1), source.transform


def _hex_polygon(x: float, y: float, radius: float) -> Polygon:
    angles = np.deg2rad(np.arange(30, 390, 60))
    return Polygon([(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles])


def _hex_grid(bounds: tuple[float, float, float, float], radius: float) -> gpd.GeoDataFrame:
    left, bottom, right, top = bounds
    dx = np.sqrt(3) * radius
    dy = 1.5 * radius
    rows = []
    row = 0
    y = bottom + radius
    while y <= top - radius:
        x = left + radius + (dx / 2 if row % 2 else 0)
        while x <= right - radius:
            rows.append({"hex_id": f"H{len(rows):05d}", "geometry": _hex_polygon(x, y, radius), "cx": x, "cy": y})
            x += dx
        y += dy
        row += 1
    return gpd.GeoDataFrame(rows, geometry="geometry")


def _zonal_mean(zone_ids: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    valid = zone_ids > 0
    sums = np.bincount(zone_ids[valid], weights=values[valid], minlength=count + 1)[1:]
    n = np.bincount(zone_ids[valid], minlength=count + 1)[1:]
    return np.divide(sums, n, out=np.zeros(count), where=n > 0)


def _mask_fraction(mask: np.ndarray, buffered: Any, transform: Any, shape_: tuple[int, int]) -> float:
    corridor = rasterio.features.geometry_mask([buffered], out_shape=shape_, transform=transform, invert=True)
    pixels = int(corridor.sum())
    return float(mask[corridor].mean()) if pixels else 0.0


def _exposure_class(values: pd.Series) -> np.ndarray:
    """Terciles within the study area, because absolute exposure is region-specific."""

    low, high = values.quantile([1 / 3, 2 / 3])
    return np.where(values < low, "low", np.where(values < high, "medium", "high"))


def _layout_coordinates(net: Any, rank_blend: float) -> dict[int, np.ndarray]:
    """Deterministic Kamada-Kawai layout, blended with a rank transform.

    The rank blend spreads the benchmark across the whole study box instead of
    leaving the dense IEEE-118 core in one corner.  This is a drawing choice for
    an illustrative embedding, not a claim about where any line runs.
    """

    graph = network_graph(net)
    positions = nx.kamada_kawai_layout(graph)
    indices = list(positions)
    points = np.asarray([positions[i] for i in indices], dtype=float)
    points = (points - points.min(axis=0)) / np.maximum(points.max(axis=0) - points.min(axis=0), 1e-9)
    ranks = np.column_stack([rankdata(points[:, 0]), rankdata(points[:, 1])])
    ranks = (ranks - 1) / max(len(indices) - 1, 1)
    blended = (1 - rank_blend) * points + rank_blend * ranks
    return {index: blended[k] for k, index in enumerate(indices)}


# ------------------------------------------------------------- network layer


def _embed_network(
    config: dict[str, Any], area: StudyArea, landcover: np.ndarray, transform: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Spring-layout IEEE-118 inside the real study box; nudge buses off water."""

    embedding = config["network_embedding"]
    net = pn.case118()
    raw = _layout_coordinates(net, float(embedding.get("rank_blend", 0.5)))
    _, _, _, (left, bottom, right, top) = area.metric_grid()
    margin = float(embedding.get("margin_km", 6)) * 1000
    xs = np.array([raw[i][0] for i in raw])
    ys = np.array([raw[i][1] for i in raw])
    sx = (right - left - 2 * margin) / (xs.max() - xs.min())
    sy = (top - bottom - 2 * margin) / (ys.max() - ys.min())
    scale = min(sx, sy)
    coords = {
        index: (left + margin + (raw[index][0] - xs.min()) * scale, bottom + margin + (raw[index][1] - ys.min()) * scale)
        for index in raw
    }
    if embedding.get("avoid_water", True):
        water = landcover == 80
        if water.any():
            rows, cols = np.where(~water & (landcover > 0))
            land_xy = np.column_stack(rasterio.transform.xy(transform, rows, cols))
            tree = cKDTree(land_xy)
            for index, (x, y) in coords.items():
                r, c = rasterio.transform.rowcol(transform, x, y)
                if 0 <= r < landcover.shape[0] and 0 <= c < landcover.shape[1] and water[r, c]:
                    _, nearest = tree.query([x, y])
                    coords[index] = (float(land_xy[nearest, 0]), float(land_xy[nearest, 1]))
    buses = build_bus_layer(net, coords).set_crs(area.metric_crs, allow_override=True)
    branches = build_branch_layer(net, coords).set_crs(area.metric_crs, allow_override=True)
    buses = buses.rename(columns={"x_schematic": "x_utm", "y_schematic": "y_utm"})
    wgs = buses.geometry.to_crs("EPSG:4326")
    buses["lon"], buses["lat"] = wgs.x, wgs.y
    branches["length_km"] = branches.geometry.length / 1000
    return buses, branches


def _pfdelta_branch_index(branches: gpd.GeoDataFrame, raw_directory: Path, source_scenario_id: int) -> gpd.GeoDataFrame:
    """Attach the released PFDelta branch index used for N-1 masking."""

    from genco.native_pfdelta_scenarios import load_snapshot
    from genco.scenario_inference import raw_branch_mapping

    snapshot = load_snapshot(raw_directory, source_scenario_id)
    mapping = raw_branch_mapping(pn.case118(), snapshot.branch)
    mapping["pfdelta_branch_index"] = snapshot.branch["idx"].to_numpy(dtype=int)
    mapping["rate_a_mva"] = mapping["rate_a"]
    return branches.merge(mapping[["branch_id", "pfdelta_branch_index", "rate_a_mva"]], on="branch_id", how="left", validate="one_to_one")


# ------------------------------------------------------------- hex scoring


def _score_hexes(
    config: dict[str, Any], area: StudyArea, hexes: gpd.GeoDataFrame, rasters: dict[str, np.ndarray], transform: Any,
    protected: gpd.GeoDataFrame, settlements: gpd.GeoDataFrame, pv_points: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, np.ndarray]]:
    rules = config["candidate_sites"]
    shape_ = rasters["landcover"].shape
    zone = rasterio.features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(hexes.geometry)), out_shape=shape_, transform=transform, fill=0, dtype="int32",
    )
    count = len(hexes)
    landcover = rasters["landcover"]
    hexes = hexes.copy()
    hexes["mean_slope_deg"] = _zonal_mean(zone, rasters["slope"].astype(float), count)
    hexes["mean_elevation_m"] = _zonal_mean(zone, rasters["dem"].astype(float), count)
    suitability = {int(k): float(v) for k, v in rules["land_cover_suitability"].items()}
    land_score = np.zeros(count)
    for code, name in WORLDCOVER_CLASSES.items():
        fraction = _zonal_mean(zone, (landcover == code).astype(float), count)
        hexes[f"frac_{name}"] = fraction
        land_score += fraction * suitability.get(code, 0.0)
    hexes["land_suitability_score"] = land_score
    protected_mask = np.zeros(shape_, dtype=bool)
    if len(protected):
        protected_mask = rasterio.features.geometry_mask(protected.geometry, out_shape=shape_, transform=transform, invert=True)
    hexes["protected_fraction"] = _zonal_mean(zone, protected_mask.astype(float), count)
    buffers = rules["exclusion_rules"]["settlement_buffer_m"]
    settlement_mask = np.zeros(shape_, dtype=bool)
    if len(settlements):
        buffered = [geom.buffer(float(buffers.get(place, 500))) for geom, place in zip(settlements.geometry, settlements["place"])]
        settlement_mask = rasterio.features.geometry_mask(buffered, out_shape=shape_, transform=transform, invert=True)
    hexes["settlement_buffer_fraction"] = _zonal_mean(zone, settlement_mask.astype(float), count)
    burned = rasters["burned"] > 0
    hexes["burned_fraction_2015_2024"] = _zonal_mean(zone, burned.astype(float), count)
    hexes["burn_count_mean"] = _zonal_mean(zone, np.clip(rasters["burned"], 0, None).astype(float), count)
    fuel = np.isin(landcover, [int(c) for c in rules["wildfire"]["fuel_classes"]])
    hexes["fuel_fraction"] = _zonal_mean(zone, fuel.astype(float), count)
    weights = rules["wildfire"]["exposure_weights"]
    hexes["wildfire_exposure_score"] = np.clip(
        float(weights["burned"]) * hexes["burned_fraction_2015_2024"] + float(weights["fuel"]) * hexes["fuel_fraction"], 0, 1,
    )
    # Solar resource: PVGIS yield interpolated to cell centres, min-max scaled in the box.
    xy = np.column_stack([pv_points.geometry.x, pv_points.geometry.y])
    yields = pv_points["pv_yield_kwh_per_kwp"].to_numpy()
    centres = np.column_stack([hexes["cx"], hexes["cy"]])
    interpolated = griddata(xy, yields, centres, method="linear")
    nearest = griddata(xy, yields, centres, method="nearest")
    interpolated = np.where(np.isnan(interpolated), nearest, interpolated)
    hexes["pv_yield_kwh_per_kwp"] = interpolated
    span = max(interpolated.max() - interpolated.min(), 1e-9)
    hexes["solar_resource_score"] = (interpolated - interpolated.min()) / span
    full, zero = float(rules["slope_full_score_deg"]), float(rules["slope_zero_score_deg"])
    hexes["slope_constructability_score"] = np.clip((zero - hexes["mean_slope_deg"]) / (zero - full), 0, 1)
    excl = rules["exclusion_rules"]
    reasons = []
    for row in hexes.itertuples():
        why = []
        if row.protected_fraction > float(excl["protected_fraction_gt"]):
            why.append("protected area")
        if row.frac_water > float(excl["water_fraction_gt"]):
            why.append("water body")
        if row.frac_built_up > float(excl["built_fraction_gt"]):
            why.append("built-up land")
        if row.settlement_buffer_fraction > 0.5:
            why.append("settlement buffer")
        reasons.append("; ".join(why))
    hexes["exclusion_reason"] = reasons
    hexes["spatially_excluded"] = hexes["exclusion_reason"].ne("")
    sw = rules["spatial_weights"]
    score = (
        float(sw["resource"]) * hexes["solar_resource_score"]
        + float(sw["land"]) * hexes["land_suitability_score"]
        + float(sw["slope"]) * hexes["slope_constructability_score"]
        - float(sw["exclusion_penalty"]) * hexes["protected_fraction"]
    )
    hexes["spatial_opportunity_score"] = np.where(hexes["spatially_excluded"], 0.0, np.clip(score, 0, 1))
    return hexes, {"burned": burned, "fuel": fuel}


# ------------------------------------------------------- candidate selection


def _attach_nearest_bus(hexes: gpd.GeoDataFrame, buses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tree = cKDTree(np.column_stack([buses.geometry.x, buses.geometry.y]))
    distances, nearest = tree.query(np.column_stack([hexes["cx"], hexes["cy"]]))
    hexes = hexes.copy()
    hexes["nearest_bus_id"] = buses["bus_id"].to_numpy()[nearest]
    hexes["nearest_bus_distance_m"] = distances
    return hexes


def _select_candidates(config: dict[str, Any], hexes: gpd.GeoDataFrame, buses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rules = config["candidate_sites"]
    eligible = hexes.loc[~hexes["spatially_excluded"] & hexes["nearest_bus_distance_m"].le(float(rules["connection_radius_m"]))]
    threshold = float(eligible["spatial_opportunity_score"].quantile(float(rules["pool_score_quantile"])))
    pool = eligible.loc[eligible["spatial_opportunity_score"].ge(threshold)].copy()
    pool_xy = np.column_stack([pool["cx"], pool["cy"]])
    chosen: list[int] = [int(pool["spatial_opportunity_score"].to_numpy().argmax())]
    used_buses = {int(pool.iloc[chosen[0]]["nearest_bus_id"])}
    min_dist = np.linalg.norm(pool_xy - pool_xy[chosen[0]], axis=1)
    target = int(rules["count"])
    while len(chosen) < target:
        order = np.argsort(-min_dist)
        pick = next((int(i) for i in order if i not in chosen and (not rules.get("distinct_buses", True) or int(pool.iloc[i]["nearest_bus_id"]) not in used_buses)), None)
        if pick is None:
            raise RuntimeError(f"Only {len(chosen)} candidates satisfy the selection rule; relax pool_score_quantile or connection_radius_m.")
        chosen.append(pick)
        used_buses.add(int(pool.iloc[pick]["nearest_bus_id"]))
        min_dist = np.minimum(min_dist, np.linalg.norm(pool_xy - pool_xy[pick], axis=1))
    selected = pool.iloc[chosen].sort_values("nearest_bus_id").reset_index(drop=True)
    selected.insert(0, "candidate_id", [f"A-{i:02d}" for i in range(1, len(selected) + 1)])
    selected = selected.rename(columns={"nearest_bus_id": "linked_bus_id", "nearest_bus_distance_m": "connection_distance_m"})
    selected["selection_pool_threshold"] = threshold
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=hexes.crs)


# ---------------------------------------------------------- wildfire corridors


def _corridor_exposure(
    lines: gpd.GeoDataFrame, masks: dict[str, np.ndarray], transform: Any, buffer_m: float, weights: dict[str, float],
) -> pd.DataFrame:
    shape_ = masks["burned"].shape
    burned, fuel = [], []
    for geom in lines.geometry:
        corridor = geom.buffer(buffer_m)
        burned.append(_mask_fraction(masks["burned"], corridor, transform, shape_))
        fuel.append(_mask_fraction(masks["fuel"], corridor, transform, shape_))
    result = pd.DataFrame({"corridor_burned_fraction": burned, "corridor_fuel_fraction": fuel}, index=lines.index)
    result["wildfire_exposure_score"] = np.clip(
        float(weights["burned"]) * result["corridor_burned_fraction"] + float(weights["fuel"]) * result["corridor_fuel_fraction"], 0, 1,
    )
    return result


def _select_contingencies(config: dict[str, Any], branches: gpd.GeoDataFrame) -> pd.DataFrame:
    rule = config["contingencies"]["n1_selection"]
    if rule["kind"] != "wildfire_exposure_top_k":
        raise NotImplementedError(rule["kind"])
    graph = nx.MultiGraph()
    for row in branches.loc[branches["in_service"]].itertuples():
        graph.add_edge(int(row.from_bus), int(row.to_bus), key=row.branch_id)
    pool = branches.loc[branches["in_service"]]
    if rule.get("exclude_transformers", True):
        pool = pool.loc[pool["branch_type"].eq("line")]
    pool = pool.sort_values("wildfire_exposure_score", ascending=False)
    rows = [{"contingency_id": "n0", "kind": "none", "branch_id": pd.NA, "pfdelta_branch_index": pd.NA, "from_bus": pd.NA, "to_bus": pd.NA, "wildfire_exposure_score": np.nan, "selection_reason": "intact network"}]
    for row in pool.itertuples():
        if len(rows) > int(rule["k"]):
            break
        trial = graph.copy()
        trial.remove_edge(int(row.from_bus), int(row.to_bus), key=row.branch_id)
        if not nx.is_connected(trial):
            continue
        rows.append({
            "contingency_id": f"n1_{int(row.pfdelta_branch_index):03d}", "kind": "branch_outage", "branch_id": row.branch_id,
            "pfdelta_branch_index": int(row.pfdelta_branch_index), "from_bus": int(row.from_bus), "to_bus": int(row.to_bus),
            "wildfire_exposure_score": float(row.wildfire_exposure_score),
            "selection_reason": f"rank {len(rows)} corridor wildfire exposure among connectivity-preserving lines",
        })
    if len(rows) != int(rule["k"]) + 1:
        raise RuntimeError("Could not find enough connectivity-preserving exposed lines for N-1.")
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- driver


def build_case_study_layers(
    config_path: Path, output_directory: Path, raw_geodata_dir: Path, pfdelta_raw_directory: Path, pfdelta_scenario_id: int,
) -> dict[str, int]:
    config = load_config(config_path)
    area = study_area_from_config(config)
    rasters_dir, vectors_dir = raw_geodata_dir / "rasters", raw_geodata_dir / "vectors"
    years = config["study_area"].get("burned_area_years", {"start": 2015, "end": 2024})
    landcover, transform = _read_raster(rasters_dir / "worldcover_2021_100m.tif")
    slope, _ = _read_raster(rasters_dir / "slope_deg_100m.tif")
    dem, _ = _read_raster(rasters_dir / "dem_100m.tif")
    burned, _ = _read_raster(rasters_dir / f"modis_burn_count_{years['start']}_{years['end']}_100m.tif")
    rasters = {"landcover": landcover, "slope": slope, "dem": dem, "burned": burned}
    protected = gpd.read_parquet(vectors_dir / "protected_areas_real.parquet").to_crs(area.metric_crs)
    settlements = gpd.read_parquet(vectors_dir / "osm_settlements.parquet").to_crs(area.metric_crs)
    substations = gpd.read_parquet(vectors_dir / "osm_substations_context.parquet").to_crs(area.metric_crs)
    pv_points = gpd.read_parquet(vectors_dir / "pvgis_yield_points.parquet").to_crs(area.metric_crs)

    buses, branches = _embed_network(config, area, landcover, transform)
    branches = _pfdelta_branch_index(branches, pfdelta_raw_directory, pfdelta_scenario_id)
    _, _, _, bounds = area.metric_grid()
    hexes = _hex_grid(bounds, float(config["candidate_sites"]["hex_circumradius_m"])).set_crs(area.metric_crs)
    hexes = hexes.loc[hexes.intersects(area.polygon_wgs84.to_crs(area.metric_crs).geometry.iloc[0])].reset_index(drop=True)
    hexes, masks = _score_hexes(config, area, hexes, rasters, transform, protected, settlements, pv_points)
    wildfire_cfg = config["candidate_sites"]["wildfire"]
    exposure = _corridor_exposure(branches, masks, transform, float(wildfire_cfg["corridor_buffer_m"]), wildfire_cfg["exposure_weights"])
    branches = pd.concat([branches, exposure], axis=1)
    branches["exposure_class"] = _exposure_class(branches["wildfire_exposure_score"])
    branches = gpd.GeoDataFrame(branches, geometry="geometry", crs=area.metric_crs)
    contingencies = _select_contingencies(config, branches)
    hexes = _attach_nearest_bus(hexes, buses)
    hexes["within_connection_radius"] = hexes["nearest_bus_distance_m"].le(float(config["candidate_sites"]["connection_radius_m"]))
    candidates = _select_candidates(config, hexes, buses)
    bus_lookup = buses.set_index("bus_id").geometry
    connections = gpd.GeoDataFrame({
        "candidate_id": candidates["candidate_id"], "linked_bus_id": candidates["linked_bus_id"],
        "geometry": [LineString([Point(cx, cy), bus_lookup.loc[int(b)]]) for cx, cy, b in zip(candidates["cx"], candidates["cy"], candidates["linked_bus_id"])],
    }, geometry="geometry", crs=area.metric_crs)
    corridor = _corridor_exposure(connections, masks, transform, float(wildfire_cfg["corridor_buffer_m"]), wildfire_cfg["exposure_weights"])
    candidates["corridor_wildfire_exposure"] = corridor["wildfire_exposure_score"].to_numpy()
    connections["wildfire_exposure_score"] = corridor["wildfire_exposure_score"].to_numpy()
    candidates["asset_type"] = str(config["case_study"]["asset"]["type"])
    candidates["battery_power_mw"] = float(config["case_study"]["asset"].get("battery_power_mw", 0))
    centre_wgs = gpd.GeoSeries([Point(x, y) for x, y in zip(candidates["cx"], candidates["cy"])], crs=area.metric_crs).to_crs("EPSG:4326")
    candidates["lon"], candidates["lat"] = centre_wgs.x, centre_wgs.y

    burned_shapes = [
        {"burn_count": int(value), "geometry": shape(geom)}
        for geom, value in rasterio.features.shapes((burned > 0).astype("uint8"), mask=burned > 0, transform=transform)
    ]
    burned_layer = gpd.GeoDataFrame(burned_shapes if burned_shapes else {"burn_count": [], "geometry": []}, geometry="geometry", crs=area.metric_crs)
    buffers = config["candidate_sites"]["exclusion_rules"]["settlement_buffer_m"]
    settlement_buffers = settlements.copy()
    settlement_buffers["geometry"] = [geom.buffer(float(buffers.get(place, 500))) for geom, place in zip(settlements.geometry, settlements["place"])]
    study_area = gpd.GeoDataFrame({"layer_name": ["study_area"]}, geometry=[box(*bounds)], crs=area.metric_crs)

    layers: dict[str, gpd.GeoDataFrame] = {
        "study_area": study_area, "buses": buses, "branches": branches, "hex_cells": hexes, "candidate_sites": candidates,
        "candidate_connections": connections, "protected_areas": protected, "burned_area": burned_layer,
        "settlements": settlements, "settlement_buffers": settlement_buffers, "osm_substations_context": substations,
        "pvgis_points": pv_points,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, layer in layers.items():
        layer.to_parquet(output_directory / f"{name}.parquet", index=False)
        layer.to_crs("EPSG:4326").to_file(output_directory / f"{name}.geojson", driver="GeoJSON")
    contingencies.to_csv(output_directory / "contingencies.csv", index=False)
    provenance = {}
    sources_path = raw_geodata_dir / "geodata_sources.json"
    if sources_path.exists():
        provenance = json.loads(sources_path.read_text(encoding="utf-8"))
    schematic = {"buses", "branches", "candidate_connections"}
    metadata = {
        "title": "GeoGridLens Alentejo case-study layers",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": area.metric_crs,
        "study_area_wgs84": list(area.bbox),
        "geographic_context": config["case_study"]["geographic_context"],
        "network_geometry_is_schematic": True,
        "geodata_is_real": True,
        "not_for_operational_or_geographic_use": True,
        "disclaimer": (
            "Terrain, land cover, protected areas, burned areas, solar resource and settlements are real open data. "
            "The IEEE-118 network is a benchmark laid out deterministically inside the study box; bus and line positions, "
            "the candidate-to-bus links and every power-flow result are illustrative and are not statements about Portugal's grid."
        ),
        "layers": {
            name: {
                "parquet": f"{name}.parquet", "geojson": f"{name}.geojson", "feature_count": int(len(layer)),
                "nature": "schematic network embedding" if name in schematic else ("derived from real geodata" if name in {"hex_cells", "candidate_sites", "settlement_buffers", "study_area"} else "real geodata"),
            }
            for name, layer in layers.items()
        },
        "contingencies": contingencies.to_dict(orient="records"),
        "geodata_sources": provenance.get("sources", []),
    }
    (output_directory / "spatial_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {name: int(len(layer)) for name, layer in layers.items()}
