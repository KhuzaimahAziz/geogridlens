"""Path-traced 3D view of the study area with forge3d.

Real Copernicus terrain at 30 m with real ESA WorldCover as the surface
colour, then the analysis layers burned into the same 30 m colour map:
protected areas (tinted), MODIS burn scars, the schematic IEEE-118 lines
coloured by wildfire-consequence class, candidate cells coloured by final
score, the modelled buses, and the real OSM transmission substations shown
as context only. Rendered with forge3d's fused GPU path tracer from an
oblique pinhole camera.

Writes ``outputs/figures/terrain_3d_oblique.png`` (south-west view) and
``terrain_3d_northeast.png``, with copies in ``docs/figures``.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from case_study.geodata import SourceLog, StudyArea, fetch_dem_and_slope, fetch_landcover, study_area_from_config  # noqa: E402
from case_study.manifest import load_config  # noqa: E402

CONFIG = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
RENDER_RES_M = 30.0
RASTERS = PROJECT_ROOT / CONFIG["outputs"]["raw_geodata_directory"] / "rasters_30m"
SPATIAL = PROJECT_ROOT / CONFIG["outputs"]["spatial_directory"]
IND = PROJECT_ROOT / CONFIG["outputs"]["indicators_directory"]
FIG = PROJECT_ROOT / "outputs" / "figures"

LANDCOVER_RGB = {
    10: (52, 104, 60), 20: (158, 146, 84), 30: (206, 198, 150), 40: (226, 206, 132),
    50: (124, 96, 96), 60: (200, 192, 174), 80: (64, 106, 170), 90: (108, 160, 168),
}
CLASSES = {
    "protected": ("Protected area (Natura 2000 / national), tinted", (40, 130, 70)),
    "burned": ("Burned 2015-2024 (MODIS)", (120, 40, 160)),
    "line_low": ("IEEE-118 line: low priority", (70, 70, 75)),
    "line_mid": ("IEEE-118 line: medium priority", (245, 150, 35)),
    "line_high": ("IEEE-118 line: high exposure x high consequence", (190, 20, 20)),
    "cand_constrained": ("Candidate: constrained", (255, 236, 170)),
    "cand_conditional": ("Candidate: conditionally suitable", (120, 200, 90)),
    "cand_priority": ("Candidate: priority for deeper study", (8, 95, 45)),
    "bus": ("Modelled bus (IEEE-118, schematic)", (15, 15, 15)),
    "substation": ("Real REN transmission substation (OpenStreetMap, context only)", (255, 255, 255)),
}


def ensure_render_rasters() -> tuple[Path, Path]:
    area = study_area_from_config(CONFIG)
    area = StudyArea(area.west, area.south, area.east, area.north, area.metric_crs, RENDER_RES_M)
    log = SourceLog()
    dem, _ = fetch_dem_and_slope(area, RASTERS, log)
    landcover = fetch_landcover(area, RASTERS, log)
    return dem, landcover


def build_albedo(transform, shape, landcover: np.ndarray) -> np.ndarray:
    rgb = np.zeros(shape + (3,), dtype=np.float32)
    for code, colour in LANDCOVER_RGB.items():
        rgb[landcover == code] = colour
    rgb[landcover == 0] = (190, 185, 175)

    def mask(shapes) -> np.ndarray:
        return rasterio.features.rasterize(((g, 1) for g in shapes), out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)

    branches = gpd.read_parquet(SPATIAL / "branches.parquet")
    hazard = pd.read_parquet(IND / "branch_hazard_consequence_scores.parquet")
    branches = branches.merge(hazard[["branch_id", "exposure_tercile", "consequence_tercile"]], on="branch_id")
    final = pd.read_parquet(IND / "candidate_final_scores.parquet")
    final = final.loc[final["weight_profile"].eq("balanced")]
    candidates = gpd.read_parquet(SPATIAL / "candidate_sites.parquet").merge(final[["candidate_id", "classification"]], on="candidate_id")
    protected = gpd.read_parquet(SPATIAL / "protected_areas.parquet")
    burned = gpd.read_parquet(SPATIAL / "burned_area.parquet")
    buses = gpd.read_parquet(SPATIAL / "buses.parquet")
    substations = gpd.read_parquet(SPATIAL / "osm_substations_context.parquet")
    substations = substations.loc[substations["substation"].eq("transmission")]

    tint = mask(protected.geometry)
    rgb[tint] = 0.55 * rgb[tint] + 0.45 * np.asarray(CLASSES["protected"][1], dtype=np.float32)
    rgb[mask(g.buffer(60) for g in burned.geometry)] = CLASSES["burned"][1]

    def line_key(row) -> str:
        if row.exposure_tercile == "high" and row.consequence_tercile == "high":
            return "line_high"
        if "high" in (row.exposure_tercile, row.consequence_tercile) or row.exposure_tercile == row.consequence_tercile == "medium":
            return "line_mid"
        return "line_low"

    for key, width in (("line_low", 110), ("line_mid", 130), ("line_high", 170)):
        rows = [r.geometry.buffer(width) for r in branches.itertuples() if line_key(r) == key]
        if rows:
            rgb[mask(rows)] = CLASSES[key][1]
    cand_key = {"Constrained": "cand_constrained", "Conditionally suitable": "cand_conditional", "Priority for deeper study": "cand_priority"}
    for label, key in cand_key.items():
        rows = [g.buffer(150) for g, c in zip(candidates.geometry, candidates["classification"]) if c == label]
        if rows:
            rgb[mask(rows)] = CLASSES[key][1]
            rgb[mask(g.exterior.buffer(90) for g in rows)] = (20, 20, 20)
    connections = gpd.read_parquet(SPATIAL / "candidate_connections.parquet")
    rgb[mask(g.buffer(60) for g in connections.geometry)] = (25, 25, 25)
    rgb[mask(g.buffer(260) for g in buses.geometry)] = CLASSES["bus"][1]
    rgb[mask(g.buffer(560, cap_style=3) for g in substations.geometry)] = (30, 30, 30)
    rgb[mask(g.buffer(420, cap_style=3) for g in substations.geometry)] = CLASSES["substation"][1]
    return np.concatenate([np.clip(rgb, 0, 255).astype(np.uint8), np.full(shape + (1,), 255, dtype=np.uint8)], axis=-1)


def annotate(path: Path, title: str, legend_keys: list[str], footer: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGBA")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 38)
        small = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    draw.rectangle([0, 0, w, 82], fill=(12, 16, 24, 190))
    draw.text((36, 20), title, fill=(245, 245, 245, 255), font=font)
    items = [(CLASSES[k][1], CLASSES[k][0]) for k in legend_keys]
    box_h = 36 * len(items) + 28
    draw.rectangle([24, h - box_h - 70, 760, h - 70], fill=(12, 16, 24, 200))
    y = h - box_h - 56
    for colour, label in items:
        draw.rectangle([40, y, 68, y + 24], fill=colour + (255,), outline=(240, 240, 240, 255))
        draw.text((82, y - 3), label, fill=(235, 235, 235, 255), font=small)
        y += 36
    draw.rectangle([0, h - 56, w, h], fill=(12, 16, 24, 190))
    draw.text((36, h - 44), footer, fill=(205, 205, 205, 255), font=small)
    image.convert("RGB").save(path, optimize=True)


def main() -> None:
    import forge3d
    from forge3d import FusedCamera, FusedTerrain, render_fused

    dem_path, landcover_path = ensure_render_rasters()
    with rasterio.open(dem_path) as src:
        heights = src.read(1).astype(np.float32)
        transform, shape = src.transform, (src.height, src.width)
    with rasterio.open(landcover_path) as src:
        landcover = src.read(1)
    albedo = build_albedo(transform, shape, landcover)
    heights = heights - float(np.nanmin(heights))
    rows, cols = heights.shape
    width_m, depth_m = cols * RENDER_RES_M, rows * RENDER_RES_M
    exaggeration = 8.0
    terrain = FusedTerrain(heights=heights, spacing=(RENDER_RES_M, RENDER_RES_M), exaggeration=exaggeration, albedo_map=albedo, albedo_sampling="nearest")
    # forge3d fused frame: x east, y up, z south (row 0 is the north edge).
    views = {
        "terrain_3d_oblique": {"origin": (-0.12 * width_m, 0.34 * depth_m, 0.80 * depth_m), "look_at": (0.03 * width_m, 0.0, -0.02 * depth_m), "sun": (300.0, 38.0)},
        "terrain_3d_northeast": {"origin": (0.80 * width_m, 0.34 * depth_m, -0.72 * depth_m), "look_at": (-0.03 * width_m, 0.0, 0.03 * depth_m), "sun": (120.0, 42.0)},
    }
    FIG.mkdir(parents=True, exist_ok=True)
    legend_keys = ["protected", "burned", "line_low", "line_mid", "line_high", "cand_constrained", "cand_conditional", "cand_priority", "bus", "substation"]
    footer = "Copernicus DEM 30 m (8x vertical exaggeration), ESA WorldCover 2021, EEA Natura 2000/CDDA, NASA MODIS MCD64A1, OpenStreetMap. Path-traced with forge3d. Grid geometry is schematic."
    only_candidates = "--candidates-only" in sys.argv
    for name, view in ({} if only_candidates else views).items():
        started = time.time()
        camera = FusedCamera(origin=view["origin"], look_at=view["look_at"], fov_y_deg=40.0)
        image = render_fused(terrain=terrain, camera=camera, width=2400, height=1500, samples=40, sun_azimuth_deg=view["sun"][0], sun_elevation_deg=view["sun"][1],
                             sun_intensity=2.8, sky_intensity=0.6, exposure=1.3, seed=3, tile=(800, 750))
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        path = FIG / f"{name}.png"
        forge3d.numpy_to_png(str(path), image)
        annotate(path, "Alentejo study area: real terrain and land cover, schematic IEEE-118 grid, GENCO screening results", legend_keys, footer)
        print(f"{name}: {time.time() - started:.0f} s")
    render_candidate_closeups(heights, albedo, transform, render_fused, FusedCamera, FusedTerrain, forge3d)
    docs_figures = PROJECT_ROOT / "docs" / "figures"
    if docs_figures.exists():
        for name in list(views) + ["terrain_3d_candidates"]:
            if (FIG / f"{name}.png").exists():
                shutil.copy(FIG / f"{name}.png", docs_figures / f"{name}.png")
        (docs_figures / "terrain_relief_map.png").unlink(missing_ok=True)
    (FIG / "terrain_relief_map.png").unlink(missing_ok=True)


def render_candidate_closeups(heights, albedo, transform, render_fused, FusedCamera, FusedTerrain, forge3d, count: int = 3, half_window_m: float = 11_000.0) -> None:
    """One oblique close-up per top-ranked candidate, composed side by side."""

    from PIL import Image, ImageDraw, ImageFont

    final = pd.read_parquet(IND / "candidate_final_scores.parquet")
    balanced = final.loc[final["weight_profile"].eq("balanced")].dropna(subset=["rank"]).sort_values("rank").head(count)
    candidates = gpd.read_parquet(SPATIAL / "candidate_sites.parquet").set_index("candidate_id")
    metrics = pd.read_parquet(IND / "scenario_metrics.parquet").set_index("scenario_id")
    panels = []
    for row in balanced.itertuples():
        site = candidates.loc[row.candidate_id]
        cx, cy = float(site["cx"]), float(site["cy"])
        r0, c0 = rasterio.transform.rowcol(transform, cx - half_window_m, cy + half_window_m)
        r1, c1 = rasterio.transform.rowcol(transform, cx + half_window_m, cy - half_window_m)
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, heights.shape[0]), min(c1, heights.shape[1])
        z = heights[r0:r1, c0:c1]
        z = z - float(np.nanmin(z))
        a = albedo[r0:r1, c0:c1]
        rows, cols = z.shape
        width_m, depth_m = cols * RENDER_RES_M, rows * RENDER_RES_M
        terrain = FusedTerrain(heights=z, spacing=(RENDER_RES_M, RENDER_RES_M), exaggeration=3.0, albedo_map=np.ascontiguousarray(a), albedo_sampling="nearest")
        camera = FusedCamera(origin=(-0.10 * width_m, 0.30 * depth_m, 0.68 * depth_m), look_at=(0.0, 0.0, -0.10 * depth_m), fov_y_deg=36.0)
        image = render_fused(terrain=terrain, camera=camera, width=1200, height=1000, samples=48, sun_azimuth_deg=300.0, sun_elevation_deg=40.0,
                             sun_intensity=2.8, sky_intensity=0.6, exposure=1.3, seed=5, tile=(600, 500))
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        panel = Image.fromarray(image[..., :3]).convert("RGBA")
        m = metrics.loc[f"{row.candidate_id}__100mw__normal_demand__n0"]
        panels.append((panel, row, m))
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()
    pw, ph = panels[0][0].size
    header, footer = 150, 84
    canvas = Image.new("RGB", (pw * len(panels), ph + header + footer), (12, 16, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, row, m) in enumerate(panels):
        x = i * pw
        canvas.paste(panel.convert("RGB"), (x, header))
        draw.text((x + 24, 18), f"#{int(row.rank)}  {row.candidate_id}  ->  bus {int(row.linked_bus_id)}  ({row.connection_distance_m / 1000:.1f} km)", fill=(245, 245, 245), font=font)
        draw.text((x + 24, 62), f"spatial {row.spatial_score:.2f}   grid {row.grid_score:.2f}   resilience {row.resilience_score:.2f}   final {row.final_score:.2f}", fill=(220, 220, 220), font=small)
        draw.text((x + 24, 94), f"100 MW, normal demand: max loading change {m.max_loading_delta:+.2f} on {m.worst_branch_id}", fill=(200, 200, 200), font=small)
        draw.text((x + 24, 120), f"{row.classification}", fill=(140, 220, 150), font=small)
    legend = "Hexagon = candidate cell (dark green = priority); dashed black = connector to its bus; black dots = IEEE-118 buses; orange/red = lines by consequence class; white squares = real REN substations (context); green tint = protected land; purple = burned 2015-2024."
    draw.text((24, ph + header + 16), legend, fill=(205, 205, 205), font=small)
    draw.text((24, ph + header + 42), f"~{int(2 * half_window_m / 1000)} km windows, Copernicus DEM 30 m with 3x vertical exaggeration, ESA WorldCover 2021. Path-traced with forge3d. Grid geometry is schematic.", fill=(160, 160, 160), font=small)
    canvas.save(FIG / "terrain_3d_candidates.png", optimize=True)
    print("terrain_3d_candidates: done")


if __name__ == "__main__":
    main()
