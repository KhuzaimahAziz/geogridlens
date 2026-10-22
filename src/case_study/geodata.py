"""Download and cache real public geospatial layers for the Alentejo study area.

Every layer here is genuine geographic data from an open provider.  The
electrical network that is later drawn over these layers is still the IEEE-118
benchmark, so the *combination* is an illustrative embedding; the terrain,
land cover, protected areas, burned areas, solar resource and settlements are
real.  Each fetch is cached under ``data/raw/alentejo`` and documented in
``geodata_sources.json`` with URL, access date and resolution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

USER_AGENT = "GeoGridLens-case-study/0.2 (open research prototype)"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
NATURA_URL = (
    "https://bio.discomap.eea.europa.eu/arcgis/rest/services/ProtectedSites/"
    "Natura2000Sites/MapServer/{layer}/query"
)
CDDA_URL = (
    "https://bio.discomap.eea.europa.eu/arcgis/rest/services/ProtectedSites/"
    "CDDA_Dyna_WM/MapServer/4/query"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"

WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "herbaceous_wetland", 95: "mangroves", 100: "moss_lichen",
}


@dataclass(frozen=True)
class StudyArea:
    """A WGS84 bounding box plus the metric CRS used for every analysis grid."""

    west: float
    south: float
    east: float
    north: float
    metric_crs: str = "EPSG:32629"  # WGS 84 / UTM zone 29N covers Alentejo.
    resolution_m: float = 100.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def polygon_wgs84(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"layer_name": ["study_area"]}, geometry=[box(*self.bbox)], crs="EPSG:4326")

    def metric_grid(self) -> tuple[Any, int, int, tuple[float, float, float, float]]:
        """Return (transform, height, width, metric bounds) of the analysis grid."""

        left, bottom, right, top = transform_bounds("EPSG:4326", self.metric_crs, *self.bbox, densify_pts=21)
        step = self.resolution_m
        left, bottom = np.floor(left / step) * step, np.floor(bottom / step) * step
        right, top = np.ceil(right / step) * step, np.ceil(top / step) * step
        width = int(round((right - left) / step))
        height = int(round((top - bottom) / step))
        return from_origin(left, top, step, step), height, width, (left, bottom, right, top)


def study_area_from_config(config: dict[str, Any]) -> StudyArea:
    area = config["study_area"]
    return StudyArea(
        west=float(area["west"]), south=float(area["south"]),
        east=float(area["east"]), north=float(area["north"]),
        metric_crs=str(area.get("metric_crs", "EPSG:32629")),
        resolution_m=float(area.get("resolution_m", 100.0)),
    )


class SourceLog:
    """Collects provenance records so the dashboard can show them verbatim."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, **record: Any) -> None:
        record.setdefault("accessed_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.records.append(record)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({"sources": self.records}, indent=2), encoding="utf-8")


def _write_raster(path: Path, array: np.ndarray, transform: Any, crs: str, nodata: float | None, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate", tiled=True,
    ) as target:
        target.write(array.astype(dtype), 1)


def _stac_client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _warp_to_grid(
    source: np.ndarray, source_transform: Any, source_crs: Any, area: StudyArea,
    resampling: Resampling, nodata: float, dtype: str,
) -> np.ndarray:
    transform, height, width, _ = area.metric_grid()
    target = np.full((height, width), nodata, dtype=dtype)
    reproject(
        source=source, destination=target, src_transform=source_transform, src_crs=source_crs,
        src_nodata=nodata, dst_transform=transform, dst_crs=area.metric_crs, dst_nodata=nodata,
        resampling=resampling,
    )
    return target


def _clip_window(source: Any, bounds: tuple[float, float, float, float]) -> rasterio.windows.Window | None:
    window = from_bounds(*bounds, source.transform).round_offsets().round_lengths()
    window = window.intersection(rasterio.windows.Window(0, 0, source.width, source.height))
    if window.width <= 0 or window.height <= 0:
        return None
    return window


# --------------------------------------------------------------------------- DEM


def fetch_dem_and_slope(area: StudyArea, raster_dir: Path, log: SourceLog) -> tuple[Path, Path]:
    """Copernicus DEM GLO-30 mosaic, reprojected to the metric grid, plus slope."""

    dem_path, slope_path = raster_dir / "dem_100m.tif", raster_dir / "slope_deg_100m.tif"
    if dem_path.exists() and slope_path.exists():
        return dem_path, slope_path
    client = _stac_client()
    items = list(client.search(collections=["cop-dem-glo-30"], bbox=list(area.bbox), max_items=20).items())
    if not items:
        raise RuntimeError("No Copernicus DEM tiles found for the study area.")
    transform, height, width, _ = area.metric_grid()
    mosaic = np.full((height, width), np.nan, dtype="float32")
    for item in items:
        with rasterio.open(item.assets["data"].href) as source:
            window = _clip_window(source, area.bbox)
            if window is None:
                continue
            data = source.read(1, window=window).astype("float32")
            warped = _warp_to_grid(data, source.window_transform(window), source.crs, area, Resampling.bilinear, np.nan, "float32")
        mosaic = np.where(np.isnan(mosaic), warped, mosaic)
    if np.isnan(mosaic).all():
        raise RuntimeError("DEM mosaic is empty after warping.")
    filled = np.where(np.isnan(mosaic), np.nanmedian(mosaic), mosaic)
    gy, gx = np.gradient(filled, area.resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    _write_raster(dem_path, filled, transform, area.metric_crs, -9999.0, "float32")
    _write_raster(slope_path, slope, transform, area.metric_crs, -9999.0, "float32")
    log.add(
        layer="dem", title="Copernicus DEM GLO-30 (ESA / Airbus, via Microsoft Planetary Computer)",
        url=f"{STAC_URL}/collections/cop-dem-glo-30", native_resolution="30 m",
        derived=f"{int(area.resolution_m)} m DEM and slope (degrees)",
        license="Copernicus DEM licence (free, attribution required)", items=[item.id for item in items],
    )
    return dem_path, slope_path


# ------------------------------------------------------------------- land cover


def _worldcover_tiles(area: StudyArea) -> list[str]:
    tiles = set()
    for lat in range(int(np.floor(area.south / 3) * 3), int(np.ceil(area.north)), 3):
        for lon in range(int(np.floor(area.west / 3) * 3), int(np.ceil(area.east)), 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.add(f"{ns}{ew}")
    return sorted(tiles)


def fetch_landcover(area: StudyArea, raster_dir: Path, log: SourceLog) -> Path:
    """ESA WorldCover 2021 v200 resampled (majority class) to the analysis grid."""

    path = raster_dir / "worldcover_2021_100m.tif"
    if path.exists():
        return path
    transform, height, width, _ = area.metric_grid()
    mosaic = np.zeros((height, width), dtype="uint8")
    tiles = _worldcover_tiles(area)
    for tile in tiles:
        url = WORLDCOVER_URL.format(tile=tile)
        try:
            with rasterio.open(url) as source:
                window = _clip_window(source, area.bbox)
                if window is None:
                    continue
                # Read at roughly 30 m through the COG overviews, then warp to 100 m.
                out_shape = (max(1, int(window.height // 3)), max(1, int(window.width // 3)))
                data = source.read(1, window=window, out_shape=out_shape, resampling=Resampling.mode)
                src_transform = rasterio.transform.from_bounds(*source.window_bounds(window), out_shape[1], out_shape[0])
                warped = _warp_to_grid(data, src_transform, source.crs, area, Resampling.mode, 0, "uint8")
        except rasterio.errors.RasterioIOError:
            continue
        mosaic = np.where(mosaic == 0, warped, mosaic)
    if (mosaic == 0).all():
        raise RuntimeError("WorldCover mosaic is empty.")
    _write_raster(path, mosaic, transform, area.metric_crs, 0, "uint8")
    log.add(
        layer="land_cover", title="ESA WorldCover 10 m 2021 v200", url=WORLDCOVER_URL.format(tile="{tile}"),
        native_resolution="10 m", derived=f"{int(area.resolution_m)} m majority class", license="CC BY 4.0",
        tiles=tiles, classes=WORLDCOVER_CLASSES,
    )
    return path


# -------------------------------------------------------------- protected areas


def _arcgis_query(url: str, area: StudyArea, out_fields: str) -> gpd.GeoDataFrame:
    params = {
        "where": "1=1", "geometry": ",".join(str(v) for v in area.bbox), "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "outFields": out_fields, "returnGeometry": "true",
        "outSR": "4326", "f": "geojson", "resultRecordCount": "1000",
    }
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS query failed: {payload['error']}")
    features = payload.get("features", [])
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def fetch_protected_areas(area: StudyArea, vector_dir: Path, log: SourceLog) -> Path:
    """Natura 2000 (SAC + SPA) and nationally designated areas clipped to the box."""

    path = vector_dir / "protected_areas_real.parquet"
    if path.exists():
        return path
    frames = []
    for layer, label in ((0, "Natura 2000 Habitats Directive site (SCI/SAC)"), (1, "Natura 2000 Birds Directive site (SPA)")):
        frame = _arcgis_query(NATURA_URL.format(layer=layer), area, "SITECODE,SITENAME,SITETYPE")
        if len(frame):
            frame = frame.rename(columns={"SITECODE": "site_code", "SITENAME": "site_name"})
            frame["designation"] = label
            frame["source"] = "EEA Natura 2000 (2024)"
            frames.append(frame[["site_code", "site_name", "designation", "source", "geometry"]])
    cdda = _arcgis_query(CDDA_URL, area, "*")
    if len(cdda):
        code = next((c for c in cdda.columns if c.upper() in {"CDDA_ID", "SITE_CODE", "CDDAID"}), None)
        name = next((c for c in cdda.columns if c.upper() in {"SITE_NAME", "SITENAME", "NAME"}), None)
        desig = next((c for c in cdda.columns if "DESIG" in c.upper() or "IUCN" in c.upper()), None)
        frames.append(gpd.GeoDataFrame({
            "site_code": cdda[code].astype(str) if code else [f"cdda_{i}" for i in range(len(cdda))],
            "site_name": cdda[name].astype(str) if name else "Nationally designated area",
            "designation": ("National designation: " + cdda[desig].astype(str)) if desig else "Nationally designated area",
            "source": "EEA Nationally designated areas (CDDA)",
        }, geometry=cdda.geometry.values, crs="EPSG:4326"))
    if not frames:
        raise RuntimeError("No protected-area features returned.")
    protected = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    protected = gpd.clip(protected, area.polygon_wgs84).to_crs(area.metric_crs)
    protected = protected.loc[~protected.geometry.is_empty].reset_index(drop=True)
    protected["area_km2"] = protected.geometry.area / 1e6
    vector_dir.mkdir(parents=True, exist_ok=True)
    protected.to_parquet(path, index=False)
    log.add(
        layer="protected_areas", title="Natura 2000 (EEA, 2024) and Nationally designated areas (EEA CDDA)",
        url=[NATURA_URL.format(layer="0|1"), CDDA_URL], native_resolution="vector",
        license="EEA standard re-use policy (free, attribution)", feature_count=int(len(protected)),
    )
    return path


# ----------------------------------------------------------------- burned area


def fetch_burned_area(area: StudyArea, raster_dir: Path, log: SourceLog, start_year: int, end_year: int) -> Path:
    """MODIS MCD64A1 monthly burned area: count of burn detections per 100 m cell."""

    path = raster_dir / f"modis_burn_count_{start_year}_{end_year}_100m.tif"
    if path.exists():
        return path
    client = _stac_client()
    items = list(client.search(
        collections=["modis-64A1-061"], bbox=list(area.bbox),
        datetime=f"{start_year}-01-01/{end_year}-12-31", max_items=1000,
    ).items())
    if not items:
        raise RuntimeError("No MODIS MCD64A1 items found.")
    transform, height, width, _ = area.metric_grid()
    count = np.zeros((height, width), dtype="int16")
    burned_months: list[str] = []
    for item in items:
        with rasterio.open(item.assets["Burn_Date"].href) as source:
            bounds = transform_bounds("EPSG:4326", source.crs, *area.bbox, densify_pts=21)
            window = _clip_window(source, bounds)
            if window is None:
                continue
            burn = source.read(1, window=window)
            if not (burn > 0).any():
                continue
            mask = (burn > 0).astype("uint8")
            warped = _warp_to_grid(mask, source.window_transform(window), source.crs, area, Resampling.nearest, 0, "uint8")
        if warped.any():
            burned_months.append(item.datetime.strftime("%Y-%m") if item.datetime else item.id)
            count += warped.astype("int16")
    _write_raster(path, count, transform, area.metric_crs, -1, "int16")
    log.add(
        layer="burned_area",
        title="MODIS/Terra+Aqua Burned Area Monthly L3 500 m (MCD64A1 v061, NASA LP DAAC via Planetary Computer)",
        url=f"{STAC_URL}/collections/modis-64A1-061", native_resolution="500 m",
        derived=f"count of monthly burn detections {start_year}-{end_year} on the {int(area.resolution_m)} m grid",
        license="NASA open data", months_with_fire_in_area=sorted(burned_months), item_count=len(items),
    )
    return path


# -------------------------------------------------------------- solar resource


def fetch_pvgis_points(area: StudyArea, vector_dir: Path, log: SourceLog, grid_size: int = 6) -> Path:
    """PVGIS yearly yield (kWh per kWp, optimal fixed tilt) on a regular point grid."""

    path = vector_dir / "pvgis_yield_points.parquet"
    if path.exists():
        return path
    lons = np.linspace(area.west, area.east, grid_size)
    lats = np.linspace(area.south, area.north, grid_size)
    records = []
    for lat in lats:
        for lon in lons:
            params = {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "peakpower": 1, "loss": 14, "optimalinclination": 1, "outputformat": "json"}
            payload = None
            for attempt in range(3):
                try:
                    response = requests.get(PVGIS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError):
                    time.sleep(2 * (attempt + 1))
            if payload is None:
                raise RuntimeError(f"PVGIS failed at {lat}, {lon}")
            totals = payload["outputs"]["totals"]["fixed"]
            records.append({
                "lat": float(lat), "lon": float(lon),
                "pv_yield_kwh_per_kwp": float(totals["E_y"]),
                "irradiation_kwh_m2": float(totals["H(i)_y"]),
                "optimal_tilt_deg": float(payload["inputs"]["mounting_system"]["fixed"]["slope"]["value"]),
                "radiation_db": payload["inputs"]["meteo_data"]["radiation_db"],
            })
            time.sleep(0.4)
    points = gpd.GeoDataFrame(
        records, geometry=gpd.points_from_xy([r["lon"] for r in records], [r["lat"] for r in records]), crs="EPSG:4326",
    ).to_crs(area.metric_crs)
    vector_dir.mkdir(parents=True, exist_ok=True)
    points.to_parquet(path, index=False)
    log.add(
        layer="solar_resource", title="PVGIS 5.3 PVcalc (JRC), SARAH3 radiation, optimal fixed tilt, 14 % system loss",
        url=PVGIS_URL, native_resolution=f"{grid_size}x{grid_size} point grid, interpolated to cells",
        license="JRC PVGIS free use", point_count=len(points),
    )
    return path


# ------------------------------------------------------------------------- OSM


def _overpass(query: str) -> dict[str, Any]:
    response = None
    for attempt in range(3):
        response = requests.post(OVERPASS_URL, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=180)
        if response.status_code == 200:
            return response.json()
        time.sleep(5 * (attempt + 1))
    assert response is not None
    response.raise_for_status()
    raise RuntimeError("Overpass unavailable")


def _points_frame(rows: list[dict[str, Any]], crs: str) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs="EPSG:4326")
    return frame.to_crs(crs)


def fetch_osm_context(area: StudyArea, vector_dir: Path, log: SourceLog) -> tuple[Path, Path]:
    """Settlements (for buffers) and real substations (context only) from OSM."""

    settlements_path = vector_dir / "osm_settlements.parquet"
    substations_path = vector_dir / "osm_substations_context.parquet"
    if settlements_path.exists() and substations_path.exists():
        return settlements_path, substations_path
    bbox = f"({area.south},{area.west},{area.north},{area.east})"
    place_filter = '["place"~"^(city|town|village)$"]'
    payload = _overpass(f"[out:json][timeout:170];(node{place_filter}{bbox};);out body;")
    rows = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        rows.append({
            "osm_id": element["id"], "name": tags.get("name", ""), "place": tags["place"],
            "population": pd.to_numeric(tags.get("population"), errors="coerce"),
            "lon": element["lon"], "lat": element["lat"],
        })
    settlements = _points_frame(rows, area.metric_crs)
    sub_filter = '["power"="substation"]'
    payload = _overpass(f"[out:json][timeout:170];(node{sub_filter}{bbox};way{sub_filter}{bbox};relation{sub_filter}{bbox};);out center;")
    rows = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        centre = element.get("center", element)
        if "lon" not in centre:
            continue
        rows.append({
            "osm_id": element["id"], "name": tags.get("name", ""), "operator": tags.get("operator", ""),
            "voltage": tags.get("voltage", ""), "substation": tags.get("substation", ""),
            "lon": centre["lon"], "lat": centre["lat"],
        })
    substations = _points_frame(rows, area.metric_crs)
    vector_dir.mkdir(parents=True, exist_ok=True)
    settlements.to_parquet(settlements_path, index=False)
    substations.to_parquet(substations_path, index=False)
    log.add(layer="settlements", title="OpenStreetMap places (city/town/village)", url=OVERPASS_URL, license="ODbL", feature_count=len(settlements))
    log.add(
        layer="osm_substations_context", title="OpenStreetMap power=substation (display context only; NOT the modelled network)",
        url=OVERPASS_URL, license="ODbL", feature_count=len(substations),
    )
    return settlements_path, substations_path


# --------------------------------------------------------------------- driver


def fetch_all(config: dict[str, Any], raw_dir: Path) -> dict[str, Path]:
    area = study_area_from_config(config)
    raster_dir, vector_dir = raw_dir / "rasters", raw_dir / "vectors"
    log = SourceLog()
    log_path = raw_dir / "geodata_sources.json"
    years = config["study_area"].get("burned_area_years", {"start": 2015, "end": 2024})
    paths: dict[str, Path] = {}

    def persist() -> None:
        if not log.records:
            return
        existing = json.loads(log_path.read_text(encoding="utf-8"))["sources"] if log_path.exists() else []
        merged = {record["layer"]: record for record in existing}
        merged.update({record["layer"]: record for record in log.records})
        SourceLog.write(type("_", (), {"records": list(merged.values())})(), log_path)

    paths["dem"], paths["slope"] = fetch_dem_and_slope(area, raster_dir, log)
    persist()
    paths["landcover"] = fetch_landcover(area, raster_dir, log)
    persist()
    paths["protected"] = fetch_protected_areas(area, vector_dir, log)
    persist()
    paths["burned"] = fetch_burned_area(area, raster_dir, log, int(years["start"]), int(years["end"]))
    persist()
    paths["pvgis"] = fetch_pvgis_points(area, vector_dir, log)
    persist()
    paths["settlements"], paths["substations"] = fetch_osm_context(area, vector_dir, log)
    persist()
    paths["sources"] = log_path
    return paths
