"""Presentation visuals: the animated capacity sweep and the land-vs-grid pair.

* ``capacity_sweep.gif``  - one good and one poor site, 0 to 200 MW in 10 MW
  steps, lines coloured by GENCO loading change (the "what GENCO sees" hero).
* ``same_land_different_grid.png`` - the best-land site and the best-grid site
  side by side: identical spatial quality, very different grid response.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from case_study.manifest import load_config  # noqa: E402

CONFIG = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
SPATIAL = PROJECT_ROOT / CONFIG["outputs"]["spatial_directory"]
CASE = PROJECT_ROOT / CONFIG["outputs"]["root"]
IND = PROJECT_ROOT / CONFIG["outputs"]["indicators_directory"]
FIG = PROJECT_ROOT / "outputs" / "figures"
SWEEP = CASE / "capacity_sweep"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.titleweight": "semibold", "figure.facecolor": "white"})


def _layers():
    return {name: gpd.read_parquet(SPATIAL / f"{name}.parquet") for name in ["hex_cells", "buses", "branches", "candidate_sites", "candidate_connections", "protected_areas", "study_area"]}


def _base(ax, layers, title=None):
    hexes = layers["hex_cells"]
    hexes.plot(column="spatial_opportunity_score", cmap="YlOrBr", ax=ax, edgecolor="none", vmin=0, vmax=1, alpha=0.45)
    hexes[hexes["spatially_excluded"]].plot(color="#d9d9d9", ax=ax, alpha=0.9)
    layers["protected_areas"].boundary.plot(ax=ax, color="#1b7837", linewidth=0.5)
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left")


def _network(ax, layers, branch_values: pd.Series, bus_values: pd.Series, vmax_l: float, site: gpd.GeoDataFrame | None = None):
    branches = layers["branches"].merge(branch_values.rename("value"), left_on="branch_id", right_index=True, how="left")
    segments = [np.asarray(geom.coords) for geom in branches.geometry]
    norm = TwoSlopeNorm(vmin=-vmax_l, vcenter=0, vmax=vmax_l)
    widths = 0.8 + 6 * np.clip(branches["value"].abs().fillna(0) / vmax_l, 0, 1)
    collection = LineCollection(segments, cmap="RdBu_r", norm=norm, linewidths=widths, zorder=3)
    collection.set_array(branches["value"].fillna(0).to_numpy())
    ax.add_collection(collection)
    buses = layers["buses"].merge(bus_values.rename("value"), left_on="bus_id", right_index=True, how="left")
    ax.scatter(buses.geometry.x, buses.geometry.y, c=buses["value"].fillna(0), cmap="PuOr_r", norm=Normalize(-0.02, 0.02), s=14, edgecolors="#222", linewidths=0.4, zorder=4)
    if site is not None:
        site.plot(ax=ax, facecolor="#ffd54f", edgecolor="#111", linewidth=2, zorder=5)
    return collection


def capacity_gif(layers):
    sweep = pd.read_csv(SWEEP / "capacity_sweep.csv")
    sites = sweep["candidate_id"].unique().tolist()
    capacities = sorted(sweep["capacity_mw"].unique())
    frames = []
    bounds = layers["study_area"].total_bounds
    for capacity in capacities:
        fig, axes = plt.subplots(1, len(sites), figsize=(7.2 * len(sites), 7.4))
        for ax, candidate in zip(np.atleast_1d(axes), sites):
            sid = f"{candidate}__{int(capacity):03d}mw"
            branch = pd.read_csv(SWEEP / sid / "branch_results.csv").set_index("branch_id")
            bus = pd.read_csv(SWEEP / sid / "bus_results.csv").set_index("bus_id")
            row = sweep.loc[sweep["scenario_id"].eq(sid)].iloc[0]
            site = layers["candidate_sites"].loc[layers["candidate_sites"]["candidate_id"].eq(candidate)]
            _base(ax, layers)
            _network(ax, layers, branch["loading_delta"], bus["voltage_delta_pu"], 0.5, site)
            layers["candidate_connections"].loc[layers["candidate_connections"]["candidate_id"].eq(candidate)].plot(ax=ax, color="#111", linewidth=1.2, linestyle="--", zorder=5)
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            status = "within limits" if row.max_loading_delta < 0.20 else "beyond +0.20 loading-change limit"
            ax.set_title(f"{candidate} at bus {int(row.bus_id)}: {int(capacity)} MW solar\nmax loading change {row.max_loading_delta:+.2f}  |  {status}", loc="left", fontsize=12)
        fig.suptitle("What GENCO sees as a solar plant grows: line colour = change in loading vs. no-plant baseline (blue down, red up), width = size of change", fontsize=12, y=0.995)
        fig.text(0.5, 0.01, "GENCO neural power flow, ~44 ms per frame on a laptop CPU. IEEE-118 benchmark embedded in real Alentejo land-cover and protected-area data; grid geometry is schematic.", ha="center", fontsize=9, color="#555")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        frame = frame.resize((frame.width * 3 // 5, frame.height * 3 // 5), Image.LANCZOS).quantize(colors=160, method=Image.MEDIANCUT)
        frames.append(frame)
        plt.close(fig)
    hold = [frames[0]] * 4 + frames + [frames[-1]] * 10
    hold[0].save(FIG / "capacity_sweep.gif", save_all=True, append_images=hold[1:], duration=220, loop=0, optimize=False)
    frames[-1].convert("RGB").save(FIG / "capacity_sweep_200mw.png")


def land_vs_grid(layers):
    final = pd.read_parquet(IND / "candidate_final_scores.parquet")
    balanced = final.loc[final["weight_profile"].eq("balanced")].dropna(subset=["rank"])
    best_land = balanced.sort_values("spatial_score", ascending=False).iloc[0]
    best_grid = balanced.sort_values("grid_score", ascending=False).iloc[0]
    metrics = pd.read_parquet(IND / "scenario_metrics.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    bounds = layers["study_area"].total_bounds
    for ax, row, label in zip(axes, (best_land, best_grid), ("Best land in the set", "Best grid response in the set")):
        sid = f"{row.candidate_id}__100mw__normal_demand__n0"
        branch = pd.read_csv(CASE / "genco_results" / sid / "branch_results.csv").set_index("branch_id")
        bus = pd.read_csv(CASE / "genco_results" / sid / "bus_results.csv").set_index("bus_id")
        m = metrics.set_index("scenario_id").loc[sid]
        site = layers["candidate_sites"].loc[layers["candidate_sites"]["candidate_id"].eq(row.candidate_id)]
        _base(ax, layers)
        _network(ax, layers, branch["loading_delta"], bus["voltage_delta_pu"], 0.3, site)
        cx, cy = float(site["cx"].iloc[0]), float(site["cy"].iloc[0])
        half = 30_000
        x0 = min(max(cx - half, bounds[0]), bounds[2] - 2 * half)
        y0 = min(max(cy - half, bounds[1]), bounds[3] - 2 * half)
        ax.set_xlim(x0, x0 + 2 * half)
        ax.set_ylim(y0, y0 + 2 * half)
        ax.set_title(
            f"{label}: {row.candidate_id} (bus {int(row.linked_bus_id)})\n"
            f"spatial {row.spatial_score:.2f} · grid {row.grid_score:.2f} · final rank {int(row['rank'])}\n"
            f"100 MW, normal demand: max loading change {m.max_loading_delta:+.2f} on {m.worst_branch_id}", loc="left", fontsize=11,
        )
    fig.suptitle("Same land quality, different grid: why the spatial score alone cannot pick the site", fontsize=14, y=0.99)
    fig.text(0.5, 0.01, "Background: real spatial-opportunity cells (grey = excluded). Lines: GENCO loading change for a 100 MW plant at each site (blue down, red up). 60 km window around each site.", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG / "same_land_different_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    layers = _layers()
    land_vs_grid(layers)
    if (SWEEP / "capacity_sweep.csv").exists():
        capacity_gif(layers)
    for name in ["same_land_different_grid.png", "capacity_sweep.gif", "capacity_sweep_200mw.png"]:
        source = FIG / name
        if source.exists() and (PROJECT_ROOT / "docs" / "figures").exists():
            (PROJECT_ROOT / "docs" / "figures" / name).write_bytes(source.read_bytes())
    print("visuals written to", FIG)


if __name__ == "__main__":
    main()
